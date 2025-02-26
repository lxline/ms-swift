import os
from copy import deepcopy
from typing import Any, Dict, List, Union

import json

from swift.llm import InferRequest
from swift.utils import get_logger

logger = get_logger()


class PRM:

    def __call__(self, **kwargs) -> List[Any]:
        raise NotImplementedError


SYSTEM = """
You are a process reward model, give the reward value of the answer, you must follow the instructions below:

1. Output a float reward value between -1.0 and 1.0, -1.0 means the worst answer, 1.0 means the best answer, please think step by step to give your reasons and thoughts, but the reward must appare at the end with this format: **Reward: your-reward-value**.

2. The answer may be incomplete, you must give the reward by the existing part of the answer, taking into account semantic coherence, logical correctness, and clarity.

3. A ground truth answer will be given to you, it may be not the best one, consider it as a reference example.

Begin!
""" # noqa

QUERY = """
The original question or the previous conversation:

#query#

Here is the ground truth as the reference:

#ground_truth#

Given the upper information, give your reward(-1.0~1.0) of the following answer:

#response#
"""


class QwenMaxPRM(PRM):

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        # TODO: check request_config
        rewards = []

        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        for request, ground_truth in zip(infer_requests, ground_truths):
            previous = request['messages'][:-1]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert request['messages'][-1]['role'] == 'assistant'
            query = QUERY.replace('#query#', json.dumps(previous))
            query = query.replace('#ground_truth#', ground_truth)
            query = query.replace('#response#', request['messages'][-1]['content'])
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM
                },
                {
                    'role': 'user',
                    'content': query
                },
            ]
            completion = client.chat.completions.create(
                model='qwen-max',
                messages=messages,
            )

            content = completion.choices[0].message.content
            if 'Reward:' not in content:
                rewards.append(0.)
            else:
                try:
                    reward = float(content.split('Reward:')[1].strip().replace('*', ''))
                    rewards.append(reward)
                except Exception:
                    rewards.append(0.)

        return rewards


class LLMJudgePRM(PRM):
    """
    LLM as a judge.
    The Lessons of Developing Process Reward Models in Mathematical Reasoning, https://arxiv.org/abs/2501.07301
    """
    def __init__(self,
                 system,
                 query,
                 api_key=None,
                 base_url=None,
                 model=None):
        from swift.llm import InferClient
        import os

        if system is not None and system.endswith('.txt'):
            assert os.path.isfile(system), f'system: {system}'
            with open(system, 'r') as f:
                self.system = f.read()
        else:
            self.system = system

        if query is not None and query.endswith('.txt'):
            assert os.path.isfile(query), f'query: {query}'
            with open(query, 'r') as f:
                self.query = f.read()
        else:
            self.query = query

        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
        if base_url is None:
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        if model is None:
            model = 'qwen-max'

        self.infer_engine = InferClient(base_url=base_url, api_key=api_key)
        self.infer_kwargs = {
            'model': model,
            'use_tqdm': False,
        }

    def __call__(self,
                 infer_requests: List[Union[InferRequest, Dict]],
                 **kwargs) -> List[float]:
        prm_infer_requests = []
        request_config = kwargs.get('request_config')
        for request in infer_requests:
            previous = request.messages[:]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert previous[0]['role'] == 'user'
            assert previous[-1]['role'] == 'assistant'

            problem = previous[0]['content']
            tagged_problem = f'<math_problem>\n{problem}\n</math_problem>'
            steps = [x['content'] for x in previous[1:]]
            tagged_response = ''
            for i, step in enumerate(steps):
                tagged_response += f'<paragraph_{i + 1}>\n{step}\n</paragraph_{i + 1}>\n\n'

            query = deepcopy(self.query)

            query = query.replace('{tagged_problem}', tagged_problem)
            query = query.replace('{tagged_response}', tagged_response)
            messages = [
                {
                    'role': 'system',
                    'content': self.system,
                },
                {
                    'role': 'user',
                    'content': query,
                },
            ]

            prm_infer_requests.append(InferRequest(messages=messages))

        rewards = []
        try:
            responses = self.infer_engine.infer(prm_infer_requests, request_config=request_config, **self.infer_kwargs)
            for response in responses:
                content = response.choices[0].message.content
                if '<conclusion>' in content and 'Incorrect' in content:
                    rewards.append(0.)
                else:
                    rewards.append(1.)
        except Exception:
            for request in prm_infer_requests:
                try:
                    response = self.infer_engine.infer([request], request_config=request_config, **self.infer_kwargs)
                    content = response[0].choices[0].message.content
                    if '<conclusion>' in content and 'Incorrect' in content:
                        rewards.append(0.)
                    else:
                        rewards.append(1.)
                except Exception as e:
                    logger.info(f'Got exception: {e}. Not a valid reward, set to 0.')
                    rewards.append(0.)
        return rewards


class vLLMPRM(PRM):

    def __init__(self,
                 api_key=None,
                 base_url=None,
                 model='Qwen2.5-Math-PRM-7B',):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "User-Agent": "vLLMPRM Client",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def post_http_request(self, prm_requests: list):
        from concurrent.futures import ThreadPoolExecutor
        import requests

        def send_request(prm_request):
            response = requests.post(self.base_url, headers=self.headers, json=prm_request)
            return response

        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(send_request, prm_requests))

        return responses

    def __call__(self,
                 infer_requests: List[Union[InferRequest, Dict]],
                 **kwargs):
        prm_infer_requests = []
        system = 'Please reason step by step, and put your final answer within \\boxed{}.'
        for request in infer_requests:
            previous = request.messages[:]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert previous[0]['role'] == 'user'
            assert previous[-1]['role'] == 'assistant'

            request = {
                'model': self.model,
                'messages' : [{
                    'role': 'system',
                    'content': system
                }] + previous,
                'mm_processor_kwargs': None
            }
            prm_infer_requests.append(request)

        responses = self.post_http_request(prm_infer_requests)
        rewards = []
        for response in responses:
            try:
                content = response.content
                float_list = json.loads(content)['data'][0]['data']

                if (isinstance(float_list, list)
                    and all(isinstance(x, list) for x in float_list)
                    and all(isinstance(x, float) for sublist in float_list for x in sublist)):
                    rewards.append([_list[-1] for _list in float_list])
                else:
                    raise ValueError(f'Failed to parse Response: {response}')
            except Exception:
                rewards.append([0.])
        return rewards


class QwenPRM(PRM):

    def __init__(self, model='Qwen/Qwen2.5-Math-PRM-7B', model_type='qwen2_5_prm'):
        from swift.llm import PtEngine
        self.engine = PtEngine(model, model_type=model_type)

    def __call__(self,
                 infer_requests: List[Union[InferRequest, Dict]],
                 **kwargs) -> List[float]:
        prm_infer_requests = []
        system = 'Please reason step by step, and put your final answer within \\boxed{}.'
        for request in infer_requests:
            previous = request.messages[:]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert previous[0]['role'] == 'user'
            assert previous[-1]['role'] == 'assistant'

            problem = previous[0]['content']
            steps = [x['content'] for x in previous[1:]]

            messages = [{
                'role': 'system',
                'content': system
            }, {
                'role': 'user',
                'content': problem
            }, {
                'role': 'assistant',
                'content': '<extra_0>'.join(steps) + '<extra_0>'
            }]
            prm_infer_requests.append(InferRequest(messages=messages))

        responses = self.engine.infer(prm_infer_requests, **kwargs)
        rewards = []
        for response in responses:
            try:
                content = response.choices[0].message.content
                float_list = json.loads(content)

                if isinstance(float_list, list) and all(isinstance(x, (int, float)) for x in float_list):
                    rewards.append(float_list[-1])
                else:
                    raise ValueError(f'Failed to parse Response: {response}')
            except Exception:
                rewards.append(0.)
        return rewards


prms = {
    'qwen_max': QwenMaxPRM,
    'llm_judge': LLMJudgePRM,
    'vllm': vLLMPRM,
    'qwen': QwenPRM,
}