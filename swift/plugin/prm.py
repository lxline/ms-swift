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


# TODO: read from file
LLM_as_judge_SYS = """I will provide a math problem along with a solution. They will be formatted as follows:

[Math Problem]

<math_problem>
...(math problem)...
</math_problem>

[Solution]

<paragraph_1>
...(paragraph 1 of solution)...
</paragraph_1>

...

<paragraph_n>
...(paragraph n of solution)...
</paragraph_n>

Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. You need to provide the analyses and the conclusion in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

...

<analysis_n>
...(analysis of paragraph n)...
</analysis_n>

<conclusion>
Correct/Incorrect
</conclusion>

* When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please elaborate on the analysis process carefully.

* If an error is detected in any paragraph, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach. Once a paragraph is found to contain any error, stop further analysis of subsequent paragraphs (as they may depend on the identified error) and directly provide the conclusion of "Incorrect."

For instance, given a solution of five paragraphs, if an error is found in the third paragraph, you should reply in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

<analysis_2>
...(analysis of paragraph 2)...
</analysis_2>

<analysis_3>
...(analysis of paragraph 3; since an error is found here, also provide detailed critique and correction guideline)...
</analysis_3>

<conclusion>
Incorrect
</conclusion>

Note that the analyses of paragraphs 4 and 5 should be skipped as the paragraph3 has been found to contain an error.

* Respond with your analyses and conclusion directly.
"""

LLM_as_judge_QUERY = """The following is the math problem and the solution for you task:

[Math Problem]

{tagged_problem}

[Solution]

{tagged_response}
"""


class ClientPRM(PRM):

    def __init__(self, api_key=None, base_url=None, model=None):
        from swift.llm import InferClient
        import os
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
                 ground_truths: List[str] = None,
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

            query = deepcopy(LLM_as_judge_QUERY)

            query = query.replace('{tagged_problem}', tagged_problem)
            query = query.replace('{tagged_response}', tagged_response)
            messages = [
                {
                    'role': 'system',
                    'content': LLM_as_judge_SYS
                },
                {
                    'role': 'user',
                    'content': query
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


class QwenPRM(PRM):

    def __init__(self, model='Qwen/Qwen2.5-Math-PRM-7B', model_type='qwen2_5_prm'):
        from swift.llm import PtEngine
        self.engine = PtEngine(model, model_type=model_type)

    def __call__(self,
                 infer_requests: List[Union[InferRequest, Dict]],
                 ground_truths: List[str] = None,
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
            content = response.choices[0].message.content
            try:
                float_list = json.loads(content)

                if isinstance(float_list, list) and all(isinstance(x, (int, float)) for x in float_list):
                    rewards.append(float_list[-1])
                else:
                    raise ValueError('wrong response')
            except Exception:
                rewards.append(0.)
        return rewards


prms = {
    'qwen_max': QwenMaxPRM,
    'client': ClientPRM,
    'qwen': QwenPRM,
}
