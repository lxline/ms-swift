import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from copy import deepcopy

import json
import numpy as np

import math
from typing import Literal, List

from swift.llm import InferRequest
from swift.llm.argument.sampling_args import SamplingArguments
from swift.llm.infer.protocol import UsageInfo
from swift.utils import get_logger
from .base import Sampler
from .utils import get_reward, perform_infer


logger = get_logger()


NXT_PROMPT = "Continue."

@dataclass
class Beam:
    current_texts: List[str] = None,
    rollout_texts: List[str] = None,
    current_scores: List[float] = None,
    rollout_scores: List[float] = None,
    outcome_score: float = 0.0,
    terminated: bool = False


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


class LanguageTree:
    def __init__(self,
                 query,
                 ground_truth,
                 prefix_messages,
                 beam,
                 config,
                 generator,
                 orm_model,
                 prm_model,):
        self.query = query
        self.ground_truth = ground_truth
        self.query_message = [{
            'role': 'user',
            'content': query,
        }]
        # self.prefix_messages = config.system_message + self.query_message
        self.prefix_messages = prefix_messages
        self.suffix_messages = [{
            'role': 'user',
            'content': NXT_PROMPT,
        }]
        self.config = config

        self.generator = generator
        self.orm_model = orm_model
        self.prm_model = prm_model

        self.usage_info = UsageInfo(0, 0, 0)
        self.step_index = 0
        self.answers = []
        self.beams = [beam]

    def update_usage_info(self, response):
        for key, value in self.usage_info.__dict__.items():
            update_value = getattr(response.usage, key, None) + value
            setattr(self.usage_info, key, update_value)

    def build(self):
        while len(self.answers) < self.config.num_return_sequences:
            gen_beams = [beam for beam in self.beams if not beam.terminated]
            next_beams = []
            for beam in gen_beams:
                next_beams += self.expand(beam)
            self.beams = self.rollout(next_beams)

    def expand(self, curr_beam):
        _config = self.config
        n = _config.num_return_sequences - len(self.answers)
        history_messages = []
        suffix_messages = []
        if self.step_index > 0:
            history_messages = [{
                'role': 'assistant',
                'content': step,
            } for step in curr_beam.current_texts]
            suffix_messages = self.suffix_messages
        infer_requests = [InferRequest(self.prefix_messages + history_messages + suffix_messages) for _ in range(n)]

        # e_time = time.time()
        # To perform the Expand operation in parallel,
        # there's no need to consider the order for now, since the Prompt is the same.
        expand_iter_index = 0
        while True:
            responses = perform_infer(self.generator, infer_requests, _config.expand_request_configs,
                                      **_config.infer_kwargs)
            if len(responses) > 0:
                break
            if expand_iter_index == 5:
                raise ValueError('Expand did not return any response')
            expand_iter_index += 1
        # logger.info(f"expand.expand time: {time.time() - e_time}")

        # To fetch Process Reward in parallel
        # e_time = time.time()
        unique_output = set()
        prm_infer_requests = []
        for response in responses:
            self.update_usage_info(response)
            output = response.choices[0].message.content.rstrip(_config.sep_token + '\n').split(_config.sep_token)[0]
            if output in unique_output:
                continue
            unique_output.add(output)
            infer_request = InferRequest(self.prefix_messages + history_messages
                                         + [{'role': 'assistant', 'content': output}]
                                         + suffix_messages)
            prm_infer_requests.append(infer_request)

        prm_score, _prm_mask = get_reward(
            self.prm_model,
            prm_infer_requests,
            threshold=_config.prm_threshold,
            normalize=False,
        )
        # logger.info(f"expand.prm time: {time.time() - e_time}")

        new_beams = []
        for output, score in zip(unique_output, prm_score):
            new_beam = Beam(
                current_texts=curr_beam.current_texts + [output],
                current_scores=curr_beam.current_scores + [score],
            )
            new_beams.append(new_beam)
        return new_beams

    def rollout(self, next_beams):
        _config = self.config
        index2rollout_beams = {}
        active_index = []
        for index, beam in enumerate(next_beams):
            if self.orm_model.check_terminate(beam.current_texts[-1])[0]:
                beam.terminated = True
            else:
                index2rollout_beams[index] = beam
                active_index.append(index)
        for i in range(_config.rollout_depth):
            if len(active_index) == 0:
                break
            infer_requests = []
            for index in active_index:
                beam = index2rollout_beams[index]
                history_messages = [{
                    'role': 'assistant',
                    'content': step,
                } for step in beam.current_texts] + [{
                    'role': 'user',
                    'content': step,
                } for step in beam.rollout_texts]
                infer_request = InferRequest(self.prefix_messages + history_messages + self.suffix_messages)
                infer_requests.append(infer_request)

            responses = perform_infer(self.generator, infer_requests, _config.expand_request_configs,
                                      **_config.infer_kwargs)

            prm_infer_requests = []
            for index, response in zip(active_index, responses):
                output = response.choices[0].message.content.rstrip(_config.sep_token + '\n').split(_config.sep_token)[0]
                beam = index2rollout_beams[index]
                beam.rollout_texts.append(output)
                history_messages = [{
                    'role': 'assistant',
                    'content': step,
                } for step in beam.current_texts] + [{
                    'role': 'user',
                    'content': step,
                } for step in beam.rollout_texts]
                infer_request = InferRequest(self.prefix_messages + history_messages)
                prm_infer_requests.append(infer_request)

            prm_score, _prm_mask = get_reward(
                self.prm_model,
                prm_infer_requests,
                threshold=_config.prm_threshold,
                normalize=False,
            )

            nxt_index = []
            for index, score in zip(active_index, prm_score):
                beam = index2rollout_beams[index]
                beam.rollout_scores.append(score)
                if not self.orm_model.check_terminate(beam.rollout_texts[-1])[0]:
                    nxt_index.append(index)
            active_index = nxt_index


class DvtsSampler(Sampler):

    def __init__(self, input_args: SamplingArguments):
        super().__init__(input_args)

    def _prepare_model_tokenizer(self):
        args = self.args
        self.infer_kwargs = {}
        if args.sampler_engine == 'client':
            from swift.llm import InferClient
            api_key = args.api_key
            base_url = args.base_url
            self.infer_engine = [
                InferClient(base_url=base_url, api_key=api_key) for _ in range(args.num_return_sequences)
            ]
            self.infer_kwargs['model'] = args.model
        else:
            _Engine = self.get_infer_engine()
            self.infer_engine = _Engine(self.args.model, model_type=self.args.model_type, **self.args.engine_kwargs)

    def get_infer_engine(self):
        if self.args.sampler_engine == 'pt':
            from swift.llm import PtEngine
            _Engine = PtEngine
        elif self.args.sampler_engine == 'vllm':
            from swift.llm import VllmEngine
            _Engine = VllmEngine
        elif self.args.sampler_engine == 'lmdeploy':
            from swift.llm import LmdeployEngine
            _Engine = LmdeployEngine
        elif self.args.sampler_engine == 'no':
            _Engine = None
        else:
            raise ValueError(f'Cannot find engine name: {self.args.sampler_engine}')
        return _Engine

    def _prepare_template(self) -> None:
        # Hack from super()
        self._prepare_request_configs()

    def _prepare_request_configs(self):
        _args = self.args
        request_config = _args.get_request_config()
        request_config.stop = _args.stop_words
        request_config.seed = _args.seed
        self.expand_request_configs = []
        self.rollout_request_configs = []
        for i in range(_args.num_return_sequences):
            expand_request_config = deepcopy(request_config)
            expand_request_config.n = 1
            expand_request_config.num_beams = expand_request_config.n
            expand_request_config.seed += i
            self.expand_request_configs.append(expand_request_config)
            rollout_request_config = deepcopy(request_config)
            rollout_request_config.max_tokens = 500
            rollout_request_config.temperature = 0.0
            rollout_request_config.n = 1
            self.rollout_request_configs.append(rollout_request_config)

    def do_sample(self, data):
        if not isinstance(data, list):
            data = [data]
        generated = []
        for item in data:
            logger.info(f'time: {time.ctime(time.time())}')
            try:
                messages = item['messages'][0]
                query = messages[0]['content']
                ground_truth = messages[1]['content']
                generated.append(self.search_single(query, ground_truth) + '\n')
            except Exception as e:
                logger.error(f'Error: {e}')
                logger.error(f'Traceback: {traceback.format_exc()}')
        return generated
