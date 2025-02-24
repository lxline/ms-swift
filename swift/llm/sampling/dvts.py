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
    children: List['Beam'] = None


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


class BeamSearchTree:
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
            _beams = [beam for beam in self.beams if not beam.terminated]
            next_beams = []
            for beam in _beams:
                gen_beams = self.expand(beam)
                self.rollout(gen_beams)
                next_beams.append(self.prune(gen_beams))
            self.beams = next_beams
        return self.answers

    def expand(self, curr_beam: Beam):
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
            # self.update_usage_info(response)
            output = response.choices[0].message.content.rstrip("".join(_config.stop_words)).split(_config.stop_words[0])[0]
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
            do_normalize=False,
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

    def rollout(self, next_beams: List[Beam]):
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
                output = response.choices[0].message.content.rstrip("".join(_config.stop_words)).split(_config.stop_words[0])[0]
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
                do_normalize=False,
            )

            nxt_index = []
            for index, score in zip(active_index, prm_score):
                beam = index2rollout_beams[index]
                beam.rollout_scores.append(score)
                if not self.orm_model.check_terminate(beam.rollout_texts[-1])[0]:
                    nxt_index.append(index)
            active_index = nxt_index

    def prune(self, gen_beams):
        score2beam = {}
        for beam in gen_beams:
            mean_score = np.mean([beam.current_scores[-1]] + beam.rollout_scores)
            score2beam[mean_score] = beam
        best_score = max(score2beam.keys())
        best_beam = score2beam[best_score]
        return best_beam


class DvtsSampler(Sampler):

    def __init__(self, input_args: SamplingArguments):
        super().__init__(input_args)

    def _prepare_model_tokenizer(self):
        _args = self.args
        self.infer_kwargs = {}
        if _args.sampler_engine == 'multi_clients':
            from swift.llm import InferClient
            self.infer_engine = [
                InferClient(**_args.engine_kwargs) for _ in range(_args.num_trees * _args.beam_width)
            ]
            self.infer_kwargs['model'] = _args.model
        elif _args.sampler_engine == 'client':
            from swift.llm import InferClient
            self.infer_engine = InferClient(**_args.engine_kwargs)
            self.infer_kwargs['model'] = _args.model
        elif _args.sampler_engine == 'pt':
            from swift.llm import PtEngine
            self.infer_engine = PtEngine(_args.model, model_type=_args.model_type, **_args.engine_kwargs)
        elif _args.sampler_engine == 'vllm':
            from swift.llm import VllmEngine
            self.infer_engine = VllmEngine(_args.model, model_type=_args.model_type, **_args.engine_kwargs)
        elif _args.sampler_engine == 'lmdeploy':
            from swift.llm import LmdeployEngine
            self.infer_engine = LmdeployEngine(_args.model, model_type=_args.model_type, **_args.engine_kwargs)
        elif _args.sampler_engine == 'no':
            _Engine = None
            raise ValueError(f'sampler_engine was set to {_args.sampler_engine}, so no engine is used.')
        else:
            raise ValueError(f'Cannot find engine name: {_args.sampler_engine}')

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
        for i in range(max(_args.num_trees, _args.num_trees * _args.beam_width)):
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

    def create_trees(self, query: str, ground_truth: str):
        _args = self.args
        _args.expand_request_configs = self.expand_request_configs
        _args.rollout_request_configs = self.rollout_request_configs
        _args.infer_kwargs = self.infer_kwargs

        query_message = [{
            'role': 'user',
            'content': query,
        }]
        prefix_messages = _args.system_message + query_message
        infer_requests = [InferRequest(prefix_messages) for _ in range(_args.num_trees)]

        # e_time = time.time()
        expand_iter_index = 0
        unique_output = set()
        prm_infer_requests = []
        while True:
            responses = perform_infer(self.infer_engine, infer_requests, self.expand_request_configs,
                                      **self.infer_kwargs)
            for response in responses:
                output = \
                response.choices[0].message.content.rstrip("".join(_args.stop_words)).split(_args.stop_words[0])[0]
                if output in unique_output:
                    continue
                unique_output.add(output)
                infer_request = InferRequest(prefix_messages + [{'role': 'assistant', 'content': output}])
                prm_infer_requests.append(infer_request)
            if len(unique_output) > _args.num_trees:
                break
            if expand_iter_index == 5:
                raise ValueError(f'5 iterations did get enough responses for {_args.num_trees} trees')
            expand_iter_index += 1

        prm_score, _prm_mask = get_reward(
            self.prm_model,
            prm_infer_requests,
            threshold=_args.prm_threshold,
            do_normalize=False,
        )
        # logger.info(f"expand.prm time: {time.time() - e_time}")

        answers = []
        for output, score in zip(unique_output, prm_score):
            beam = Beam(
                current_texts = [output],
                current_scores = [score],
            )
            tree = BeamSearchTree(
                query,
                ground_truth,
                prefix_messages,
                beam,
                _args,
                self.infer_engine,
                self.orm_model,
                self.prm_model,
            )
            tree_answers = tree.build()
            answers.append(tree_answers)
        return answers

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
                answers = self.create_trees(query, ground_truth)
                generated.append(answers)
            except Exception as e:
                logger.error(f'Error: {e}')
                logger.error(f'Traceback: {traceback.format_exc()}')
        return generated
