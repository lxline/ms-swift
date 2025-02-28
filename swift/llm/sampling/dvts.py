import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from copy import deepcopy

import json
import warnings
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
    current_texts: List[str] = field(default_factory=list)
    rollout_texts: List[str] = field(default_factory=list)
    current_scores: List[float] = field(default_factory=list)
    rollout_scores: List[float] = field(default_factory=list)
    outcome_score: float = 0.0
    terminated: bool = False
    children: List['Beam'] = field(default_factory=list)

    def to_dict(self):
        return {
            'current_texts': self.current_texts,
            'rollout_texts': self.rollout_texts,
            'current_scores': self.current_scores,
            'rollout_scores': self.rollout_scores,
            'outcome_score': self.outcome_score,
            'terminated': self.terminated,
            'children': [child.to_dict() for child in self.children] if self.children else []
        }


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
                 suffix_messages,
                 config,
                 generator,
                 orm_model,
                 prm_model,):
        self.query = query
        self.ground_truth = ground_truth
        self.prefix_messages = prefix_messages
        self.suffix_messages = suffix_messages
        self.config = config

        self.generator = generator
        self.orm_model = orm_model
        self.prm_model = prm_model

        self.usage_info = UsageInfo(0, 0, 0)
        self.step_index = 0
        self.root = Beam()
        self.beams = [self.root]

    def update_usage_info(self, response):
        for key, value in self.usage_info.__dict__.items():
            update_value = getattr(response.usage, key, None) + value
            setattr(self.usage_info, key, update_value)

    def build(self):
        for iter_index in range(self.config.max_iterations):
            if len(self.beams) == 0:
                break
            self.expand()
            self.rollout()
            self.prune(iter_index)
        answers = self.collect()
        return answers

    def generate_k_steps(self, beams, k):
        def process_infer_requests():
            pass

        def process_generate_context():
            pass

        if self.config.generate_strategy == "chat":
            pass
        elif self.config.generate_strategy == "generate":
            pass

    def expand(self):
        _config = self.config
        n = _config.dvts_beam_width
        infer_requests = []
        for _beam in self.beams:
            infer_messages = self.prefix_messages[:]
            if len(_beam.current_texts) > 0:
                infer_messages += [{
                    'role': 'assistant',
                    'content': _config.stop_words[0].join(_beam.current_texts),
                }] + self.suffix_messages
            infer_requests += [InferRequest(infer_messages) for _ in range(n)]

        # e_time = time.time()
        unique_output = set()
        responses = perform_infer(self.generator, infer_requests, _config.expand_request_configs,
                                  **_config.infer_kwargs)
        for index, response in enumerate(responses):
            # self.update_usage_info(response)
            if response == None:
                continue
            output = \
                response[0].choices[0].message.content.rstrip("".join(_config.stop_words)).split(_config.stop_words[0])[0]
            if output in unique_output:
                continue
            unique_output.add(output)
            parent_beam = self.beams[index // n]
            terminated = self.orm_model.check_terminate(output)[0]
            new_beam = Beam(
                current_texts=parent_beam.current_texts[:] + [output],
                current_scores=parent_beam.current_scores[:],
                terminated=terminated,
            )
            parent_beam.children.append(new_beam)

        # To fetch Process Reward in parallel
        # e_time = time.time()
        orm_infer_requests, prm_infer_requests = [], []
        for _beam in self.beams:
            for child in _beam.children:
                history_messages = [{
                    'role': 'assistant',
                    'content': step,
                } for step in child.current_texts]
                infer_request = InferRequest(self.prefix_messages + history_messages)
                prm_infer_requests.append(infer_request)
                orm_infer_requests.append(InferRequest([{'role': 'assistant', 'content': child.current_texts[-1]}]))
        prm_score, _prm_mask = get_reward(
            self.prm_model,
            prm_infer_requests,
            threshold=_config.prm_threshold,
            do_normalize=False,
        )
        if self.config.process_reward_rate < 1:
            orm_score, _orm_mask = get_reward(
                self.orm_model,
                orm_infer_requests,
                ground_truths=[self.ground_truth] * len(orm_infer_requests),
                threshold=0.0)
        else:
            orm_score = [0.0] * len(orm_infer_requests)
        # logger.info(f"expand.prm time: {time.time() - e_time}")

        score_index = 0
        for _beam in self.beams:
            all_terminated = True
            for child in _beam.children:
                child.current_scores.append(prm_score[score_index])
                if child.terminated:
                    child.outcome_score = orm_score[score_index]
                else:
                    all_terminated = False
                score_index += 1
            if all_terminated:
                _beam.terminated = True

    def rollout(self):
        _config = self.config
        index2rollout_beams = {}
        active_index = []
        for _beam in self.beams:
            for child in _beam.children:
                if not child.terminated:
                    index2rollout_beams[len(active_index)] = child
                    active_index.append(len(active_index))

        for i in range(_config.rollout_depth):
            if len(active_index) == 0:
                break
            infer_requests = []
            for index in active_index:
                beam = index2rollout_beams[index]
                history_messages = [{
                    'role': 'assistant',
                    'content': _config.stop_words[0].join(beam.current_texts + beam.rollout_texts),
                }]
                infer_request = InferRequest(self.prefix_messages + history_messages + self.suffix_messages)
                infer_requests.append(infer_request)

            responses = perform_infer(self.generator, infer_requests, _config.rollout_request_configs,
                                      **_config.infer_kwargs)

            for index, response in zip(active_index, responses):
                output = response[0].choices[0].message.content.rstrip("".join(_config.stop_words)).split(_config.stop_words[0])[0]
                beam = index2rollout_beams[index]
                beam.rollout_texts.append(output)

            nxt_index = []
            for index in active_index:
                beam = index2rollout_beams[index]
                if not self.orm_model.check_terminate(beam.rollout_texts[-1])[0]:
                    nxt_index.append(index)
            active_index = nxt_index

        prm_infer_requests = []
        for _beam in self.beams:
            for child in _beam.children:
                history_messages = [{
                    'role': 'assistant',
                    'content': step,
                } for step in child.current_texts] + [{
                    'role': 'assistant',
                    'content': step,
                } for step in child.rollout_texts]
                infer_request = InferRequest(self.prefix_messages + history_messages)
                prm_infer_requests.append(infer_request)

        prm_score, _prm_mask = get_reward(
            self.prm_model,
            prm_infer_requests,
            threshold=_config.prm_threshold,
            do_normalize=False,
        )

        prm_score_index = 0
        for _beam in self.beams:
            for child in _beam.children:
                child.rollout_scores.append(prm_score[prm_score_index])
                prm_score_index += 1

    def prune(self, iter_index):
        next_beams = []
        for _beam in self.beams:
            if not _beam.terminated and len(_beam.children) > 0:
                if iter_index > 0:
                    next_beams.append(max(_beam.children, key=lambda x: np.mean([x.current_scores[-1]] + x.rollout_scores)))
                else:
                    next_beams = _beam.children[:]
        self.beams = next_beams

    def collect(self):
        return self.root.to_dict()


class DvtsSampler(Sampler):

    def __init__(self, input_args: SamplingArguments):
        super().__init__(input_args)

    def _prepare_model_tokenizer(self):
        _args = self.args
        self.infer_kwargs = {}
        if _args.sampler_engine == 'multi_clients':
            from swift.llm import InferClient
            self.infer_engine = [
                InferClient(**_args.engine_kwargs) for _ in range(_args.dvts_beam_size * _args.dvts_beam_width)
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
        _args = self.args
        if _args.system is not None:
            _args.system_message = [{
                'role': 'system',
                'content': _args.system,
            }]
        else:
            _args.system_message = []

        if _args.continue_prompt is not None:
            _args.suffix_messages = [{
                'role': 'user',
                'content': _args.continue_prompt,
            }]
        else:
            _args.suffix_messages = []

        _args.sep_words = _args.stop_words[0]

        self._prepare_request_configs()

    def _prepare_request_configs(self):
        _args = self.args
        request_config = _args.get_request_config()
        request_config.stop = _args.stop_words
        request_config.seed = _args.seed
        self.expand_request_configs = []
        self.rollout_request_configs = []
        for i in range(_args.dvts_beam_size * _args.dvts_beam_width):
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
        answers = BeamSearchTree(
            query=query,
            ground_truth=ground_truth,
            prefix_messages=prefix_messages,
            suffix_messages=_args.suffix_messages,
            config=_args,
            generator=self.infer_engine,
            orm_model=self.orm_model,
            prm_model=self.prm_model,
        ).build()
        return answers

    def process_item(self, messages):
        try:
            query = messages[0]['content']
            ground_truth = messages[1]['content']
            answers = self.create_trees(query, ground_truth)
            result = {
                "query": query,
                "ground_truth": ground_truth,
                "answers": answers,
            }
            return result
        except Exception as e:
            logger.error(f'Error: {e}')
            logger.error(f'Traceback: {traceback.format_exc()}')
            return None

    def do_sample(self, data):
        batch_messages = data['messages']
        if not isinstance(batch_messages, list):
            batch_messages = [batch_messages]

        generated = []
        s_time = time.time()
        logger.info(f'Batch started time: {time.ctime(s_time)}')
        with ThreadPoolExecutor() as executor:
            future_to_item = {executor.submit(self.process_item, messages): messages for messages in batch_messages}
            for future in as_completed(future_to_item):
                result = future.result()
                if result is not None:
                    generated.append(result)

        generated_json = json.dumps(generated, ensure_ascii=False) + '\n'
        logger.info(f'Batch generated: {generated_json}')
        logger.info(f"used time: {time.time() - s_time}")
        return generated_json
