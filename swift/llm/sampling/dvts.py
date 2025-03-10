import time
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from copy import deepcopy

import json
import warnings
import numpy as np

import math
from typing import Literal, List, Dict
from swift.llm import InferRequest
from swift.llm.argument.sampling_args import SamplingArguments
from swift.llm.infer.protocol import UsageInfo
from swift.utils import get_logger
from .base import Sampler
from .utils import get_reward, perform_infer, async_perform_generate


logger = get_logger()


NXT_PROMPT = "Continue."

Qwen_template = """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

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
            'current_text': self.current_texts[-1] if len(self.current_texts) else "",
            'rollout_texts': self.rollout_texts,
            'current_score': self.current_scores[-1] if len(self.current_scores) else 0,
            'rollout_scores': self.rollout_scores,
            'outcome_score': self.outcome_score,
            'terminated': self.terminated,
            'children': [child.to_dict() for child in self.children] if self.children else []
        }


@dataclass
class BeamSearchTree:
    query: str
    ground_truth: str
    prefix_text: str
    suffix_text: str
    prefix_messages: List[Dict[str, str]] = field(default_factory=list)
    suffix_messages: List[Dict[str, str]] = field(default_factory=list)
    step_index: int = 0
    root: 'Beam' = field(default_factory=Beam)
    beams: List['Beam'] = field(default_factory=list)

    def __post_init__(self):
        self.beams = [self.root]

    def to_dict(self):
        return self.root.to_dict()

    def terminated(self):
        return len(self.beams) == 0


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
        # super()._prepare_template()
        # Hack from super()
        self.args.sep_word = self.args.stop_words[0]
        _args = self.args

        self._prepare_generation_configs()

    def _prepare_generation_configs(self):
        _args = self.args
        request_config = _args.get_request_config()
        request_config.n = 1
        request_config.stop = _args.stop_words
        request_config.seed = _args.seed
        generation_config = self.infer_engine._prepare_generation_config(request_config)
        self.expand_generation_configs = []
        self.rollout_generation_configs = []
        for i in range(_args.dvts_beam_size * _args.dvts_beam_width):
            expand_generation_config = deepcopy(generation_config)
            expand_generation_config.seed += i
            self.expand_generation_configs.append(expand_generation_config)
            rollout_generation_config = deepcopy(generation_config)
            rollout_generation_config.max_tokens = 500
            rollout_generation_config.temperature = 0.0
            self.rollout_generation_configs.append(rollout_generation_config)

    def build_trees(self):
        for iter_index in range(self.args.max_iterations):
            if all([bst.terminated() for bst in self.beam_search_trees]):
                break
            self.expand()
            self.rollout()
            self.prune(iter_index)

    def generate_k_steps(self,
                         prompts,
                         generation_configs,
                         k,):
        _args = self.args

        results, active_index = [], []
        for i in range(len(prompts)):
            results.append([])
            active_index.append(i)

        for step_index in range(k):
            if len(active_index) == 0:
                break

            step_answers = self.infer_engine.generate(prompts, generation_configs)
            nxt_index = []
            for index, step_answer in zip(active_index, step_answers):
                results[index].append(step_answer + _args.sep_word)
                if not self.orm_model.check_terminate(step_answer)[0]:
                    nxt_index.append(index)
            active_index = nxt_index

        return results

    def expand(self):
        _args = self.args
        n = _args.dvts_beam_width
        expand_prompts = []
        expand_generation_configs = []
        for tree in self.beam_search_trees:
            for i, _beam in enumerate(tree.beams):
                for j in range(n):
                    prompt = tree.prefix_text + "".join(_beam.current_texts)
                    expand_prompts.append(prompt)
                    expand_generation_configs.append(_args.expand_generation_configs[i * n + j])

        results = self.generate_k_steps(expand_prompts, expand_generation_configs, 1)

        for tree in self.beam_search_trees:
            unique_output = set()
            for i, _beam in enumerate(tree.beams):
                for j in range(n):
                    index = i * n + j
                    output = results[index][0]
                    if output in unique_output:
                        continue
                    unique_output.add(output)
                    terminated = self.orm_model.check_terminate(output)[0]
                    new_beam = Beam(
                        current_texts=_beam.current_texts[:] + [output],
                        current_scores=_beam.current_scores[:],
                        terminated=terminated,
                    )
                    _beam.children.append(new_beam)

        # To fetch Process Reward in parallel
        # e_time = time.time()
        orm_infer_requests, prm_infer_requests, ground_truths = [], [], []
        for tree in self.beam_search_trees:
            for i, _beam in enumerate(tree.beams):
                for child in _beam.children:
                    history_messages = [{
                        'role': 'assistant',
                        'content': step.strip(),
                    } for step in child.current_texts]
                    infer_request = InferRequest(tree.prefix_messages + history_messages)
                    prm_infer_requests.append(infer_request)
                    orm_infer_requests.append(InferRequest([{'role': 'assistant', 'content': child.current_texts[-1]}]))
                    ground_truths.append(tree.ground_truth)

        prm_score, _prm_mask = get_reward(
            self.prm_model,
            prm_infer_requests,
            threshold=_args.prm_threshold,
            strategy="last",
            do_normalize=False,
        )
        if _args.process_reward_rate < 1:
            orm_score, _orm_mask = get_reward(
                self.orm_model,
                orm_infer_requests,
                ground_truths=ground_truths,
                threshold=0.0)
        else:
            orm_score = [0.0] * len(orm_infer_requests)
        # logger.info(f"expand.prm time: {time.time() - e_time}")

        score_index = 0
        for tree in self.beam_search_trees:
            for i, _beam in enumerate(tree.beams):
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
        _args = self.args
        if _args.rollout_depth == 0:
            return

        rollout_prompts = []
        rollout_generation_configs = []
        for tree in self.beam_search_trees:
            for i, _beam in enumerate(tree.beams):
                for j, child in enumerate(_beam.children):
                    if child.terminated:
                        continue
                    prompt = tree.prefix_text + "".join(_beam.current_texts)
                    rollout_prompts.append(prompt)
                    rollout_generation_configs.append(_args.rollout_generation_configs[i * _args.dvts_beam_size + j])

        results = self.generate_k_steps(rollout_prompts, rollout_generation_configs, _args.rollout_depth)

        prm_infer_requests = []
        for tree in self.beam_search_trees:
            for i, _beam in enumerate(tree.beams):
                for j, child in enumerate(_beam.children):
                    child.rollout_texts = results.pop(0)
                    history_messages = [{
                        'role': 'assistant',
                        'content': step.strip(),
                    } for step in child.current_texts] + [{
                        'role': 'assistant',
                        'content': step.strip(),
                    } for step in child.rollout_texts]
                    infer_request = InferRequest(tree.prefix_messages + history_messages)
                    prm_infer_requests.append(infer_request)

        prm_score, _prm_mask = get_reward(
            self.prm_model,
            prm_infer_requests,
            threshold=_args.prm_threshold,
            strategy="last",
            do_normalize=False,
        )

        prm_score_index = 0
        for tree in self.beam_search_trees:
            for i, _beam in enumerate(tree.beams):
                for j, child in enumerate(_beam.children):
                    child.rollout_scores.append(prm_score[prm_score_index])
                    prm_score_index += 1

    def prune(self, iter_index):
        _args = self.args
        def cal_score(beam):
            prm_score = np.mean([beam.current_scores[-1]] + beam.rollout_scores)
            orm_score = beam.outcome_score
            score = _args.process_reward_rate * prm_score + (1 - _args.process_reward_rate) * orm_score
            return score
        for tree in self.beam_search_trees:
            next_beams = []
            for _beam in tree.beams:
                if not _beam.terminated and len(_beam.children) > 0:
                    if iter_index > 0:
                        best_beam = max(_beam.children, key=lambda x: cal_score(x))
                        if not best_beam.terminated:
                            next_beams.append(best_beam)
                    else:
                        next_beams = _beam.children[:]
            tree.beams = next_beams

    def create_trees(self, batch_messages):
        _args = self.args
        _args.expand_generation_configs = self.expand_generation_configs
        _args.rollout_generation_configs = self.rollout_generation_configs
        _args.infer_kwargs = self.infer_kwargs

        self.beam_search_trees = []
        for messages in batch_messages:
            query = messages[0]['content']
            ground_truth = messages[1]['content']

            prompt = deepcopy(Qwen_template)
            prompt = prompt.replace("{{ .System }}", _args.system)
            prompt = prompt.replace("{{ .Prompt }}", query)

            prefix_messages = [
                {
                    "role": "system",
                    "content": _args.system,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ]
            suffix_messages = []

            bst = BeamSearchTree(query=query,
                                 ground_truth=ground_truth,
                                 prefix_text=prompt,
                                 suffix_text="",
                                 prefix_messages=prefix_messages,
                                 suffix_messages=suffix_messages,)

            self.beam_search_trees.append(bst)

        self.build_trees()

        results = []
        for bst in self.beam_search_trees:
            result = {
                "query": bst.query,
                "ground_truth": bst.ground_truth,
                "answers": bst.to_dict(),
            }
            results.append(result)

        return results

    def do_sample(self, data):
        batch_messages = data['messages']
        if not isinstance(batch_messages, list):
            batch_messages = [batch_messages]

        s_time = time.time()
        logger.info(f'Batch started time: {time.ctime(s_time)}')

        generated = self.create_trees(batch_messages)

        generated_json = json.dumps(generated, ensure_ascii=False) + '\n'
        logger.info(f'Batch generated: {generated_json}')
        logger.info(f"used time: {time.time() - s_time}")
        return generated_json
