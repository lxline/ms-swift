import hashlib
import inspect
from copy import copy
from typing import Any, Dict, List, Optional

import json
import numpy as np

from swift.llm import InferRequest, RequestConfig
from swift.utils import get_logger

logger = get_logger()


def get_messages_md5(row: Dict[str, Any]):
    row = copy(row)
    row.pop('choices', None)
    serialized = json.dumps(row, sort_keys=True)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()


def get_reward(model: Any,
               infer_requests: List[InferRequest],
               request_config: RequestConfig = None,
               ground_truths: List[str] = None,
               threshold: Optional[float] = None,
               strategy: Optional[str] = "min",
               do_normalize: bool = True,):
    """Get reward from an RM model.

    Args:
        model: The model instance or an RM evaluator
        infer_requests: Infer requests sent to the model
        request_config: Infer config
        ground_truths: The ground truth list
        threshold: An optional threshold to generate the mask
        do_normalize: Whether to normalize the scores

    Returns:
        Tuple
        Index 0: The scores matched the infer_requests
        Index 1: The mask filtered by the threshold
    """
    from swift.llm import InferEngine
    infer_func = model.infer if isinstance(model, InferEngine) else model.__call__
    parameters = inspect.signature(infer_func).parameters
    gt_param = {}
    if 'ground_truths' in parameters:
        gt_param = {'ground_truths': ground_truths}
    rewards = infer_func(infer_requests, request_config=request_config, **gt_param)
    from swift.llm.infer.protocol import ChatCompletionResponse
    if isinstance(rewards[0], ChatCompletionResponse):
        rewards = [float(r.choices[0].message.content) for r in rewards]
    arr = []
    for reward in rewards:
        if isinstance(reward, (list, tuple)):
            if strategy == 'max':
                arr.append(max(reward))
            elif strategy == 'min':
                arr.append(min(reward))
            elif strategy == 'mean':
                arr.append(np.mean(reward))
            elif strategy == 'last':
                arr.append(reward[-1])
        else:
            arr.append(float(reward))

    _mask = np.array([True] * len(arr))
    if threshold is not None:
        # > not >=, orm caller passes 0, which will cause error
        _mask = np.array([a > threshold for a in arr])

    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val == max_val:
            if min_val == 0:
                constant_value = 0.0
            else:
                constant_value = min(1.0, min_val)
            return np.full_like(arr, fill_value=constant_value, dtype=np.float64)
        normalized = (arr - min_val) / (max_val - min_val + 1e-5)
        return normalized

    if do_normalize:
        return normalize(arr), _mask
    return arr, _mask


def perform_infer(infer_engines, infer_requests, request_configs, **infer_kwargs):
    if isinstance(infer_engines, list):
        assert len(infer_engines) >= len(request_configs) >= len(infer_requests)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n = len(infer_requests)
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = {
                executor.submit(perform_infer, infer_engines[i], infer_requests[i], request_configs[i], **infer_kwargs):
                i
                for i in range(n)
            }
            responses = [None] * n
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    responses[task_id] = future.result()
                except Exception as e:
                    logger.info(f'Perform infer task: {task_id} get an error: {e}')
        return responses
    elif isinstance(infer_requests, list):
        responses = []
        if isinstance(request_configs, list):
            assert len(infer_requests) <= len(request_configs)
            for i in range(len(infer_requests)):
                responses += infer_engines.infer(
                    [infer_requests[i]],
                    request_configs[i],
                    **infer_kwargs,
                )
        elif isinstance(request_configs, RequestConfig):
            for infer_request in infer_requests:
                responses += infer_engines.infer(
                    [infer_request],
                    request_configs,
                    **infer_kwargs,
                )
        return responses
    return infer_engines.infer(
        [infer_requests],
        request_configs,
        **infer_kwargs,
    )

async def async_perform_generate(generator, prompts, generate_configs):
    from swift.llm.infer.protocol import random_uuid
    async def consume_async_generator(async_gen):
        result = None
        async for result in async_gen:
            pass
        assert result is not None
        return result


    llm_inputs, request_ids = [], []
    request_id2index = {}
    for index in range(len(prompts)):
        request_ids.append(random_uuid())
        request_id2index[request_ids[-1]] = index

    tasks = [
        asyncio.create_task(
            consume_async_generator(generator.generate(prompts[i], generate_configs[i], request_ids[i])))
        for i in range(len(llm_inputs))
    ]

    done, pending = await asyncio.wait(tasks)

    answers = [""] * len(prompts)
    for task in done:
        try:
            result = task.result()
            for output in result.outputs:
                answers[request_id2index[result.request_id]] = output.text
        except Exception as e:
            print(f"Task failed with exception: {e}")

    return answers


def collect_from_mct(monte_carlo_tree, collect_filter_threshold):
    from transformers.utils import strtobool
    if isinstance(monte_carlo_tree, str):
        monte_carlo_tree = json.loads(monte_carlo_tree)

    def _collect(collect_curr_node, _path: list[str], _outcome_rewards: list[float], _process_rewards: list[float]):
        _prefer_pairs, _correct_answers, _incorrect_answers = [], [], []
        _outcome_rewards = _outcome_rewards[:] + [collect_curr_node['outcome_reward']]
        _process_rewards = _process_rewards[:] + [collect_curr_node['process_reward']]
        if len(collect_curr_node['children']) > 0:
            for child in collect_curr_node['children']:
                p, c, i = _collect(child, _path[:] + [collect_curr_node['step']], _outcome_rewards, _process_rewards)
                _prefer_pairs += p
                _correct_answers += c
                _incorrect_answers += i
            sorted_children = sorted(collect_curr_node['children'], key=lambda x: x['outcome_reward'])
            if sorted_children[-1]['outcome_reward'] - sorted_children[0]['outcome_reward'] > collect_filter_threshold:
                # TODO: filter with visit count
                prefer_pair = {
                    'path': _path[:],
                    'good': sorted_children[-1]['step'],
                    'good_score': sorted_children[-1]['outcome_reward'],
                    'bad': sorted_children[0]['step'],
                    'bad_score': sorted_children[0]['outcome_reward'],
                }
                _prefer_pairs.append(prefer_pair)
        if strtobool(collect_curr_node['terminated']):
            _answer = {
                'answer': '\n\n'.join(_path[:]),
                'mean_outcome_reward': np.mean(_outcome_rewards),
                'min_outcome_reward': np.min(_outcome_rewards),
                'mean_process_reward': np.mean(_process_rewards),
                'min_process_reward': np.min(_process_rewards),
            }
            if strtobool(collect_curr_node['correct']):
                _correct_answers.append(_answer)
            else:
                _incorrect_answers.append(_answer)
        return _prefer_pairs, _correct_answers, _incorrect_answers

    _root = monte_carlo_tree
    prefer_pairs, correct_answers, incorrect_answers = _collect(_root, [], [], [])
    return prefer_pairs, correct_answers, incorrect_answers
