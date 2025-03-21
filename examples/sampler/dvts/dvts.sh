USE_OPENCOMPASS_EVALUATOR=True

python -m swift.llm.sampling.sampling \
    --model qwen-max \
    --orm_model math \
    --prm_model vllm \
    --prm_kwargs '{"base_url":"http://localhost:8000/pooling", "api_key":"EMPTY", "model":"Qwen2.5-Math-PRM-7B"}' \
    --sampler_type dvts \
    --system ./system.txt \
    --seed 42 \
    --dataset ./datasets/math_splits/train.jsonl \
    --max_length 2048 \
    --load_args false \
    --sampler_engine vllm \
    --engine_kwargs '{"gpu_memory_utilization": 0.9}' \
    --max_new_tokens 768 \
    --override_exist_file true \
    --num_sampling_per_gpu_batch_size 1 \
    --max_iterations 200 \
    --rollout_depth 0 \
    --dvts_beam_size 8 \
    --dvts_beam_width 4 \
    --process_reward_rate 0.5 \
    --output_dir .output/sampler/dvts/ \
    --cache_files cache.jsonl \
    --output_file output.jsonl \
    --temperature 1.0