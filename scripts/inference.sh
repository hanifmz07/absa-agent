# # defaults
# python -m src.main.run_agent_async

# custom path + retries
source .venv/bin/activate
for max_retries in 1 3 5 10; do
    for seed in 42 123 777 2024 31415; do
        python -m src.main.run_agent_async \
            --model_path Qwen/Qwen3-8B \
            --test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
            --max_retries $max_retries \
            --prompt_set exp1 \
            --seed $seed \
            --track_tokens
    done
done

# # custom path + retries
# python -m src.main.run_agent_async \
#     --test_case_path dataset/hoasa_hotel/indo/mvp_aos/test.json \
#     --max_retries 3 \
#     --prompt_set exp1 \
#     --seed 42

# python -m src.main.eval_agent \
#     --max_retries 3 \
#     --prompt_set exp1 \
#     --seed 42 \
#     --runner_type async