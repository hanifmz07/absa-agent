# # basic
# python -m src.main.eval_agent --results_path results/absa_results.json

# custom path + retries
for max_retries in 1 3 5 10; do
    for seed in 42 123 777 2024 31415; do
        python -m src.main.eval_agent \
            --max_retries $max_retries \
            --prompt_set exp1 \
            --seed $seed \
            --runner_type async_0.6B
    done
done

# # with lowercasing + custom output dir
# python -m src.main.eval_agent \
#     --results_path results/absa_results_async.json \
#     --output_dir   results/eval \
#     --lowercase

# # custom path + retries
# python -m src.main.run_agent_async \
#     --test_case_path dataset/hoasa_hotel/indo/mvp_aos/test.json \
#     --max_retries 3 \
#     --prompt_set exp1 \
#     --seed 42