for mode in zero static_curated static_random dynamic; do
  case $mode in
    zero) ps=fewshot_zeroshot ;;
    static_curated) ps=exp1 ;;
    static_random) ps=fewshot_static_random ;;
    dynamic) ps=fewshot_dynamic ;;
  esac
  python -m src.main.run_agent_fewshot \
    --model_path Qwen/Qwen3-8B \
    --test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
    --prompt_set "$ps" --fewshot_mode "$mode" --seed 42 \
    --gpu_memory_utilization 0.9
done