export SWEBENCH_PRO_IMAGE_REPO_TEMPLATE="jefzda/sweap-images"
export PYTHONPATH=./src
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
unset DOCKER_HOST
unset DOCKER_CONTEXT
unset LLM_BASE_URL
unset OMNIMAAS_API_KEY
unset OPENAI_API_KEY
unset LITELLM_API_KEY

./.venv/bin/python3 scripts/run_swebenchpro_subset.py \
  --parquet-path train/SWE-bench_Pro/data/test-00000-of-00001.parquet \
  --k 10 \
  --traj-dir ./traj \
  --local-model-path  /Users/bytedance/R2E-Gym/models/FrogMini-14B-2510 \
  --backend docker \
  --seed 42 \
  --shuffle \
  --start-idx 0 \
  --max-steps  25 \
  --max-steps-absolute 30 \
  --temperature 0.0 \
  --max-reward-calc-time 240 \
  --max-tokens 80000
