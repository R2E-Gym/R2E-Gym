export SWEBENCH_PRO_IMAGE_REPO_TEMPLATE="jefzda/sweap-images"
export PYTHONPATH=./src
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
export LLM_BASE_URL="https://api.cloubic.com/v1/messages"
export CLOUBIC_API_KEY=sk-uekHcvjVEGr6LwW2SO0FzieL3ykG1BVV44h0nNYUlY8
LLM_PROVIDER="${LLM_PROVIDER:-}"
LLM_NAME="claude-opus-4-5-20251101"
if [ -z "${LLM_PROVIDER:-}" ]; then
  case "${LLM_BASE_URL:-}" in
    */v1/messages|*/messages) LLM_PROVIDER="anthropic" ;;
    *) LLM_PROVIDER="openai" ;;
  esac
fi

if [ "${LLM_PROVIDER}" = "anthropic" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    if [ -n "${CLOUBIC_API_KEY:-}" ]; then
      export ANTHROPIC_API_KEY="$CLOUBIC_API_KEY"
    elif [ -n "${OPENAI_API_KEY:-}" ]; then
      export ANTHROPIC_API_KEY="$OPENAI_API_KEY"
    fi
  fi
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Missing ANTHROPIC_API_KEY (or set CLOUBIC_API_KEY)"
    exit 1
  fi
else
  if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${CLOUBIC_API_KEY:-}" ]; then
    export OPENAI_API_KEY="$CLOUBIC_API_KEY"
  fi
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Missing OPENAI_API_KEY (or set CLOUBIC_API_KEY)"
    exit 1
  fi
fi

if [ -z "${DOCKER_HOST:-}" ] && command -v docker >/dev/null 2>&1; then
  DOCKER_HOST_FROM_CONTEXT="$(docker context inspect --format '{{ .Endpoints.docker.Host }}' 2>/dev/null | head -n 1)"
  if [ -n "${DOCKER_HOST_FROM_CONTEXT:-}" ]; then
    export DOCKER_HOST="$DOCKER_HOST_FROM_CONTEXT"
  fi
fi

./.venv/bin/python3 scripts/run_swebenchpro_subset.py \
  --parquet-path train/SWE-bench_Pro/data/test-00000-of-00001.parquet \
  --k 15 \
  --traj-dir ./traj \
  --llm-name "${LLM_PROVIDER}/${LLM_NAME}" \
  --backend docker \
  --seed 42 \
  --shuffle \
  --start-idx 0 \
  --max-steps 15 \
  --max-steps-absolute 30 \
  --temperature 0.0 \
  --max-reward-calc-time 240 \
  --max-tokens 16384
