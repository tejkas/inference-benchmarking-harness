#!/bin/bash
# Knee search against a vLLM server on one GPU node. The client runs on the same node as the
# server, so network latency doesn't blur millisecond-level timings.
#
#   sbatch cluster/knee_vllm.sbatch                                # raw capacity
#   SLO_TTFT_MS=500 SLO_TPOT_MS=50 sbatch cluster/knee_vllm.sbatch # capacity under SLOs
#
#SBATCH --job-name=knee-vllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}  # no Qwen/DeepSeek on GT
PORT=${PORT:-8000}
VENV=${VENV:-$HOME/scratch/venvs/bench}           # has vllm + requirements.txt installed
VLLM_ARGS=${VLLM_ARGS:-}                          # e.g. "--max-num-seqs 64" for ground-truth runs
# Token and weights live on scratch: home quota is too small for model weights
export HF_HOME=${HF_HOME:-$HOME/scratch/.cache/huggingface}

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# Record the hardware with the results: GPU model, memory, driver, interconnect
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi topo -m || true
python -c "import vllm; print('vllm', vllm.__version__)"

vllm serve "$MODEL" --port "$PORT" $VLLM_ARGS > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

# Loading weights and compiling CUDA graphs can take several minutes
for _ in $(seq 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  if ! kill -0 $SERVER 2>/dev/null; then
    echo "vLLM exited during startup:"; tail -50 "logs/vllm-$SLURM_JOB_ID.log"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not ready after 15 min"; exit 1; }

SLO_ARGS=()
[ -n "${SLO_TTFT_MS:-}" ] && SLO_ARGS+=(--slo-ttft-ms "$SLO_TTFT_MS")
[ -n "${SLO_TPOT_MS:-}" ] && SLO_ARGS+=(--slo-tpot-ms "$SLO_TPOT_MS")

python run.py --model "$MODEL" --base-url "http://127.0.0.1:$PORT/v1" --find-knee \
  --dataset sharegpt --tokenizer "$MODEL" --ignore-eos ${SLO_ARGS[@]+"${SLO_ARGS[@]}"}  # empty-array safe under set -u
