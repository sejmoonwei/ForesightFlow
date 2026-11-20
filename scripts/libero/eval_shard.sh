#!/usr/bin/env bash
# Run one LIBERO evaluation shard without saving videos.
set -Eeuo pipefail

if [[ $# -ne 11 ]]; then
  echo "Usage: $0 <CKPT_DIR> <CKPT_STEP> <RUN_LABEL> <RESULTS_DIR> <SUITE> <TASK_START> <TASK_END> <NUM_TRIALS> <GPU> <PORT> <TOTAL_SHARDS>" >&2
  exit 2
fi

CKPT_DIR=$1; CKPT_STEP=$2; RUN_LABEL=$3; RESULTS_DIR=$4
SUITE=$5; TASK_START=$6; TASK_END=$7; NUM_TRIALS=$8
GPU=$9; PORT=${10}; TOTAL_SHARDS=${11}

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${ROOT}
SERVER_SCRIPT=${ROOT}/scripts/libero/serve.py
CLIENT_SCRIPT=${ROOT}/scripts/libero/eval.py
FPF_SOURCE=${ROOT}/src
OPENPI_SOURCE=${ROOT}/vendor/openpi/src
OPENPI_CLIENT_SOURCE=${ROOT}/vendor/openpi/packages/openpi-client/src
LIBERO_EVAL_SOURCE=${ROOT}/sim
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:?Launch through scripts/run.py}
SERVER_PYTHON=${SERVER_PYTHON:?Launch through scripts/run.py}
CLIENT_PYTHON=${CLIENT_PYTHON:?Launch through scripts/run.py}
OUTPUT_ROOT=${OUTPUT_ROOT:?Launch through scripts/run.py}

MEM_FRACTION=${MEM_FRACTION:-0.82}
MUJOCO_GL=${MUJOCO_GL:-egl}
EVAL_SEED=${EVAL_SEED:?EVAL_SEED is required}
POLICY_SEED=${POLICY_SEED:?POLICY_SEED is required}
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-10}
SCORE_MODE=${SCORE_MODE:-shared_raw}
SELECTION_MODE=${SELECTION_MODE:-argmax}
TACTILE_SUBGEOM=${TACTILE_SUBGEOM:-1}
ROLLOUTS_DIR_ROOT=${ROLLOUTS_DIR_ROOT:-/tmp/fpf_libero_rollouts/${RUN_LABEL}}

[[ "${TACTILE_SUBGEOM}" == 1 ]] || {
  echo "[FATAL] Evaluation requires TACTILE_SUBGEOM=1" >&2
  exit 2
}

SHARD_TAG=${SUITE}_t${TASK_START}_${TASK_END}
SHARDS_DIR=${RESULTS_DIR}/shards/${RUN_LABEL}
SHARD_FILE=${SHARDS_DIR}/${SHARD_TAG}.json
MERGE_LOCK=${RESULTS_DIR}/.merge_${RUN_LABEL}.lock
RUN_ID=${RUN_LABEL}_${SHARD_TAG}_$(date +%Y%m%d_%H%M%S)
LOGS_DIR=${OUTPUT_ROOT}/logs/${RUN_ID}
SERVER_LOG=${LOGS_DIR}/server.log
SUITE_LOG=${LOGS_DIR}/eval_${SUITE}.log
SERVER_PID_FILE=${LOGS_DIR}/server.pid

mkdir -p "${LOGS_DIR}" "${SHARDS_DIR}" "${ROLLOUTS_DIR_ROOT}"

echo "[eval] label=${RUN_LABEL} shard=${SHARD_TAG} trials=${NUM_TRIALS}"
echo "[eval] seed=${EVAL_SEED} policy_seed=${POLICY_SEED} gpu=${GPU} port=${PORT}"
echo "[eval] K=${NUM_CANDIDATES} flow=${NUM_FLOW_STEPS} score=${SCORE_MODE} selection=${SELECTION_MODE} replan=5 video=off trajectory=off"

[[ -d "${CKPT_DIR}/params" ]] || { echo "[FATAL] missing params: ${CKPT_DIR}" >&2; exit 1; }
[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || { echo "[FATAL] missing LIBERO config" >&2; exit 1; }
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "[FATAL] port occupied: ${PORT}" >&2
  exit 1
fi

cleanup() {
  local rc=$? server_pid
  if [[ -f "${SERVER_PID_FILE}" ]]; then
    server_pid=$(<"${SERVER_PID_FILE}")
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "${server_pid}" 2>/dev/null || break
        sleep 1
      done
      kill -9 "${server_pid}" 2>/dev/null || true
    fi
    rm -f "${SERVER_PID_FILE}"
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM HUP

(
  export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:?}
  export HF_HOME=${HF_HOME:?}
  unset LEROBOT_HOME
  export PYTHONPATH=${FPF_SOURCE}:${OPENPI_SOURCE}:${OPENPI_CLIENT_SOURCE}
  export LD_PRELOAD=${FPF_CUDNN_PRELOAD:-}
  export CUDA_VISIBLE_DEVICES=${GPU}
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
  export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR:-${ROOT}/outputs/cache/jax}
  export TMPDIR=${EVAL_TMPDIR:-/tmp/fpf_libero_tmp}
  export PYTHONUNBUFFERED=1
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

  exec "${SERVER_PYTHON}" "${SERVER_SCRIPT}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --port "${PORT}" \
    --policy_seed "${POLICY_SEED}" \
    --action_horizon 10 \
    --num_steps "${NUM_FLOW_STEPS}" \
    --num_candidates "${NUM_CANDIDATES}" \
    --score_mode "${SCORE_MODE}" \
    --selection_mode "${SELECTION_MODE}" \
    --critic_uncertainty_beta 0 \
    --critic_score_horizon 0
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" >"${SERVER_PID_FILE}"

MAX_WAIT=${SERVER_START_TIMEOUT_SECONDS:-900}; elapsed=0; ready=0
while (( elapsed < MAX_WAIT )); do
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "[${SHARD_TAG}] server ready after ${elapsed}s"
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[FATAL ${SHARD_TAG}] server exited during startup" >&2
    tail -80 "${SERVER_LOG}"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done
(( ready == 1 )) || { tail -80 "${SERVER_LOG}"; exit 1; }

eval_start=$(date +%s)
set +e
(
  cd "${OPENPI_ROOT}"
  export PYTHONPATH=${ROOT}/scripts/libero:${ROOT}/vendor/libero:${LIBERO_EVAL_SOURCE}:${OPENPI_CLIENT_SOURCE}
  export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:?}
  export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}
  unset LEROBOT_HOME
  export MUJOCO_GL=${MUJOCO_GL}
  export MUJOCO_EGL_DEVICE_ID=${GPU}
  "${CLIENT_PYTHON}" "${CLIENT_SCRIPT}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.resize_size 224 \
    --args.replan_steps 5 \
    --args.task_suite_name "${SUITE}" \
    --args.num_steps_wait 10 \
    --args.num_trials_per_task "${NUM_TRIALS}" \
    --args.success_settle_steps 60 \
    --args.task_id_start "${TASK_START}" \
    --args.task_id_end "${TASK_END}" \
    --args.video_out_path "${ROLLOUTS_DIR_ROOT}" \
    --args.trajectory_out_path "${ROLLOUTS_DIR_ROOT}" \
    --args.trajectory_run_id "${SHARD_TAG}" \
    --args.tactile_tier 3 \
    --args.seed "${EVAL_SEED}"
) 2>&1 | tee "${SUITE_LOG}"
rc=${PIPESTATUS[0]}
set -e

eval_end=$(date +%s)
elapsed_total=$((eval_end - eval_start))
rate_line=$(grep 'Total success rate:' "${SUITE_LOG}" | tail -1 || true)
episodes_line=$(grep 'Total episodes:' "${SUITE_LOG}" | tail -1 || true)
success_line=$(grep '# successes:' "${SUITE_LOG}" | tail -1 || true)
rate=""; total_episodes=""; total_successes=""
[[ -z "${rate_line}" ]] || rate=$(awk -F': ' '{print $NF}' <<<"${rate_line}")
[[ -z "${episodes_line}" ]] || total_episodes=$(awk -F': ' '{print $NF}' <<<"${episodes_line}")
[[ -z "${success_line}" ]] || total_successes=$(sed -E 's/.*# successes: ([0-9]+).*/\1/' <<<"${success_line}")

jq -n \
  --arg run_label "${RUN_LABEL}" --argjson ckpt_step "${CKPT_STEP}" \
  --arg ckpt_dir "${CKPT_DIR}" --arg suite "${SUITE}" \
  --argjson task_id_start "${TASK_START}" --argjson task_id_end "${TASK_END}" \
  --argjson num_trials_per_task "${NUM_TRIALS}" \
  --argjson eval_seed "${EVAL_SEED}" --argjson policy_seed "${POLICY_SEED}" \
  --arg selection_mode "${SELECTION_MODE}" --argjson gpu "${GPU}" \
  --argjson port "${PORT}" --argjson rc "${rc}" \
  --argjson elapsed_seconds "${elapsed_total}" --arg finished_at "$(date -Iseconds)" \
  --arg logs_dir "${LOGS_DIR}" --argjson success_rate "${rate:-null}" \
  --argjson total_episodes "${total_episodes:-0}" \
  --argjson total_successes "${total_successes:-0}" \
  '{run_label:$run_label,ckpt_step:$ckpt_step,ckpt_dir:$ckpt_dir,suite:$suite,
    task_id_start:$task_id_start,task_id_end:$task_id_end,
    num_trials_per_task:$num_trials_per_task,eval_seed:$eval_seed,
    policy_seed:$policy_seed,v31_decision_seeded:true,num_candidates:5,
    num_flow_steps:10,replan_steps:5,score_mode:"shared_raw",
    selection_mode:$selection_mode,critic_uncertainty_beta:0,
    critic_score_horizon:0,actor_only:false,tactile_subgeom:true,
    save_video:false,save_trajectory:false,gpu:$gpu,port:$port,rc:$rc,
    elapsed_seconds:$elapsed_seconds,finished_at:$finished_at,logs_dir:$logs_dir,
    success_rate:$success_rate,total_episodes:$total_episodes,
    total_successes:$total_successes}' >"${SHARD_FILE}"

echo "[${SHARD_TAG}] completed rc=${rc} successes=${total_successes:-0}/${total_episodes:-0} elapsed=$((elapsed_total/60))m"
(( rc == 0 )) || exit "${rc}"

(
  flock -x 200
  cd "${OPENPI_ROOT}"
  "${SERVER_PYTHON}" scripts/libero/merge_shards.py \
    --shards-dir "${SHARDS_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --ckpt-step "${CKPT_STEP}" \
    --run-label "${RUN_LABEL}" \
    --expected "${TOTAL_SHARDS}" \
    --ckpt-dir "${CKPT_DIR}" || echo "[WARN ${SHARD_TAG}] merge deferred"
) 200>"${MERGE_LOCK}"

echo "[${SHARD_TAG}] exit"
