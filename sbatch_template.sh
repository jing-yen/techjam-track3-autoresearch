#!/bin/bash
# Slurm array job template rendered by runner.py. One array task per candidate.
# Placeholders in ${...} are substituted by runner.render_sbatch().
#SBATCH --job-name=${JOB_NAME}
${PARTITION_LINE}
${ACCOUNT_LINE}
${EXCLUDE_LINE}
#SBATCH --gres=${GRES}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/%A_%a.out
#SBATCH --array=0-${ARRAY_MAX}%1
set -euo pipefail

${MODULE_LOAD}

cd ${WORKDIR}

# Candidate paths, indexed by the array task id.
CANDIDATES=(${CANDIDATE_LIST})
CAND="${CANDIDATES[$SLURM_ARRAY_TASK_ID]}"

echo "[sbatch] task=$SLURM_ARRAY_TASK_ID candidate=$CAND host=$(hostname)"
nvidia-smi -L || true

${PYTHON} bench_harness.py \
  --candidate "$CAND" \
  --shapes ${SHAPES} \
  --dtype ${DTYPE} \
  --device ${DEVICE} \
  ${EXTRA_ARGS} \
  --out ${RESULTDIR}/result_${SLURM_ARRAY_TASK_ID}.json
