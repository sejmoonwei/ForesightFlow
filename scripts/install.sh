#!/usr/bin/env bash
# Installs only new virtualenvs INSIDE this release directory.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-all}
[[ "$MODE" == server || "$MODE" == sim || "$MODE" == all ]] || { echo 'Usage: bash scripts/install.sh [server|sim|all]' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv first (python3 -m pip install --user uv), then put its executable on PATH.' >&2; exit 1; }
if [[ "$MODE" == server || "$MODE" == all ]]; then
  [[ ! -e "$ROOT/.venv-server" ]] || { echo '.venv-server already exists; refusing to alter it.' >&2; exit 1; }
  GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$ROOT/.venv-server" \
    uv sync --project "$ROOT/vendor/openpi" --python 3.11.14 --frozen --no-dev
  uv pip install --python "$ROOT/.venv-server/bin/python" --no-deps -e "$ROOT"
  "$ROOT/.venv-server/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import shutil, sys, sysconfig
root=Path(sys.argv[1])
source=root/'vendor/openpi/src/openpi/models_pytorch/transformers_replace'
target=Path(sysconfig.get_paths()['purelib'])/'transformers'
assert target.is_dir(), target
shutil.copytree(source,target,dirs_exist_ok=True)
print('Applied vendored OpenPI transformers replacement to new server env.')
PY
fi
if [[ "$MODE" == sim || "$MODE" == all ]]; then
  [[ ! -e "$ROOT/.venv-sim" ]] || { echo '.venv-sim already exists; refusing to alter it.' >&2; exit 1; }
  # Set FPF_SIM_PYTHON to use an existing Python 3.8 interpreter.
  uv venv --python "${FPF_SIM_PYTHON:-3.8}" "$ROOT/.venv-sim"
  uv pip install --python "$ROOT/.venv-sim/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match \
    -r "$ROOT/requirements/simulation.txt"
  uv pip install --python "$ROOT/.venv-sim/bin/python" --no-deps \
    -e "$ROOT/vendor/libero" -e "$ROOT/vendor/openpi/packages/openpi-client"
fi
echo 'Environments ready. Configure configs/paths.local.json before running.'
