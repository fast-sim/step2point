#!/usr/bin/env bash

set -euo pipefail

KEY4HEP_SETUP="${KEY4HEP_SETUP:-/cvmfs/sw.hsf.org/key4hep/setup.sh}"
VENV_PATH="${1:-.venv-key4hep}"

if [[ ! -f "$KEY4HEP_SETUP" ]]; then
    echo "Key4hep setup script not found: $KEY4HEP_SETUP" >&2
    exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1
from podio import root_io  # noqa: F401
import edm4hep  # noqa: F401
PY
then
    # key4hep/setup.sh dereferences positional parameters directly, so it must
    # not be sourced while nounset is active and it must not see this script's
    # positional arguments.
    key4hep_args=("$@")
    set +u
    set --
    # shellcheck source=/dev/null
    source "$KEY4HEP_SETUP"
    set -- "${key4hep_args[@]}"
    set -u
fi

python -m venv "$VENV_PATH"

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cat <<EOF
Key4hep-aware virtual environment created at: $VENV_PATH

Reuse it with:
  source "$KEY4HEP_SETUP"
  source "$VENV_PATH/bin/activate"

Then run, for example:
  pytest -q tests/integration/test_root_reader_optional.py
EOF
