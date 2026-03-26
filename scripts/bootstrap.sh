#!/usr/bin/env bash
# bootstrap.sh — Set up the pyenv virtual environment for this project.
#
# Usage:
#   ./scripts/bootstrap.sh [PYTHON_VERSION] [VENV_NAME]
#
# Defaults:
#   PYTHON_VERSION = 3.12.4
#   VENV_NAME      = acic26
#
# Requires: macOS, Homebrew (to install pyenv if not present)

set -euo pipefail

PYTHON_VERSION="${1:-3.11.9}"
VENV_NAME="${2:-acic26}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[bootstrap]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[bootstrap]${RESET} $*"; }
error() { echo -e "${RED}[bootstrap]${RESET} $*" >&2; }

# ── OS check ─────────────────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
  error "This script currently supports macOS only."
  exit 1
fi

# ── 1. Ensure pyenv is installed ─────────────────────────────────────────────
if command -v pyenv &>/dev/null; then
  info "pyenv already installed: $(pyenv --version)"
else
  if ! command -v brew &>/dev/null; then
    error "Homebrew is required to install pyenv."
    error "Install Homebrew from https://brew.sh, then re-run this script."
    exit 1
  fi
  info "Installing pyenv via Homebrew..."
  brew install pyenv
fi

# ── 2. Verify pyenv is reachable in this shell session ───────────────────────
if ! command -v pyenv &>/dev/null; then
  warn "pyenv was installed but is not on your PATH in this shell session."
  warn "Add the following to your shell config (~/.zshrc or ~/.bash_profile),"
  warn "open a new terminal, then re-run this script:"
  warn ""
  warn '  export PYENV_ROOT="$HOME/.pyenv"'
  warn '  export PATH="$PYENV_ROOT/bin:$PATH"'
  warn '  eval "$(pyenv init -)"'
  exit 1
fi

# ── 3. Install target Python version ─────────────────────────────────────────
if pyenv versions --bare | grep "${PYTHON_VERSION}"; then
  info "Python ${PYTHON_VERSION} already installed in pyenv."
else
  info "Installing Python ${PYTHON_VERSION} via pyenv (this may take a few minutes)..."
  pyenv install "${PYTHON_VERSION}"
fi

# ── 4. Ensure pyenv-virtualenv is available ───────────────────────────────────
if ! pyenv commands | grep -q "^virtualenv$"; then
  info "Installing pyenv-virtualenv via Homebrew..."
  brew install pyenv-virtualenv
fi

# ── 5. Create the virtualenv ──────────────────────────────────────────────────
if pyenv virtualenvs --bare 2>/dev/null | grep "${VENV_NAME}"; then
  info "pyenv virtualenv '${VENV_NAME}' already exists."
else
  info "Creating pyenv virtualenv '${VENV_NAME}' (Python ${PYTHON_VERSION})..."
  pyenv virtualenv "${PYTHON_VERSION}" "${VENV_NAME}"
fi

# ── 6. Ensure system graphviz is installed (required by python-graphviz/pydot) ─
if command -v graphviz &>/dev/null || command -v dot &>/dev/null; then
  info "System graphviz already installed."
else
  info "Installing graphviz via Homebrew..."
  brew install graphviz
fi

# ── 7. Install Python dependencies via pip ────────────────────────────────────
REQUIREMENTS="${REPO_ROOT}/requirements.txt"
if [[ ! -f "${REQUIREMENTS}" ]]; then
  error "requirements.txt not found at ${REQUIREMENTS}"
  exit 1
fi

info "Upgrading pip..."
PYENV_VERSION="${VENV_NAME}" pyenv exec pip install --quiet --upgrade pip

info "Installing dependencies from requirements.txt..."
PYENV_VERSION="${VENV_NAME}" pyenv exec pip install -r "${REQUIREMENTS}"

# ── 8. Write .python-version so pyenv auto-selects the venv in this repo ─────
PYTHON_VERSION_FILE="${REPO_ROOT}/.python-version"
if [[ -f "${PYTHON_VERSION_FILE}" ]]; then
  EXISTING="$(cat "${PYTHON_VERSION_FILE}")"
  if [[ "${EXISTING}" == "${VENV_NAME}" ]]; then
    info ".python-version already set to '${VENV_NAME}'."
  else
    warn ".python-version exists with value '${EXISTING}' — leaving it unchanged."
    warn "To switch to '${VENV_NAME}', run:"
    warn "  echo '${VENV_NAME}' > .python-version"
  fi
else
  echo "${VENV_NAME}" > "${PYTHON_VERSION_FILE}"
  info "Wrote .python-version → '${VENV_NAME}'."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "✅  Bootstrap complete!"
info "   Python : ${PYTHON_VERSION}"
info "   Venv   : ${VENV_NAME}"
echo ""
info "pyenv will auto-activate '${VENV_NAME}' when you cd into this repo."
info "To activate manually in this session, run:"
info "   pyenv activate ${VENV_NAME}"
