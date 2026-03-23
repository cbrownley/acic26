# acic26
This repo contains code and files for the Society for Causal Inference's 2026 ACIC Data Challenge.

## Getting started

### Prerequisites

- macOS
- [Homebrew](https://brew.sh) — used to install `pyenv` if it isn't already present

### Setup

After cloning the repo, run the bootstrap script to install `pyenv`, set up the correct Python version, and create the project virtual environment:

```bash
./scripts/bootstrap.sh
```

This will:
1. Install `pyenv` via Homebrew if needed
2. Install Python `3.14.3` if not already available in pyenv
3. Create a pyenv virtualenv named `acic26`
4. Write a `.python-version` file so pyenv auto-activates the environment when you `cd` into the repo

The script is safe to rerun — it skips any steps that are already complete.

#### Overriding defaults

You can pass a different Python version or virtualenv name as positional arguments:

```bash
./scripts/bootstrap.sh [PYTHON_VERSION] [VENV_NAME]
# e.g.
./scripts/bootstrap.sh 3.13.2 my-env
```

#### Shell initialization

If `pyenv` was just installed for the first time, you may need to add the following to your shell config (`~/.zshrc` or `~/.bash_profile`) and open a new terminal before the auto-activation works:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

The bootstrap script will print this message and exit if it detects that pyenv isn't on your PATH yet.
