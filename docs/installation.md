# Installation

SpatialCF supports Python 3.11. `v0.1.1` is a GitHub release, not a PyPI
publication. Clone that release tag, then create an isolated environment and
install the local package with the AI2-THOR Adapter:

```bash
git clone --branch v0.1.1 --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[ai2thor]"
```

The base package contains the platform-neutral Schema, solver, generation
contracts, and verification logic. The `ai2thor` extra adds the first Adapter,
which connects those contracts to Unity/AI2-THOR.

Confirm the command-line installation with:

```bash
spatialcf --help
```

The command list contains only `generate`, `verify`, and `inspect`.

For Python development and smoke tests, install the package's `test` extra in a
separate checkout environment.
