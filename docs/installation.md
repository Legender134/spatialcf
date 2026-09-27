# Installation

Use Python 3.11. Install current public `main` from a local checkout for the CPU examples:

```bash
git clone --branch main --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

The base package provides the platform-neutral Schema, solvers and verification.
No native runtime is required for `spatialcf general`. Check `spatialcf --help`;
it exposes `generate`, `verify`, `inspect` and the `general` group.

Record `git rev-parse HEAD` with each experiment. The metadata version remains
`0.1.1`; the historical `v0.1.1` annotated tag does not contain all current-main
interfaces. Neither the tag nor these commands are a PyPI publication.

Continue with [the CPU quick start](quickstart.md). Install `.[ai2thor]` only for
the separate [Adapter workflow](adapters.md); that route requires its native
runtime. Maintainers can install `.[test]` for public smoke tests in a separate
environment. No simulator starts during those synthetic smoke tests.
