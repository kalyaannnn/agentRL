# Contributing

Thanks for contributing to AgentRL.

## Development Setup

```bash
pip install -e .
pip install -e ".[benchmark]"
pip install -e ".[dev]"
```

## Before Opening a PR

Run the focused test suite for the area you changed, and prefer adding or updating tests with every behavior change.

Examples:

```bash
pytest tests/test_base.py tests/test_trainer.py -q
pytest tests/test_rollout.py tests/test_continuous.py -q
pytest tests/test_examples.py -q
```

## Reporting Bugs

Include:

- exact command used
- config values or CLI flags
- model name
- hardware
- the last few lines of `metrics.jsonl` or stdout
- whether continuous batching, profiling, or any experimental flags were enabled

## Scope

AgentRL is intentionally focused on single-device verifier-based post-training. Keep changes narrow, well-tested, and explicit about tradeoffs.
