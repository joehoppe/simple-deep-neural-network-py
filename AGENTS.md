# Agent Instructions

When writing or modifying tests in this repository, use native pytest style.

- Write module-level `test_...` functions unless a class is genuinely useful.
- Use plain `assert` statements instead of `unittest.TestCase` assertions.
- Use `pytest.raises(...)` for expected exceptions.
- Prefer pytest fixtures and parametrization when they reduce repetition.
- Do not add `if __name__ == "__main__": unittest.main()` blocks.
