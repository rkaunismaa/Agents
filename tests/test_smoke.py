"""Smoke test: agentlab imports cleanly."""


def test_agentlab_imports():
    import agentlab
    assert agentlab.__version__ == "0.1.0"


def test_submodules_import():
    from agentlab import llm, tools, types  # noqa: F401
