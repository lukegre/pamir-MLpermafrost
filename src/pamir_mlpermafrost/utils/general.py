def DummyContextManager(*args, **kwargs):
    """A dummy context manager that does nothing."""

    class DummyContext:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    return DummyContext()
