from contextlib import contextmanager, ExitStack

@contextmanager
def null_context():
    yield

def some_cb(*a, **kw):
    pass

@contextmanager
def inner_context():
    stack = ExitStack()
    with stack:
        stack.enter_context(null_context())
        stack.callback(some_cb, 10, "hi", answer=42)
        yield "inner"

@contextmanager
def outer_context():
    with inner_context() as inner:
        yield "outer"

def example():
    with outer_context():
        yield

def call_example():
    yield from example()

gen = call_example()
next(gen)

import stackscope

stack = stackscope.extract(gen)
print(stack)
print(stack.frames[1].contexts[0].inner_stack.frames[0].contexts[0])
