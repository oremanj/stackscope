stackscope: unusually detailed Python stack introspection
=========================================================

.. image:: https://img.shields.io/pypi/v/stackscope.svg
   :target: https://pypi.org/project/stackscope
   :alt: Latest PyPI version

.. image:: https://img.shields.io/badge/docs-read%20now-blue.svg
   :target: https://stackscope.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://codecov.io/gh/oremanj/stackscope/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/oremanj/stackscope
   :alt: Test coverage

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://www.mypy-lang.org/
   :alt: Checked with mypy

``stackscope`` is a library that helps you tell what your running
Python program is doing and how it got there. It can provide detailed
stack traces, similar to what you get in an exception traceback, but
without needing to throw an exception first. Compared to standard
library facilities such as ``traceback.extract_stack()``, it is far
more versatile. It supports async tasks, generators, threads, and
greenlets; provides information about active context managers in each
stack frame; and includes a customization interface that library
authors can use to teach it to improve the stack extraction logic for
code that touches their library. (As an example of the latter, the
stack of an async task blocked in a ``run_in_thread()`` function could
be made to cover the code that's running in the thread as well.)

``stackscope`` is loosely affiliated with the `Trio
<https://trio.readthedocs.io/>`__ async framework, and shares Trio's
obsessive focus on usability and correctness. The context manager
analysis is especially helpful with Trio since you can use it to
understand where the nurseries are. You don't have to use ``stackscope``
with Trio, though; it requires only the Python standard library, 3.8
or later, and the `ExceptionGroup backport <https://pypi.org/project/exceptiongroup/>`__
on versions below 3.11.

``stackscope`` is mostly intended as a building block for other
debugging and introspection tools. You can use it directly, but
there's only rudimentary support for end-user-facing niceties such as
pretty-printed output. On the other hand, the core logic is (believed
to be) robust and flexible, exposing customization points that
third-party libraries can use to help ``stackscope`` make better
tracebacks for their code.  ``stackscope`` ships out of the box with such
"glue" for `Trio <https://trio.readthedocs.io/en/stable/>`__, `greenback
<https://greenback.readthedocs.io/en/latest/>`__, and some of their
lower-level dependencies.

``stackscope`` requires Python 3.8 or later. It is fully
type-annotated and is tested with CPython (every minor version through
3.12) and PyPy, on Linux, Windows, and macOS. It will probably
work on other operating systems.  Basic features will work on other
Python implementations, but the context manager decoding will be less
intelligent, and won't work at all without a usable
``gc.get_referents()``.

Quickstart
----------

Call ``stackscope.extract()`` to obtain a ``stackscope.Stack``
describing the stack of a coroutine object, generator iterator (sync
or async), greenlet, or thread. If you want to extract part of the
stack that led to the ``extract()`` call, then either pass a
``stackscope.StackSlice`` or use the convenience aliases
``stackscope.extract_since()`` and ``stackscope.extract_until()``.

Trio users: Try ``print(stackscope.extract(trio.lowlevel.current_root_task(),
recurse_child_tasks=True))`` to print the entire task tree of your
Trio program.

Once you have a ``Stack``, you can:

* Format it for human consumption: ``str()`` obtains a tree view as
  shown in the example below, or use ``stack.format()`` to customize
  it or ``stack.format_flat()`` to get an alternate format that
  resembles a standard Python traceback.

* Iterate over it (or equivalently, its ``frames`` attribute) to
  obtain a series of ``stackscope.Frame``\s for programmatic
  inspection.  Each frame represents one function call. In addition to
  the interpreter-level frame object, it lets you access information
  about the active context managers in that function.

* Look at its ``leaf`` attribute to see what's left once you
  peel away all the frames. For example, this might be some atomic
  awaitable such as an ``asyncio.Future``. It will be ``None`` if the
  frames tell the whole story.

* Use its ``as_stdlib_summary()`` method to get a standard library
  ``traceback.StackSummary`` object (with some loss of information),
  which can be pickled or passed to non-``stackscope``\-aware tools.

See the documentation for more details.

Example
-------

This code uses a number of context managers::

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

You can use ``stackscope`` to inspect the state of the partially-consumed generator
*gen*, showing the tree structure of all of those context managers::

    $ python3 -i example.py
    >>> import stackscope
    >>> stack = stackscope.extract(gen)
    >>> print(stack)
    stackscope.Stack (most recent call last):
    ╠ call_example in __main__ at [...]/stackscope/example.py:28
    ║ └ yield from example()
    ╠ example in __main__ at [...]/stackscope/example.py:25
    ║ ├ with outer_context():  # _: _GeneratorContextManager (line 24)
    ║ │ ╠ outer_context in __main__ at [...]/stackscope/example.py:21
    ║ │ ║ ├ with inner_context() as inner:  # inner: _GeneratorContextManager (line 20)
    ║ │ ║ │ ╠ inner_context in __main__ at [...]/stackscope/example.py:16
    ║ │ ║ │ ║ ├ with stack:  # stack: ExitStack (line 13)
    ║ │ ║ │ ║ ├── stack.enter_context(null_context(...))  # stack[0]: _GeneratorContextManager
    ║ │ ║ │ ║ │   ╠ null_context in __main__ at [...]/stackscope/example.py:5
    ║ │ ║ │ ║ │   ║ └ yield
    ║ │ ║ │ ║ ├── stack.callback(__main__.some_cb, 10, 'hi', answer=42)  # stack[1]: function
    ║ │ ║ │ ║ └ yield "inner"
    ║ │ ║ └ yield "outer"
    ║ └ yield

That full tree structure is exposed for programmatic inspection as well::

    >>> print(stack.frames[1].contexts[0].inner_stack.frames[0].contexts[0])
    inner_context(...)  # inner: _GeneratorContextManager (line 20)
    ╠ inner_context in __main__ at /Users/oremanj/dev/stackscope/example.py:16
    ║ ├ with stack:  # stack: ExitStack (line 13)
    ║ ├── stack.enter_context(null_context(...))  # stack[0]: _GeneratorContextManager
    ║ │   ╠ null_context in __main__ at /Users/oremanj/dev/stackscope/example.py:5
    ║ │   ║ └ yield
    ║ ├── stack.callback(__main__.some_cb, 10, 'hi', answer=42)  # stack[1]: function
    ║ └ yield "inner"

Of course, if you just want a "normal" stack trace without the added information,
you can get that too::

    >>> print("".join(stack.format_flat()))
    stackscope.Stack (most recent call last):
      File "/Users/oremanj/dev/stackscope/example.py", line 28, in call_example
        yield from example()
      File "/Users/oremanj/dev/stackscope/example.py", line 25, in example
        yield

Development status
------------------

While ``stackscope`` is a young project that deals with some obscure Python internals,
it is written quite defensively including 99%+ test coverage and type hints.
The author is using it (cautiously) in production and thinks you might want to as well.

License
-------

``stackscope`` is licensed under your choice of the MIT or Apache 2.0
license. See `LICENSE <https://github.com/oremanj/stackscope/blob/master/LICENSE>`__
for details.
