import contextlib
import gc
import re
import sys
import time
import threading
import types
import dataclasses
from contextlib import ExitStack, AsyncExitStack, contextmanager
from functools import partial
from typing import List, Callable, Any, AsyncIterator, cast, TYPE_CHECKING

import pytest

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from stackscope import (
    Context,
    Frame,
    Stack,
    StackSlice,
    customize,
    elaborate_frame,
    extract,
    extract_since,
    extract_until,
    extract_outermost,
    extract_child,
    fill_context,
    unwrap_stackitem,
    unwrap_context,
    unwrap_context_generator,
)


def remove_address_details(line):
    return re.sub(r"\b0x[0-9A-Fa-f]+\b", "(address)", line)


def clean_line(line):
    return remove_address_details(line).partition("  #")[0]


def flatten(frames):
    for frame in frames:
        if frame.hide:
            continue
        for summary in frame.as_stdlib_summary_with_contexts():
            func, _, context_info = summary.name.partition(" ")
            line = clean_line(summary.line)
            if context_info:
                if ":" in context_info:
                    context_name, context_typename = context_info.strip(" ()").split(
                        ": "
                    )
                    if context_name == "_":
                        context_name = None
                    yield func, line, context_name, context_typename
                else:  # pragma: no cover
                    yield func, line, context_info.strip(" ()"), None
            else:
                yield func, line, None, None


def assert_stack_matches(stack, expected, leaf_typename=None, error=None):
    # smoke test:
    str(stack)
    list(stack)
    list(stack.format_flat())
    list(stack.format(show_contexts=False))
    stack.as_stdlib_summary()
    stack.as_stdlib_summary(show_contexts=True, capture_locals=True)
    for frame in stack.frames:
        str(frame)

    try:
        if error is None and stack.error is not None:  # pragma: no cover
            raise stack.error
        if leaf_typename is None:
            assert stack.leaf is None
        else:
            assert type(stack.leaf).__name__ == leaf_typename
        assert type(stack.error) is type(error)
        assert remove_address_details(str(stack.error)) == remove_address_details(
            str(error)
        )
        assert list(flatten(stack.frames)) == expected
    except Exception:  # pragma: no cover
        print_assert_matches("stack")
        raise


def print_assert_matches(get_stack):  # pragma: no cover
    parent = sys._getframe(1)
    get_stack_code = compile(get_stack, "<eval>", "eval")
    stack = eval(get_stack_code, parent.f_globals, parent.f_locals)
    print("---")
    print(str(stack).rstrip())
    print("---")
    print("    assert_stack_matches(")
    print("        " + get_stack + ",")
    print("        [")
    for frame in stack.frames:
        for func, line, ctxname, ctxtype in flatten([frame]):
            if frame.pyframe.f_code is get_stack_code:
                func = parent.f_code.co_name
                line = get_stack + ","
            record = (func, line, ctxname, ctxtype)
            print("            " + repr(record) + ",")
    print("        ],")
    if stack.leaf:
        print(f"        leaf_typename={type(stack.leaf).__name__!r},")
    if stack.error:
        print(f"        error={remove_address_details(repr(stack.error))},")
    print("    )")


def no_abort(_):  # pragma: no cover
    import trio

    return trio.lowlevel.Abort.FAILED


@contextmanager
def null_context():
    yield


@contextmanager
def outer_context():
    with inner_context() as inner:  # noqa: F841
        yield


def exit_cb(*exc):
    pass


def other_cb(*a, **kw):
    pass


@contextmanager
def inner_context():
    stack = ExitStack()
    with stack:
        stack.enter_context(null_context())
        stack.push(exit_cb)
        stack.callback(other_cb, 10, "hi", answer=42)
        yield


@types.coroutine
def async_yield(value):
    return (yield value)


# There's some logic in the stack extraction of running code that
# behaves differently when it's run in a non-main greenlet on CPython,
# because we have to stitch together the stack portions from
# different greenlets. To exercise it, we'll run some tests in a
# non-main greenlet as well as at top level.
try:
    import greenlet  # type: ignore
except ImportError:

    def try_in_other_greenlet_too(fn):
        return fn

else:

    def try_in_other_greenlet_too(fn):
        def try_both():
            fn()
            greenlet.greenlet(fn).switch()

        return try_both


def frames_from_inner_context(caller):
    return [
        (
            caller,
            "with inner_context() as inner:",
            "inner",
            "_GeneratorContextManager",
        ),
        ("inner_context", "with stack:", "stack", "ExitStack"),
        (
            "inner_context",
            "# stack.enter_context(null_context(...))",
            "stack[0]",
            "_GeneratorContextManager",
        ),
        ("null_context", "yield", None, None),
        (
            "inner_context",
            "# stack.push(stackscope._tests.test_traceback.exit_cb)",
            "stack[1]",
            "function",
        ),
        (
            "inner_context",
            "# stack.callback(stackscope._tests.test_traceback.other_cb, 10, 'hi', answer=42)",
            "stack[2]",
            "function",
        ),
        ("inner_context", "yield", None, None),
    ]


def frames_from_outer_context(caller):
    return [
        (caller, "with outer_context():", None, "_GeneratorContextManager"),
        *frames_from_inner_context("outer_context"),
        ("outer_context", "yield", None, None),
    ]


@try_in_other_greenlet_too
def test_running():
    # These two layers of indirection are mostly to test that pruning works
    @customize(hide=True, prune=True)
    def call_call_extract_since(root):
        return call_extract_since(root)

    def call_extract_since(root):
        return extract_since(root)

    def sync_example(root):
        with outer_context():
            if isinstance(root, types.FrameType):
                return call_call_extract_since(root)
            else:
                return extract(root)

    # Currently running in this thread
    assert_stack_matches(
        sync_example(sys._getframe(0)),
        [
            ("test_running", "sync_example(sys._getframe(0)),", None, None),
            *frames_from_outer_context("sync_example"),
            ("sync_example", "return call_call_extract_since(root)", None, None),
        ],
    )

    async def async_example():
        root = await async_yield(None)
        await async_yield(sync_example(root))

    def generator_example():
        root = yield
        yield sync_example(root)

    async def agen_example():
        root = yield
        yield sync_example(root)

    for which in (async_example, generator_example, agen_example):
        it = which()
        if which is agen_example:

            def send(val):
                with pytest.raises(StopIteration) as info:
                    it.asend(val).send(None)
                return info.value.value

        else:
            send = it.send
        send(None)
        if which is async_example:
            line = "await async_yield(sync_example(root))"
        else:
            line = "yield sync_example(root)"
        assert_stack_matches(
            send(it),
            [
                (which.__name__, line, None, None),
                *frames_from_outer_context("sync_example"),
                ("sync_example", "return extract(root)", None, None),
            ],
        )


def test_suspended():
    async def async_example(depth):
        if depth >= 1:
            return await async_example(depth - 1)
        with outer_context():
            return await async_yield(1)

    async def agen_example(depth):
        await async_example(depth)
        yield

    agen_makers = [agen_example]

    try:
        import async_generator
    except ImportError:
        agen_backport_example = None
    else:

        @async_generator.async_generator
        async def agen_backport_example(depth):
            await async_example(depth)
            await async_generator.yield_()

        agen_makers.append(agen_backport_example)

    # Suspended coroutine
    coro = async_example(3)
    assert coro.send(None) == 1
    assert_stack_matches(
        extract(coro),
        [
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            *frames_from_outer_context("async_example"),
            ("async_example", "return await async_yield(1)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
    assert_stack_matches(
        extract(coro, with_contexts=False),
        [
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_yield(1)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
    with pytest.raises(StopIteration, match="42"):
        coro.send(42)

    # Suspended async generator
    for thing in agen_makers:
        agi = thing(3)
        ags = agi.asend(None)
        assert ags.send(None) == 1
        for view in (agi, ags):
            assert_stack_matches(
                extract(view, with_contexts=False),
                [
                    (thing.__name__, "await async_example(depth)", None, None),
                    (
                        "async_example",
                        "return await async_example(depth - 1)",
                        None,
                        None,
                    ),
                    (
                        "async_example",
                        "return await async_example(depth - 1)",
                        None,
                        None,
                    ),
                    (
                        "async_example",
                        "return await async_example(depth - 1)",
                        None,
                        None,
                    ),
                    ("async_example", "return await async_yield(1)", None, None),
                    ("async_yield", "return (yield value)", None, None),
                ],
            )
        with pytest.raises(StopIteration):
            ags.send(None)
        with pytest.raises(StopAsyncIteration):
            agi.asend(None).send(None)

    # Exhausted coro/generator has no stack
    assert_stack_matches(extract(coro), [])


def test_greenlet():
    greenlet = pytest.importorskip("greenlet")

    stack_main = extract(greenlet.getcurrent())
    assert (
        stack_main.error is None and stack_main.frames[-1].funcname == "test_greenlet"
    )

    def outer():
        with outer_context():
            return inner()

    def inner():
        # Test getting the stack of a greenlet from inside it
        assert_stack_matches(
            extract(gr),
            [
                *frames_from_outer_context("outer"),
                ("outer", "return inner()", None, None),
                ("inner", "extract(gr),", None, None),
            ],
        )
        return greenlet.getcurrent().parent.switch(1)

    gr = greenlet.greenlet(outer)
    assert_stack_matches(extract(gr), [])  # not started -> empty stack

    assert 1 == gr.switch()
    assert_stack_matches(
        extract(gr),
        [
            *frames_from_outer_context("outer"),
            ("outer", "return inner()", None, None),
            ("inner", "return greenlet.getcurrent().parent.switch(1)", None, None),
        ],
    )

    assert 2 == gr.switch(2)
    assert_stack_matches(extract(gr), [])  # dead -> empty stack

    # Test tracing into the runner for a dead greenlet

    def trivial_runner(gr):
        assert_stack_matches(
            extract_since(sys._getframe(0)),
            [("trivial_runner", "extract_since(sys._getframe(0)),", None, None)],
        )

    @elaborate_frame.register(trivial_runner)
    def elaborate(frame, next_inner):
        return frame.pyframe.f_locals.get("gr")

    trivial_runner(gr)


def test_elaborate_fails():
    outer_frame = sys._getframe(0)

    def inner():
        return extract_since(outer_frame)

    @customize(elaborate=lambda *args: {}["wheee"])
    def example():
        return inner()

    # Frames that produce an error get mentioned in the stack,
    # even if they'd otherwise be skipped
    @customize(hide=True, elaborate=lambda *args: {}["wheee"])
    def skippy_example():
        return inner()

    for fn in (example, skippy_example):
        assert_stack_matches(
            fn(),
            [
                ("test_elaborate_fails", "fn(),", None, None),
                (fn.__name__, "return inner()", None, None),
            ],
            error=KeyError("wheee"),
        )


@pytest.mark.skipif(
    sys.implementation.name == "pypy",
    reason="https://foss.heptapod.net/pypy/pypy/-/blob/branch/py3.6/lib_pypy/greenlet.py#L124",
)
def test_greenlet_in_other_thread():
    greenlet = pytest.importorskip("greenlet")
    ready_evt = threading.Event()
    done_evt = threading.Event()
    gr = None

    def thread_fn():
        def target():
            ready_evt.set()
            done_evt.wait()

        nonlocal gr
        gr = greenlet.greenlet(target)
        gr.switch()

    threading.Thread(target=thread_fn).start()
    ready_evt.wait()
    assert_stack_matches(
        extract(gr),
        [],
        leaf_typename="greenlet",
        error=RuntimeError(
            "Can't dump the stack of a greenlet running in another thread"
        ),
    )
    done_evt.set()


def test_exiting() -> None:
    # Test stack when a synchronous context manager is currently exiting.
    result: Stack

    @contextmanager
    def capture_stack_on_exit(coro):
        with inner_context() as inner:  # noqa: F841
            try:
                yield
            finally:
                nonlocal result
                result = extract(coro)

    async def async_capture_stack():
        coro = await async_yield(None)
        with capture_stack_on_exit(coro):
            pass
        await async_yield(result)

    coro = async_capture_stack()
    coro.send(None)
    assert_stack_matches(
        coro.send(coro),
        [
            (
                "async_capture_stack",
                "with capture_stack_on_exit(coro):",
                None,
                "_GeneratorContextManager",
            ),
            ("__exit__", "next(self.gen)", None, None),
            *frames_from_inner_context("capture_stack_on_exit"),
            ("capture_stack_on_exit", "result = extract(coro)", None, None),
        ],
    )

    # Test stack when an async CM is suspended in __aexit__.  The
    # definition of __aexit__ as a staticmethod is to foil the logic
    # for figuring out which context manager is exiting.

    class SillyAsyncCM:
        async def __aenter__(self):
            pass

        @staticmethod
        async def __aexit__(*stuff):
            await async_yield(None)

    async def yield_when_async_cm_exiting():
        async with SillyAsyncCM():
            pass

    coro = yield_when_async_cm_exiting()
    coro.send(None)
    assert_stack_matches(
        extract(coro),
        [
            ("yield_when_async_cm_exiting", "async with SillyAsyncCM():", None, None),
            ("__aexit__", "await async_yield(None)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )


def test_errors():
    with pytest.raises(TypeError, match="must be a frame"):
        extract_since(42)
    with pytest.raises(TypeError, match="must be a frame or integer"):
        extract_until(sys._getframe(0), limit=2.4)
    with pytest.raises(RuntimeError, match="is not an indirect caller of"):
        extract_until(sys._getframe(1), limit=sys._getframe(0))


@try_in_other_greenlet_too
def test_extract_until():
    outer = sys._getframe(0)

    def example():
        inner = sys._getframe(0)

        def get_stack(limit):
            return extract_until(inner, limit=limit)

        stack1, stack2, stack3 = [get_stack(lim) for lim in (2, outer, None)]
        assert stack1 == stack2
        assert stack3.frames[-len(stack1) :] == stack1.frames
        assert_stack_matches(
            stack1,
            [
                ("test_extract_until", "example()", None, None),
                (
                    "example",
                    "stack1, stack2, stack3 = [get_stack(lim) for lim in (2, outer, None)]",
                    None,
                    None,
                ),
            ],
        )

        stack4 = extract(StackSlice(outer=outer, limit=1))
        assert len(stack4.frames) == 1
        assert stack4.frames[0] == stack1.frames[0]

    example()


@try_in_other_greenlet_too
def test_running_in_thread():
    def thread_example(arrived_evt, depart_evt):
        with outer_context():
            arrived_evt.set()
            depart_evt.wait()

    def thread_caller(*args):
        thread_example(*args)

    # Currently running in other thread
    for cooked in (False, True):
        arrived_evt = threading.Event()
        depart_evt = threading.Event()
        thread = threading.Thread(target=thread_caller, args=(arrived_evt, depart_evt))
        thread.start()
        try:

            def trim_threading_internals(stack):  # pragma: no cover
                # Exactly where we are inside Event.wait() is indeterminate, so
                # strip frames until we find Event.wait() and then remove it.
                # We could also be in Event.set().
                while stack.frames and (
                    not stack.frames[-1].filename.endswith("threading.py")
                    or stack.frames[-1].funcname not in ("set", "wait")
                ):
                    stack.frames = stack.frames[:-1]
                if not stack.frames:
                    return False
                while stack.frames and stack.frames[-1].filename.endswith(
                    "threading.py"
                ):
                    stack.frames = stack.frames[:-1]
                return True

            arrived_evt.wait()
            deadline = time.monotonic() + 10
            while True:
                if cooked:
                    stack = extract(thread)
                else:
                    top_frame = sys._current_frames()[thread.ident]
                    while (
                        top_frame.f_back is not None
                        and top_frame.f_code.co_name != "thread_caller"
                    ):
                        top_frame = top_frame.f_back
                    stack = extract_since(top_frame)
                if time.monotonic() >= deadline:  # pragma: no cover
                    sys.stderr.write("{}\n".format(stack))
                    pytest.fail("Couldn't get a thread traceback in Event.wait()")
                if trim_threading_internals(stack):  # pragma: no branch
                    break

            assert_stack_matches(
                stack,
                [
                    ("thread_caller", "thread_example(*args)", None, None),
                    *frames_from_outer_context("thread_example"),
                    ("thread_example", "depart_evt.wait()", None, None),
                ],
            )
        finally:
            depart_evt.set()


def test_stack_of_not_alive_thread(local_registry):
    thread = threading.Thread(target=lambda: None)
    assert_stack_matches(extract(thread), [])
    thread.start()
    thread.join()
    assert_stack_matches(extract(thread), [])

    @customize(elaborate=lambda *_: thread)
    async def example():
        await async_yield(42)

    coro = example()
    coro.send(None)
    assert_stack_matches(
        extract(coro),
        [
            ("example", "await async_yield(42)", None, None),
        ],
    )


def test_trace_into_thread(local_registry):
    trio = pytest.importorskip("trio")
    import outcome

    # Extremely simplified version of trio.to_thread.run_sync
    async def run_sync_in_thread(sync_fn):
        task = trio.lowlevel.current_task()
        trio_token = trio.lowlevel.current_trio_token()

        def run_it():
            result = outcome.capture(sync_fn)
            trio_token.run_sync_soon(trio.lowlevel.reschedule, task, result)

        thread = threading.Thread(target=run_it)
        thread.start()
        return await trio.lowlevel.wait_task_rescheduled(no_abort)

    @elaborate_frame.register(run_sync_in_thread)
    def get_target(frame, next_inner):
        return (frame.pyframe.f_locals["thread"], next_inner)

    customize(run_sync_in_thread, "run_it", hide=True)

    stack = None

    async def main():
        arrived_evt = trio.Event()
        depart_evt = threading.Event()
        trio_token = trio.lowlevel.current_trio_token()
        task = trio.lowlevel.current_task()

        def sync_fn():
            with inner_context() as inner:  # noqa: F841
                trio_token.run_sync_soon(arrived_evt.set)
                depart_evt.wait()

        def sync_wrapper():
            sync_fn()

        async def capture_stack():
            nonlocal stack
            try:
                await arrived_evt.wait()
                stack = extract(task.coro)
            finally:
                depart_evt.set()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(capture_stack)
            await run_sync_in_thread(sync_wrapper)

    trio.run(main)

    # It's indeterminate where in sync_fn() the stack was taken -- it could
    # be inside run_sync_soon() or inside threading.Event.wait() -- so trim
    # stack frames until we get something reliable.
    while stack.frames[-1].filename != __file__:
        stack.frames = stack.frames[:-1]
    # <indeterminate>
    stack.frames[-1].lineno = sys._getframe(0).f_lineno - 1
    assert_stack_matches(
        stack,
        [
            (
                "main",
                "async with trio.open_nursery() as nursery:",
                "nursery",
                "Nursery",
            ),
            ("main", "await run_sync_in_thread(sync_wrapper)", None, None),
            (
                "run_sync_in_thread",
                "return await trio.lowlevel.wait_task_rescheduled(no_abort)",
                None,
                None,
            ),
            ("sync_wrapper", "sync_fn()", None, None),
            *frames_from_inner_context("sync_fn"),
            ("sync_fn", "# <indeterminate>", None, None),
        ],
    )


def test_unknown_awaitable():
    class WeirdObject:
        def __await__(self):
            return iter([42])

    async def example():
        await WeirdObject()

    coro = example()
    assert 42 == coro.send(None)
    name = "sequence" if sys.implementation.name == "pypy" else "list_"
    assert_stack_matches(
        extract(coro),
        [("example", "await WeirdObject()", None, None)],
        leaf_typename=f"{name}iterator",
    )

    assert_stack_matches(
        extract(42),
        [],
        leaf_typename="int",
    )


def test_cant_get_referents(monkeypatch):
    async def agen():
        await async_yield(1)
        yield

    async def afn():
        await async_yield(1)

    class SomeAwaitable:
        def __await__(self):
            return wrapper

    ags = agen().asend(None)
    wrapper = afn().__await__()
    real_get_referents = gc.get_referents

    def patched_get_referents(obj):
        if obj is ags or obj is wrapper:
            return []
        return real_get_referents(obj)

    monkeypatch.setattr(gc, "get_referents", patched_get_referents)

    async def await_it(thing):
        await thing

    for thing, problem, attrib in (
        (ags, ags, "an ag_frame"),
        (SomeAwaitable(), wrapper, "a cr_frame"),
    ):
        coro = await_it(thing)
        assert 1 == coro.send(None)
        assert_stack_matches(
            extract(coro),
            [("await_it", "await thing", None, None)],
            leaf_typename=type(problem).__name__,
            error=RuntimeError(
                f"{problem!r} doesn't refer to anything with {attrib} attribute"
            ),
        )
        with pytest.raises(StopIteration):
            coro.send(None)


def test_cant_find_running_frame():
    greenlet = pytest.importorskip("greenlet")

    async def caller():
        await example()

    async def example():
        with outer_context():
            greenlet.getcurrent().parent.switch(42)

    for with_other_thread in (False, True):
        if with_other_thread:
            event = threading.Event()
            thread = threading.Thread(target=event.wait)
            thread.start()
        coro = caller()
        gr = greenlet.greenlet(coro.send)
        assert gr.switch(None) == 42
        assert_stack_matches(
            extract(coro),
            [("caller", "await example()", None, None)],
            error=RuntimeError(
                "Couldn't find where the above frame is running, so can't continue "
                "traceback"
            ),
        )
        with pytest.raises(StopIteration):
            gr.switch(None)
        if with_other_thread:
            event.set()
            thread.join()


def test_with_trickery_disabled(trickery_disabled):
    import stackscope

    def sync_example(root):
        with outer_context():
            return extract_since(root)

    # CPython GC doesn't crawl currently executing frames, so we get more
    # data without trickery on PyPy than on CPython
    only_on_pypy = [
        ("sync_example", "", None, "_GeneratorContextManager"),
        ("outer_context", "", None, "_GeneratorContextManager"),
        ("inner_context", "", None, "ExitStack"),
        (
            "inner_context",
            "# _.enter_context(null_context(...))",
            "_[0]",
            "_GeneratorContextManager",
        ),
        ("null_context", "yield", None, None),
        (
            "inner_context",
            "# _.push(stackscope._tests.test_traceback.exit_cb)",
            "_[1]",
            "function",
        ),
        (
            "inner_context",
            "# _.callback(stackscope._tests.test_traceback.other_cb, 10, 'hi', answer=42)",
            "_[2]",
            "function",
        ),
        ("inner_context", "yield", None, None),
        ("outer_context", "yield", None, None),
    ]
    assert_stack_matches(
        sync_example(sys._getframe(0)),
        [
            (
                "test_with_trickery_disabled",
                "sync_example(sys._getframe(0)),",
                None,
                None,
            ),
            *(only_on_pypy if sys.implementation.name == "pypy" else []),
            ("sync_example", "return extract_since(root)", None, None),
        ],
    )

    async def async_example():
        with outer_context():
            return await async_yield(42)

    coro = async_example()
    assert 42 == coro.send(None)
    assert_stack_matches(
        extract(coro),
        [
            ("async_example", "", None, "_GeneratorContextManager"),
            ("outer_context", "", None, "_GeneratorContextManager"),
            ("inner_context", "", None, "ExitStack"),
            (
                "inner_context",
                "# _.enter_context(null_context(...))",
                "_[0]",
                "_GeneratorContextManager",
            ),
            ("null_context", "yield", None, None),
            (
                "inner_context",
                "# _.push(stackscope._tests.test_traceback.exit_cb)",
                "_[1]",
                "function",
            ),
            (
                "inner_context",
                "# _.callback(stackscope._tests.test_traceback.other_cb, 10, 'hi', answer=42)",
                "_[2]",
                "function",
            ),
            ("inner_context", "yield", None, None),
            ("outer_context", "yield", None, None),
            ("async_example", "return await async_yield(42)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )


def test_trio_nursery():
    trio = pytest.importorskip("trio")
    async_generator = pytest.importorskip("async_generator")

    @async_generator.asynccontextmanager
    @async_generator.async_generator
    async def uses_nursery():
        async with trio.open_nursery() as inner:  # noqa: F841
            inner.start_soon(trio.sleep_forever, name="inner_child")
            await async_generator.yield_()
            inner.cancel_scope.cancel()

    async def main(recurse: bool) -> Stack:
        result: Stack
        task = trio.lowlevel.current_task()

        def report_back():
            nonlocal result
            result = extract(task.coro, recurse_child_tasks=recurse)
            trio.lowlevel.reschedule(task)

        async with trio.open_nursery() as outer, uses_nursery():  # noqa: F841
            outer.start_soon(trio.sleep_forever, name="outer_child1")
            outer.start_soon(trio.sleep_forever, name="outer_child2")
            trio.lowlevel.current_trio_token().run_sync_soon(report_back)
            await trio.lowlevel.wait_task_rescheduled(no_abort)
            outer.cancel_scope.cancel()

        return result

    for recurse in (False, True):
        stack = trio.run(main, recurse)
        # assert_stack_matches() uses flatten() which doesn't show child tasks,
        # so produces the same output for both
        assert_stack_matches(
            stack,
            [
                (
                    "main",
                    "async with trio.open_nursery() as outer, uses_nursery():",
                    "outer",
                    "Nursery",
                ),
                (
                    "main",
                    "async with trio.open_nursery() as outer, uses_nursery():",
                    None,
                    "_AsyncGeneratorContextManager",
                ),
                (
                    "uses_nursery",
                    "async with trio.open_nursery() as inner:",
                    "inner",
                    "Nursery",
                ),
                ("uses_nursery", "await async_generator.yield_()", None, None),
                (
                    "main",
                    "await trio.lowlevel.wait_task_rescheduled(no_abort)",
                    None,
                    None,
                ),
            ],
        )
        print(stack)
        cstacks = (
            stack.frames[0].contexts[0].children
            + stack.frames[0].contexts[1].inner_stack.frames[0].contexts[0].children
        )
        if cstacks[0].root.name > cstacks[1].root.name:
            cstacks[0], cstacks[1] = cstacks[1], cstacks[0]
        names = [cstack.root.name for cstack in cstacks]
        assert names == ["outer_child1", "outer_child2", "inner_child"]
        if recurse:
            for cstack in cstacks:
                assert cstack.frames[0].funcname == "sleep_forever"
        else:
            assert all(not cstack.frames for cstack in cstacks)


def test_trio_threads():
    if TYPE_CHECKING:
        import trio
        import outcome
    else:
        trio = pytest.importorskip("trio")
        outcome = pytest.importorskip("outcome")
    main_task = None

    async def back_in_trio_fn() -> None:
        stack = extract(main_task.coro)
        # This tests tracing from a to_thread.run_sync fn back into Trio
        # when it calls from_thread.run
        assert_stack_matches(
            stack,
            [
                ("main", "await trio.to_thread.run_sync(thread_fn)", None, None),
                ("thread_fn", "trio.from_thread.run(back_in_trio_fn)", None, None),
                ("back_in_trio_fn", "stack = extract(main_task.coro)", None, None),
            ],
        )

    def thread_fn() -> None:
        stack = extract(main_task.coro)
        # Trim off everything inside extract():
        for i, frame in enumerate(stack.frames):  # pragma: no branch
            if frame.pyframe is sys._getframe(0):
                stack.frames = stack.frames[: i + 1]
                break
        # This tests tracing from a to_thread.run_sync call into the thread
        # that it spawned
        assert_stack_matches(
            stack,
            [
                ("main", "await trio.to_thread.run_sync(thread_fn)", None, None),
                ("thread_fn", "stack = extract(main_task.coro)", None, None),
            ],
        )
        trio.from_thread.run(back_in_trio_fn)

    async def trio_from_other_thread_fn(
        other_thread: threading.Thread, done_evt: trio.Event
    ) -> None:
        stack = extract(other_thread)
        # This tests tracing from a non-Trio-originated thread into the
        # Trio task that it created using from_thread.run
        assert_stack_matches(
            stack,
            [
                (
                    "other_thread_fn",
                    "other_thread_outcome = outcome.capture(",
                    None,
                    None,
                ),
                (
                    "trio_from_other_thread_fn",
                    "stack = extract(other_thread)",
                    None,
                    None,
                ),
            ],
        )

    other_thread_outcome = None

    def other_thread_fn(
        trio_token: trio.lowlevel.TrioToken, done_evt: trio.Event
    ) -> None:
        nonlocal other_thread_outcome
        other_thread_outcome = outcome.capture(
            trio.from_thread.run,
            trio_from_other_thread_fn,
            threading.current_thread(),
            done_evt,
            trio_token=trio_token,
        )
        trio_token.run_sync_soon(done_evt.set)

    async def main() -> None:
        nonlocal main_task
        main_task = trio.lowlevel.current_task()
        await trio.to_thread.run_sync(thread_fn)

        # Now try with the callstack originating outside Trio, to test
        # the from_thread.run() glue
        done_evt = trio.Event()
        other_thread = threading.Thread(
            target=other_thread_fn, args=(trio.lowlevel.current_trio_token(), done_evt)
        )
        other_thread.start()
        await done_evt.wait()
        other_thread_outcome.unwrap()

    trio.run(main)


def test_pytest_trio_glue() -> None:
    plugin = pytest.importorskip("pytest_trio.plugin")
    import contextvars
    import trio
    import trio.testing

    tasks: list[trio.lowlevel.Task] = []
    stacks: dict[str, Stack] = {}

    def nursery() -> Any:
        return plugin.NURSERY_FIXTURE_PLACEHOLDER

    async def completed(nursery: trio.Nursery) -> int:
        nursery.start_soon(trio.sleep_forever)
        await trio.sleep(0)
        return 42

    async def generator(completed: int) -> AsyncIterator[trio.Nursery]:
        assert completed == 42
        async with trio.open_nursery() as nursery:
            nursery.start_soon(trio.sleep_forever)
            yield nursery
            nursery.cancel_scope.cancel()

    continue_evt = trio.Event()

    async def in_progress() -> AsyncIterator[int]:
        await continue_evt.wait()
        yield 50

    async def examine(generator: trio.Nursery) -> int:
        await trio.testing.wait_all_tasks_blocked()
        for task in tasks:
            stacks[task.name] = extract(task.coro)
        continue_evt.set()
        return 100

    async def late(examine: int, in_progress: int) -> int:
        assert examine == 100 and in_progress == 50
        return 200

    async def test(generator: trio.Nursery, late: int) -> None:
        assert late == 200
        assert stacks["generator"].frames[-1].pyframe.f_locals["nursery"] is generator

    test_ctx = plugin.TrioTestContext()
    nursery_fix = plugin.TrioFixture("nursery", nursery, {})
    completed_fix = plugin.TrioFixture("completed", completed, {"nursery": nursery_fix})
    generator_fix = plugin.TrioFixture(
        "generator", generator, {"completed": completed_fix}
    )
    in_progress_fix = plugin.TrioFixture("in_progress", in_progress, {})
    examine_fix = plugin.TrioFixture("examine", examine, {"generator": generator_fix})
    late_fix = plugin.TrioFixture(
        "late", late, {"examine": examine_fix, "in_progress": in_progress_fix}
    )
    test_fix = plugin.TrioFixture(
        "test", test, {"generator": generator_fix, "late": late_fix}, is_test=True
    )

    async def main() -> None:
        contextvars_ctx = contextvars.copy_context()
        contextvars_ctx.run(plugin.canary.set, "in correct context")

        async with trio.open_nursery() as outer_nursery:
            for fixture in test_fix.register_and_collect_dependencies():
                outer_nursery.start_soon(
                    fixture.run, test_ctx, contextvars_ctx, name=fixture.name
                )
            tasks.extend(outer_nursery.child_tasks)

    trio.run(main)

    assert_stack_matches(
        stacks["completed"],
        [
            (
                "run",
                "async with self._fixture_manager(test_ctx) as nursery_fixture:",
                "nursery_fixture",
                "Nursery",
            ),
            ("run", "await event.wait()", None, None),
            ("wait", "await _core.wait_task_rescheduled(abort_fn)", None, None),
        ],
    )
    assert_stack_matches(
        stacks["generator"],
        [
            ("run", "await event.wait()", None, None),
            (
                "generator",
                "async with trio.open_nursery() as nursery:",
                "nursery",
                "Nursery",
            ),
            ("generator", "yield nursery", None, None),
        ],
    )
    assert_stack_matches(
        stacks["examine"],
        [
            ("run", "self.fixture_value = await func_value", None, None),
            ("examine", "stacks[task.name] = extract(task.coro)", None, None),
        ],
    )
    assert_stack_matches(
        stacks["in_progress"],
        [
            ("run", "self.fixture_value = await func_value.asend(None)", None, None),
            ("in_progress", "await continue_evt.wait()", None, None),
            ("wait", "await _core.wait_task_rescheduled(abort_fn)", None, None),
        ],
    )
    assert_stack_matches(
        stacks["late"],
        [
            ("run", "await value.setup_done.wait()", None, None),
            ("wait", "await _core.wait_task_rescheduled(abort_fn)", None, None),
        ],
    )


def test_greenback() -> None:
    trio = pytest.importorskip("trio")
    greenback = pytest.importorskip("greenback")
    results: List[Stack] = []

    async def outer():
        async with trio.open_nursery() as outer_nursery:  # noqa: F841
            middle()
            await inner()

    def middle():
        nursery_mgr = trio.open_nursery()
        with greenback.async_context(nursery_mgr) as middle_nursery:  # noqa: F841
            greenback.await_(inner())

            # This winds up traversing an await_ before it has a coroutine to use.
            class ExtractWhenAwaited:
                def __await__(self):
                    task = trio.lowlevel.current_task()
                    assert_stack_matches(
                        extract(task.coro),
                        [
                            (
                                "greenback_shim",
                                "return await _greenback_shim(orig_coro)",
                                None,
                                None,
                            ),
                            ("main", "return await outer()", None, None),
                            (
                                "outer",
                                "async with trio.open_nursery() as outer_nursery:",
                                "outer_nursery",
                                "Nursery",
                            ),
                            ("outer", "middle()", None, None),
                            (
                                "middle",
                                "with greenback.async_context(nursery_mgr) as middle_nursery:",
                                "middle_nursery",
                                "Nursery",
                            ),
                            (
                                "middle",
                                "greenback.await_(ExtractWhenAwaited())",
                                None,
                                None,
                            ),
                            ("adapt_awaitable", "return await aw", None, None),
                            ("__await__", "extract(task.coro),", None, None),
                        ],
                    )
                    yield from ()

            greenback.await_(ExtractWhenAwaited())  # pragma: no cover

    async def inner():
        with null_context():
            task = trio.lowlevel.current_task()

            def report_back():
                results.append(extract(task.coro))
                trio.lowlevel.reschedule(task)

            trio.lowlevel.current_trio_token().run_sync_soon(report_back)
            await trio.lowlevel.wait_task_rescheduled(no_abort)

    async def main():
        await greenback.ensure_portal()
        return await outer()

    trio.run(main)
    assert len(results) == 2
    assert_stack_matches(
        results[0],
        [
            (
                "greenback_shim",
                "return await _greenback_shim(orig_coro)",
                None,
                None,
            ),
            ("main", "return await outer()", None, None),
            (
                "outer",
                "async with trio.open_nursery() as outer_nursery:",
                "outer_nursery",
                "Nursery",
            ),
            ("outer", "middle()", None, None),
            (
                "middle",
                "with greenback.async_context(nursery_mgr) as middle_nursery:",
                "middle_nursery",
                "Nursery",
            ),
            ("middle", "greenback.await_(inner())", None, None),
            ("inner", "with null_context():", None, "_GeneratorContextManager"),
            ("null_context", "yield", None, None),
            (
                "inner",
                "await trio.lowlevel.wait_task_rescheduled(no_abort)",
                None,
                None,
            ),
        ],
    )
    assert_stack_matches(
        results[1],
        [
            (
                "greenback_shim",
                "return await _greenback_shim(orig_coro)",
                None,
                None,
            ),
            ("main", "return await outer()", None, None),
            (
                "outer",
                "async with trio.open_nursery() as outer_nursery:",
                "outer_nursery",
                "Nursery",
            ),
            ("outer", "await inner()", None, None),
            ("inner", "with null_context():", None, "_GeneratorContextManager"),
            ("null_context", "yield", None, None),
            (
                "inner",
                "await trio.lowlevel.wait_task_rescheduled(no_abort)",
                None,
                None,
            ),
        ],
    )


def test_exitstack_formatting():
    class A:
        def __repr__(self):
            return "A()"

        def method(self, *args):
            pass

    with ExitStack() as stack:
        stack.callback(A().method)
        stack.push(A().method)
        stack.callback(partial(lambda x: None, 42))
        stack = extract_since(sys._getframe(0))
        assert_stack_matches(
            stack,
            [
                (
                    "test_exitstack_formatting",
                    "with ExitStack() as stack:",
                    "stack",
                    "ExitStack",
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.callback(A().method)",
                    "stack[0]",
                    "function",
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.push(A().method)",
                    "stack[1]",
                    "A",
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.callback(functools.partial(<function test_exitstack_formatting.<locals>.<lambda> at (address)>, 42))",
                    "stack[2]",
                    "function",
                ),
                (
                    "test_exitstack_formatting",
                    "stack = extract_since(sys._getframe(0))",
                    None,
                    None,
                ),
            ],
        )


ACM_IMPLS: List[Callable[..., Any]] = [contextlib.asynccontextmanager]
try:
    import async_generator
except ImportError:
    pass
else:
    ACM_IMPLS.append(async_generator.asynccontextmanager)


@pytest.mark.parametrize("asynccontextmanager", ACM_IMPLS)
def test_asyncexitstack_formatting(asynccontextmanager):
    class A:
        def __repr__(self):
            return "<A>"

        async def __aenter__(self):
            pass

        async def __aexit__(self, *exc):
            pass

        async def aexit(self, *exc):
            pass

    async def aexit2(*exc):
        pass

    async def acallback(*args):
        pass

    @asynccontextmanager
    async def amgr():
        yield

    async def async_fn():
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(A())
            await stack.enter_async_context(amgr())
            stack.push_async_exit(A().aexit)
            stack.push_async_exit(aexit2)
            stack.push_async_callback(acallback, "hi")
            await async_yield(None)

    if asynccontextmanager.__module__.startswith("async_generator"):
        expect_name = "amgr(...)"
    elif sys.version_info >= (3, 9, 7):
        # @asynccontextmanager deletes its func/args/kwds when entering
        # after this version, but didn't before
        expect_name = "test_asyncexitstack_formatting.<locals>.amgr(...)"
    else:
        expect_name = (
            "stackscope._tests.test_traceback.test_asyncexitstack_formatting."
            "<locals>.amgr()"
        )

    coro = async_fn()
    assert coro.send(None) is None
    assert_stack_matches(
        extract(coro),
        [
            (
                "async_fn",
                "async with AsyncExitStack() as stack:",
                "stack",
                "AsyncExitStack",
            ),
            ("async_fn", "# await stack.enter_async_context(<A>)", "stack[0]", "A"),
            (
                "async_fn",
                f"# await stack.enter_async_context({expect_name})",
                "stack[1]",
                "_AsyncGeneratorContextManager",
            ),
            ("amgr", "yield", None, None),
            ("async_fn", "# stack.push_async_exit(<A>.aexit)", "stack[2]", "A"),
            (
                "async_fn",
                "# stack.push_async_exit(stackscope._tests.test_traceback.test_asyncexitstack_formatting.<locals>.aexit2)",
                "stack[3]",
                "function",
            ),
            (
                "async_fn",
                "# stack.push_async_callback(stackscope._tests.test_traceback.test_asyncexitstack_formatting.<locals>.acallback, 'hi')",
                "stack[4]",
                "function",
            ),
            ("async_fn", "await async_yield(None)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )


@pytest.mark.parametrize("asynccontextmanager", ACM_IMPLS)
def test_acm_exiting(asynccontextmanager):
    @asynccontextmanager
    async def amgr():
        try:
            yield
        finally:
            await async_yield(None)

    async def afn():
        async with amgr():
            pass

    if asynccontextmanager.__module__ != "contextlib":
        acm_entry = [
            ("__aexit__", "async with aclosing(self._agen):", None, "aclosing"),
            ("__aexit__", "await self._agen.asend(None)", None, None),
        ]
    elif sys.version_info < (3, 10):
        acm_entry = [("__aexit__", "await self.gen.__anext__()", None, None)]
    else:
        acm_entry = [("__aexit__", "await anext(self.gen)", None, None)]

    coro = afn()
    coro.send(None)
    stack = extract(coro)
    assert_stack_matches(
        stack,
        [
            ("afn", "async with amgr():", None, "_AsyncGeneratorContextManager"),
            *acm_entry,
            ("amgr", "await async_yield(None)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
    with pytest.raises(StopIteration):
        coro.send(None)


def test_unwrap_loops(local_registry: None) -> None:
    @unwrap_stackitem.register
    def dont_unwrap_int(x: int) -> int:
        return x

    result = extract(42)
    assert "42 has been unwrapped more than 100 times" in str(result.error)

    class CM:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *exc: object) -> None:
            pass

        def __repr__(self) -> str:
            return "<CM>"

    @unwrap_context.register(CM)
    def dont_unwrap_cm(cm: CM, context: Any) -> CM:
        return cm

    with CM():
        result = extract_since(sys._getframe(0))
        assert "<CM> has been unwrapped more than 100 times" in str(result.error)

    @customize(elaborate=lambda *_: 42)
    def example():
        with CM():
            result = extract_since(sys._getframe(1))
            assert type(result.error).__name__ == "ExceptionGroup"
            assert (
                "multiple errors encountered while extracting" in result.error.message
            )
            messages = sorted(str(ex) for ex in result.error.exceptions)
            assert len(messages) == 2
            assert "42 has been unwrapped" in messages[0]
            assert "<CM> has been unwrapped" in messages[1]

    example()


def test_multiple_leaves(local_registry: None) -> None:
    @unwrap_stackitem.register
    def unwrap_int(x: int) -> Any:
        if x == 0:
            return ()
        return (x - 1, str(x))

    result = extract(3)
    # [3] -> [2, "3"] -> [1, "2", "3"] -> [0, "1", "2", "3"] -> ["1", "2", "3"]
    assert result.frames == []
    assert result.leaf == ["1", "2", "3"]
    assert result.error is None


def test_format_oddities():
    from stackscope import Frame, Context

    frame = Frame(pyframe=sys._getframe(0), lineno=-100)
    assert frame.linetext == ""

    frame = Frame(pyframe=sys._getframe(0), lineno=0)
    assert frame.linetext == ""
    assert len(frame.format()) == 1

    ctx = Context(obj=None, is_async=False)
    assert ctx.format() == ["with <???>:\n"]

    ctx = Context(obj=None, is_async=True, start_line=-100)
    assert ctx.format() == ["async with <???>:  # (line -100)\n"]

    ctx = Context(obj=42, is_async=False)
    assert ctx.format() == ["with <???>:  # _: int\n"]

    ctx = Context(obj=None, varname="foo", is_async=False)
    assert ctx.format() == ["with <???>:  # foo\n"]

    ctx.children = [Stack(root=None, frames=[])]
    assert ctx.format(ascii_only=True)[1] == ". <unidentified child>\n"

    def no_locals():
        yield

    gen = no_locals()
    gen.send(None)
    assert extract(gen).frames[0].clsname is None

    def not_a_method(cls, arg):
        del cls
        yield

    gen = not_a_method(42, 10)
    gen.send(None)
    assert extract(gen).frames[0].clsname is None

    ns = {}
    exec("def no_name_in_globals(): yield", ns, ns)
    no_name_in_globals = ns["no_name_in_globals"]
    gen = no_name_in_globals()
    gen.send(None)
    assert extract(gen).frames[0].modname is None


def test_extract_outermost(local_registry: None) -> None:
    @unwrap_stackitem.register(int)
    def unwrap(value):
        raise RuntimeError("example unwrap error")

    async def outer():
        await async_yield(None)

    coro = outer()
    coro.send(None)
    frame = extract_outermost(coro)
    coro.close()
    lines = frame.format(ascii_only=True)
    assert lines[0].startswith("outer in stackscope._tests.test_traceback at ")
    assert lines[1] == "` await async_yield(None)\n"

    with pytest.raises(RuntimeError, match="example unwrap error"):
        extract_outermost(42)

    with pytest.raises(RuntimeError, match="unwrapping only reached 'hello'"):
        extract_outermost("hello")


def test_extract_child_invalid() -> None:
    with pytest.raises(RuntimeError, match="may only be called from within"):
        extract_child(None, for_task=False)


def test_unwrap_gcm(local_registry: None) -> None:
    cm = null_context()
    context = Context(obj=cm, is_async=False)
    with cm:
        pass

    # Smoke test of base case where no registered unwrapper matches
    unwrap_context(null_context(), context)

    @unwrap_context_generator.register(null_context)
    def unwrap_nullcontext(frame: Frame, context: Any) -> Any:
        return 42

    # Exhausted generator has no frame, thus can't find the unwrapper
    fill_context(context)
    assert context.obj is cm
    assert context.inner_stack is not None and not context.inner_stack.frames

    # Try the branch with no inner_stack
    context = Context(obj=cm, is_async=False, is_exiting=True)
    fill_context(context)
    assert context.obj is cm
    assert context.inner_stack is None

    # Now try both of these with a GCM that does have an associated frame.
    # These should find the above unwrapper successfully
    cm = null_context()
    context = Context(obj=cm, is_async=False)
    fill_context(context)
    assert context.obj == 42
    assert context.inner_stack is None

    context = Context(obj=cm, is_async=False, is_exiting=True)
    fill_context(context)
    assert context.obj == 42
    assert context.inner_stack is None
