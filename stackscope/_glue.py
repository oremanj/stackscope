from __future__ import annotations

# Specific customizations (for the stdlib and other libraries of interest
# to the author) that use the customization interfaces in ._customization.

import functools
import gc
import sys
import threading
import traceback
import types
import warnings
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    cast,
)

from ._types import Frame, Context, StackSlice
from ._code_dispatch import get_code
from ._customization import (
    unwrap_stackitem,
    elaborate_frame,
    customize,
    unwrap_context,
    elaborate_context,
    fill_context,
    yields_frames,
)
from . import _extract

try:
    if not TYPE_CHECKING:
        from greenlet import (
            greenlet as GreenletType,
            getcurrent as greenlet_getcurrent,
        )
except ImportError:

    class GreenletType:
        parent: Optional["GreenletType"] = None
        gr_frame: Optional[types.FrameType]

    def greenlet_getcurrent() -> GreenletType:
        return GreenletType()


InstallGlueFn = Callable[[], None]
builtin_glue_pending: Dict[str, InstallGlueFn] = {}


def builtin_glue(needs_module: str) -> Callable[[InstallGlueFn], InstallGlueFn]:
    def decorate(fn: InstallGlueFn) -> InstallGlueFn:
        if needs_module in sys.modules and "sphinx" not in sys.modules:
            fn()
        else:
            assert needs_module not in builtin_glue_pending
            builtin_glue_pending[needs_module] = fn
        return fn

    return decorate


def add_glue_as_needed() -> None:
    for module_name in list(builtin_glue_pending):
        if module_name in sys.modules:
            try:
                install_fn = builtin_glue_pending.pop(module_name)
            except KeyError:
                # Probably two simultaneous calls occurring in
                # different threads; the effect of glue is global
                # so just skip it under the assumption that
                # someone else got this one.
                pass
            else:
                try:
                    install_fn()
                except Exception as exc:
                    warnings.warn(
                        "Failed to initialize glue for {}: {}. Some tracebacks may be "
                        "presented less crisply or with missing information.".format(
                            module_name,
                            "".join(
                                traceback.format_exception_only(type(exc), exc)
                            ).strip(),
                        ),
                        RuntimeWarning,
                    )


functools_singledispatch_wrapper = get_code(functools.singledispatch, "wrapper")


def get_true_caller() -> types.FrameType:
    # Return the frame that called into the traceback-producing machinery
    caller: Optional[types.FrameType] = sys._getframe(1)

    def is_mine(name: str) -> bool:
        return name.startswith("stackscope.") and not name.startswith(
            "stackscope._tests."
        )

    while caller is not None and (
        is_mine(caller.f_globals.get("__name__", ""))
        or caller.f_code is functools_singledispatch_wrapper
    ):
        caller = caller.f_back
    assert caller is not None
    return caller


@yields_frames
def unwrap_stackslice(spec: StackSlice) -> Iterator[types.FrameType]:
    outer_frame = spec.outer
    inner_frame = spec.inner

    # Running frames are linked "outward" (from the current frame
    # backwards via f_back members), not "inward" (even if the frames
    # belong to a coroutine/generator/etc, the cr_await / gi_yieldfrom
    # / etc members are None). Thus, in order to know what comes after
    # outer_frame in the traceback, we have to find a currently
    # executing frame that has outer_frame on its stack.

    # This function tries a bunch of different ways to fill out this 'frames'
    # list, as a list of the frames named by the StackSlice, from outermost
    # to innermost.
    frames: List[types.FrameType] = []

    if sys.implementation.name == "cpython" and greenlet_getcurrent().parent:
        # On CPython, each greenlet is its own universe traceback-wise:
        # if you're inside a non-main greenlet and you follow f_back links
        # outward from your current frame, you'll only find the outermost
        # frame in this greenlet, not in this thread. We augment that by
        # following the greenlet parent link (the same path an exception
        # would take) when we reach an f_back of None.
        #
        # This only works on the current thread, since there's no way
        # to call greenlet.getcurrent() for another thread. (There's a key
        # in the thread state dictionary, which we can't safely access because
        # any other thread state can disappear out from under us whenever we
        # yield the GIL, which we can't prevent from happening. Resolving
        # this would require an extension module.)
        #
        # PyPy uses a more sensible scheme where the f_back links in the
        # current callstack always work, so it doesn't need this trick.
        this_thread_frames: List[types.FrameType] = []
        greenlet: Optional[GreenletType] = greenlet_getcurrent()
        current: Optional[types.FrameType] = get_true_caller()
        while greenlet is not None:
            while current is not None:
                this_thread_frames.append(current)
                current = current.f_back
            greenlet = greenlet.parent
            if greenlet is not None:
                current = greenlet.gr_frame
        try:
            from_idx: Optional[int]
            if inner_frame is None or this_thread_frames[0] is inner_frame:
                from_idx = None
            else:
                from_idx = this_thread_frames.index(inner_frame) - 1
            to_idx = (
                len(this_thread_frames)
                if outer_frame is None
                else this_thread_frames.index(outer_frame)
            )
        except ValueError:
            pass
        else:
            frames = this_thread_frames[to_idx:from_idx:-1]

    def try_from(potential_inner_frame: types.FrameType) -> List[types.FrameType]:
        """If potential_inner_frame has outer_frame somewhere up its callstack,
        return the list of frames between them, starting with
        outer_frame and ending with potential_inner_frame. Otherwise, return [].
        If outer_frame is None, return all frames enclosing
        potential_inner_frame, including potential_inner_frame.
        """
        frames: List[types.FrameType] = []
        current: Optional[types.FrameType] = potential_inner_frame
        while current is not None:
            frames.append(current)
            # Note: suspended greenlets on PyPy have frames that form a cycle,
            # thus the 2nd part of this condition
            if current is outer_frame or (
                outer_frame is None and current.f_back is potential_inner_frame
            ):
                break
            current = current.f_back
        if current is outer_frame or outer_frame is None:
            return frames[::-1]
        return []

    if not frames:
        frames = try_from(inner_frame or get_true_caller())

    if not frames and inner_frame is None:
        # outer_frame isn't on *our* stack, but it might be on some
        # other thread's stack. It would be nice to avoid yielding
        # the GIL while we look, but that doesn't appear to be
        # possible -- sys.setswitchinterval() changes apply only
        # to those threads that have yielded the GIL since it was
        # called. We'll just have to accept the potential race
        # condition.
        for ident, inner_frame in sys._current_frames().items():
            if ident != threading.get_ident():
                frames = try_from(inner_frame)
                if frames:
                    break

    if not frames:
        # If outer_frame is None, try_from() always returns a non-empty
        # list, so we can only get here if outer_frame is not None.
        assert outer_frame is not None
        yield outer_frame
        raise RuntimeError(
            "Couldn't find where the above frame is running, so "
            "can't continue traceback"
        )

    if spec.limit is not None and len(frames) > spec.limit:
        if inner_frame is None and outer_frame is not None:
            del frames[spec.limit :]
        else:
            del frames[: -spec.limit]

    yield from frames


@builtin_glue("builtins")
def glue_builtins() -> None:
    unwrap_stackitem.register(unwrap_stackslice)

    @unwrap_stackitem.register(types.GeneratorType)
    def unwrap_geniter(gen: types.GeneratorType[Any, Any, Any]) -> Any:
        if gen.gi_running:
            return StackSlice(outer=gen.gi_frame)
        return (gen.gi_frame, gen.gi_yieldfrom)

    @unwrap_stackitem.register(types.CoroutineType)
    def unwrap_coro(coro: types.CoroutineType[Any, Any, Any]) -> Any:
        if coro.cr_running:
            return StackSlice(outer=coro.cr_frame)
        return (coro.cr_frame, coro.cr_await)

    @unwrap_stackitem.register(types.AsyncGeneratorType)
    def unwrap_asyncgen(agen: types.AsyncGeneratorType[Any, Any]) -> Any:
        # On 3.8+, ag_running is true even if the generator
        # is blocked on an event loop trap. Those will
        # have ag_await != None (because the only way to
        # block on an event loop trap is to await something),
        # and we want to treat them as suspended for
        # traceback extraction purposes.
        if agen.ag_running and agen.ag_await is None:
            return StackSlice(outer=agen.ag_frame)
        return (agen.ag_frame, agen.ag_await)

    async def some_asyncgen() -> AsyncGenerator[None, None]:
        yield  # pragma: no cover

    # Get the types of the internal awaitables used in native async
    # generator asend/athrow calls
    asend_type = type(some_asyncgen().asend(None))
    athrow_type = type(some_asyncgen().athrow(ValueError))

    async def some_afn() -> None:
        pass  # pragma: no cover

    # Get the coroutine_wrapper type returned by <coroutine object>.__await__()
    coro = some_afn()
    coro_wrapper_type = type(coro.__await__())
    coro.close()

    @unwrap_stackitem.register(asend_type)
    @unwrap_stackitem.register(athrow_type)
    def unwrap_async_generator_asend_athrow(aw: Any) -> Any:
        # native async generator awaitable, which holds a
        # reference to its agen but doesn't expose it
        for referent in gc.get_referents(aw):
            if hasattr(referent, "ag_frame"):  # pragma: no branch
                return referent
        raise RuntimeError(
            f"{aw!r} doesn't refer to anything with an ag_frame attribute"
        )

    @unwrap_stackitem.register(coro_wrapper_type)
    def unwrap_coroutine_wrapper(aw: Any) -> Any:
        # these refer to only one other object, the underlying coroutine
        for referent in gc.get_referents(aw):
            if hasattr(referent, "cr_frame"):  # pragma: no branch
                return referent
        raise RuntimeError(
            f"{aw!r} doesn't refer to anything with a cr_frame attribute"
        )


def format_funcname(func: object) -> str:
    try:
        if isinstance(func, types.MethodType):
            return f"{func.__self__!r}.{func.__name__}"
        else:
            return f"{func.__module__}.{func.__qualname__}"  # type: ignore
    except AttributeError:
        return repr(func)


def format_funcargs(args: Sequence[Any], kw: Mapping[str, Any]) -> List[str]:
    argdescs = [repr(arg) for arg in args]
    kwdescs = [f"{k}={v!r}" for k, v in kw.items()]
    return argdescs + kwdescs


def format_funcall(func: object, args: Sequence[Any], kw: Mapping[str, Any]) -> str:
    return "{}({})".format(format_funcname(func), ", ".join(format_funcargs(args, kw)))


@builtin_glue("contextlib")
def glue_contextlib() -> None:
    import contextlib

    # GCMBase: the common base type of context manager objects returned by
    # functions decorated with either @contextlib.contextmanager or
    # @contextlib.asynccontextmanager
    GCMBase = cast(Any, contextlib)._GeneratorContextManagerBase
    ExitStackBase = cast(Any, contextlib)._BaseExitStack

    @elaborate_context.register(GCMBase)
    def elaborate_generatorbased_contextmanager(mgr: Any, context: Context) -> None:
        # Don't descend into @contextmanager frames if the context manager
        # is currently exiting, since we'll see them later in the traceback
        # anyway
        if not context.is_exiting:
            context.inner_stack = _extract.extract(mgr.gen)
        if hasattr(mgr, "func"):
            context.description = format_funcall(mgr.func, mgr.args, mgr.kwds)
        else:
            # 3.7+ delete the func/args/etc attrs once entered
            context.description = f"{mgr.gen.__qualname__}(...)"

    @elaborate_context.register(ExitStackBase)
    def elaborate_exit_stack(stack: Any, context: Context) -> None:
        stackname = context.varname or "_"
        children = []
        # List of (is_sync, callback) tuples, from outermost to innermost, where
        # each callback takes parameters following the signature of a __exit__ method
        callbacks: List[Tuple[bool, Callable[..., Any]]] = list(stack._exit_callbacks)

        for idx, (is_sync, callback) in enumerate(callbacks):
            tag = ""
            manager: object = None
            method: str
            arg: Optional[str] = None
            if hasattr(callback, "__self__"):
                manager = callback.__self__
                if (
                    # 3.7 used a wrapper function with a __self__ attribute
                    # for actual __exit__ invocations. Later versions use a method.
                    not isinstance(callback, types.MethodType)
                    or callback.__func__.__name__ in ("__exit__", "__aexit__")
                ):
                    # stack.enter_context(some_cm) or stack.push(some_cm)
                    tag = "" if is_sync else "await "
                    method = "enter_context" if is_sync else "enter_async_context"
                    arg = repr(manager)
                else:
                    # stack.push(something.exit_ish_method)
                    method = "push" if is_sync else "push_async_exit"
                    arg = format_funcname(callback)
            elif (
                hasattr(callback, "__wrapped__")
                and getattr(callback, "__name__", None) == "_exit_wrapper"
                and isinstance(callback, types.FunctionType)
                and set(callback.__code__.co_freevars) >= {"args", "kwds"}
            ):
                # Normal callback wrapped in internal _exit_wrapper function
                # to adapt it to the __exit__ protocol
                args_idx = callback.__code__.co_freevars.index("args")
                kwds_idx = callback.__code__.co_freevars.index("kwds")
                assert callback.__closure__ is not None
                arg = ", ".join(
                    [
                        format_funcname(callback.__wrapped__),  # type: ignore
                        *format_funcargs(
                            callback.__closure__[args_idx].cell_contents,
                            callback.__closure__[kwds_idx].cell_contents,
                        ),
                    ],
                )
                method = "callback" if is_sync else "push_async_callback"
            else:
                # stack.push(exit_ish_function)
                method = "push" if is_sync else "push_async_exit"
                arg = format_funcname(callback)

            child_context = Context(
                obj=manager or callback,
                is_async=not is_sync,
                varname=f"{stackname}[{idx}]",
                start_line=context.start_line,
            )
            fill_context(child_context)
            child_context.description = f"{tag}{stackname}.{method}({child_context.description or arg or '...'})"
            children.append(child_context)

        context.children = children


@builtin_glue("threading")
def glue_threading() -> None:
    import threading

    @unwrap_stackitem.register(threading.Thread)
    def unwrap_thread(thread: threading.Thread) -> Any:
        # If the thread is not alive both before and after we try to fetch
        # its frame, then it's possible that its identity was reused, and
        # we shouldn't trust the frame we get.
        was_alive = thread.is_alive()
        inner_frame = sys._current_frames().get(thread.ident)  # type: ignore
        if inner_frame is None or not thread.is_alive() or not was_alive:
            return []
        return StackSlice(inner=inner_frame)

    # Don't show thread bootstrap gunk at the base of thread stacks
    customize(threading.Thread.run, hide=True)
    for name in ("_bootstrap", "_bootstrap_inner"):
        if hasattr(threading.Thread, name):  # pragma: no branch
            customize(getattr(threading.Thread, name), hide=True)


@builtin_glue("async_generator")
def glue_async_generator() -> None:
    import async_generator

    customize(async_generator.yield_, hide=True, prune=True)

    @async_generator.async_generator
    async def noop_agen() -> None:
        pass  # pragma: no cover

    # Skip internal frames corresponding to asend() and athrow()
    # coroutines for the @async_generator backport. The native
    # versions are written in C, so won't show up regardless.
    agen_iter = noop_agen()
    asend_coro = cast(Coroutine[Any, Any, Any], agen_iter.asend(None))
    customize(asend_coro.cr_code, hide=True)

    @unwrap_stackitem.register(type(agen_iter))
    def unwrap_async_generator_backport(agen: Any) -> Any:
        return agen._coroutine

    from async_generator._impl import ANextIter  # type: ignore

    @unwrap_stackitem.register(ANextIter)
    def unwrap_async_generator_backport_next_iter(aw: Any) -> Any:
        return aw._it

    asend_coro.close()

    from async_generator._util import _AsyncGeneratorContextManager as AGCMBackport  # type: ignore

    @elaborate_context.register(AGCMBackport)
    def elaborate_generatorbased_contextmanager(
        mgr: AGCMBackport, context: Context
    ) -> None:
        # Don't descend into @contextmanager frames if the context manager
        # is currently exiting, since we'll see them later in the traceback
        # anyway
        if not context.is_exiting:
            context.inner_stack = _extract.extract(mgr._agen)
        context.description = f"{mgr._func_name}(...)"


@builtin_glue("outcome")
def glue_outcome() -> None:
    import outcome

    # Don't give simple outcome functions their own traceback frame,
    # as they tend to add clutter without adding meaning. If you see
    # 'something.send(coro)' in one frame and you're inside coro in
    # the next, it's pretty obvious what's going on.
    customize(outcome.Value.send, hide=True)
    customize(outcome.Error.send, hide=True)
    customize(outcome.capture, hide=True)
    customize(outcome.acapture, hide=True)


@builtin_glue("trio")
def glue_trio() -> None:
    import trio

    try:
        lowlevel = trio.lowlevel
    except ImportError:  # pragma: no cover
        # Support older Trio versions
        lowlevel = trio.hazmat  # type: ignore

    # Skip frames corresponding to common lowest-level Trio traps,
    # so that the traceback ends where someone says 'await
    # wait_task_rescheduled(...)' or similar.  The guts are not
    # interesting and they otherwise show up *everywhere*.
    for trap in (
        "cancel_shielded_checkpoint",
        "wait_task_rescheduled",
        "temporarily_detach_coroutine_object",
        "permanently_detach_coroutine_object",
    ):
        if hasattr(lowlevel, trap):  # pragma: no branch
            customize(getattr(lowlevel, trap), hide=True, prune=True)

    async def get_nursery_type() -> Type[Any]:
        return type(trio.open_nursery())

    try:
        trio.current_time()
    except RuntimeError:
        nursery_manager_type = trio.run(get_nursery_type)
    else:
        nursery_manager_type = type(trio.open_nursery())

    @elaborate_context.register(nursery_manager_type)
    def elaborate_nursery(manager: Any, context: Context) -> None:
        context.obj = manager._nursery


@builtin_glue("greenlet")
def glue_greenlet() -> None:
    import greenlet  # type: ignore

    @unwrap_stackitem.register(GreenletType)
    def unwrap_greenlet(glet: GreenletType) -> Any:
        inner_frame = glet.gr_frame
        outer_frame = None
        if inner_frame is None:
            if not glet:  # dead or not started
                return []
            # otherwise a None frame means it's running
            if glet is not greenlet_getcurrent():
                raise RuntimeError(
                    "Can't dump the stack of a greenlet running " "in another thread"
                )
            # since it's running in this thread, its stack is our own
            inner_frame = get_true_caller()
            if glet.parent is not None:
                outer_frame = inner_frame
                assert outer_frame is not None

                # On CPython the end of this greenlet's stack is marked
                # by None. On PyPy it gets seamlessly attached to its
                # parent's stack.
                while (
                    outer_frame.f_back is not glet.parent.gr_frame
                    and outer_frame.f_back is not None
                ):
                    outer_frame = outer_frame.f_back
        return StackSlice(outer=outer_frame, inner=inner_frame)

    if sys.implementation.name != "pypy":
        return

    # pypy greenlet is written in Python on top of the pypy-specific
    # module _continuation.  Hide traceback frames for its internals
    # for better consistency with CPython.
    customize(greenlet.greenlet.switch, hide=True)
    customize(greenlet.greenlet._greenlet__switch, hide=True)
    customize(greenlet._greenlet_start, hide=True)
    customize(greenlet._greenlet_throw, hide=True)


@builtin_glue("greenback")
def glue_greenback() -> None:
    import greenback

    @elaborate_frame.register(greenback._impl._greenback_shim)
    def elaborate_greenback_shim(frame: Frame, next_inner: object) -> object:
        frame.hide = True

        if isinstance(next_inner, Frame):
            # Greenback shim that's not suspended at its yield point requires
            # no special handling -- just keep tracebacking.
            return None

        # Greenback shim. Is the child coroutine suspended in an await_()?
        child_greenlet = frame.pyframe.f_locals.get("child_greenlet")
        orig_coro = frame.pyframe.f_locals.get("orig_coro")
        gr_frame = getattr(child_greenlet, "gr_frame", None)
        if gr_frame is not None:
            # Yep; switch to walking the greenlet stack, since orig_coro
            # will look "running" but it's not on any thread's stack.
            return child_greenlet
        elif orig_coro is not None:
            # No greenlet, so child is suspended at a regular await.
            # Continue the traceback by walking the coroutine's frames.
            return orig_coro
        else:  # pragma: no cover
            raise RuntimeError(
                "Can't identify what's going on with the greenback shim in this "
                "frame"
            )

    @elaborate_frame.register(greenback.await_)
    def elaborate_greenback_await(frame: Frame, next_inner: object) -> object:
        frame.hide = True

        if (
            isinstance(next_inner, Frame)
            and next_inner.pyframe.f_code.co_name != "switch"
        ):
            # await_ that's not suspended at greenlet.switch() requires
            # no special handling
            return None

        # Greenback-mediated await of async function from sync land.
        # If we have a coroutine to descend into, do so;
        # otherwise the traceback will unhelpfully stop here.
        # This works whether the coro is running or not.
        # (The only way to get coro=None is if we're taking
        # the traceback in the early part of await_() before
        # coro is assigned.)
        return frame.pyframe.f_locals.get("coro")

    @unwrap_context.register(greenback.async_context)
    def unwrap_greenback_async_context(manager: Any) -> Any:
        return manager._cm
