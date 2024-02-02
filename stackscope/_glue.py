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
    AsyncContextManager,
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

from ._types import Stack, Frame, Context, StackSlice
from ._code_dispatch import get_code
from ._customization import (
    unwrap_stackitem,
    elaborate_frame,
    customize,
    unwrap_context,
    unwrap_context_generator,
    elaborate_context,
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
    """Returns a decorator which marks the function it decorates as
    providing glue for *needs_module*. The decorated function will be
    called at most once, on the first attempt to extract a stack after
    *needs_module* has been loaded, in order to set up customizations
    that depend on that module.

    Modules may also provide their own glue, by defining a top-level
    function named ``_stackscope_install_glue_``. This function will
    similarly be called at most once, on the first attempt to extract
    a stack after the module defining it has been loaded. If both
    module-provided and builtin glue exist for the same module, then
    only the module-provided glue will run.
    """

    def decorate(fn: InstallGlueFn) -> InstallGlueFn:
        assert needs_module not in builtin_glue_pending
        if needs_module in sys.modules and "sphinx" not in sys.modules:
            fn()
        else:
            builtin_glue_pending[needs_module] = fn
        return fn

    return decorate


glue_lock = threading.Lock()


def add_glue_as_needed(*, _sys_modules_len_cache: list[int] = [0]) -> None:
    if len(sys.modules) == _sys_modules_len_cache[0]:
        return
    # Use a lock to avoid races between multiple threads trying to extract
    # tracebacks simultaneously
    with glue_lock:
        module_names = tuple(sys.modules)
        for module_name in module_names:
            builtin_fn = builtin_glue_pending.pop(module_name, None)
            try:
                module_fn = sys.modules[module_name].__dict__.pop(
                    "_stackscope_install_glue_", None
                )
            except Exception:  # module disappeared, doesn't have a dict, etc
                module_fn = None
            try:
                # Prefer the module-supplied glue over our builtin version
                # in case both are present
                if module_fn is not None:
                    module_fn()
                elif builtin_fn is not None:
                    builtin_fn()
            except Exception as exc:
                kind = (
                    "module-provided" if module_fn is not None else "stackscope-builtin"
                )
                exc_str = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                warnings.warn(
                    f"Failed to initialize {kind} glue for {module_name}: {exc_str}. "
                    "Some tracebacks may be presented less crisply or with "
                    "missing information.",
                    RuntimeWarning,
                )
        # Only update the length cache if we visited every module (rather
        # than bailing out with an exception)
        _sys_modules_len_cache[0] = len(module_names)


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
    agen = some_asyncgen()
    asend_type = type(agen.asend(None))
    athrow_type = type(agen.athrow(ValueError))
    try:
        # Clean up the asyncgen so it doesn't confuse any finalization hooks
        agen.aclose().send(None)  # type: ignore
    except (StopIteration, StopAsyncIteration):
        pass

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
            context.inner_stack = _extract.extract_child(mgr.gen, for_task=False)
        if hasattr(mgr, "func"):
            context.description = format_funcall(mgr.func, mgr.args, mgr.kwds)
        else:
            # 3.7+ delete the func/args/etc attrs once entered
            context.description = f"{mgr.gen.__qualname__}(...)"

    @unwrap_context.register(GCMBase)
    def unwrap_generatorbased_contextmanager(mgr: Any, context: Context) -> Any:
        mgr_code: types.CodeType
        if hasattr(mgr.gen, "gi_code"):
            mgr_code = mgr.gen.gi_code
        elif hasattr(mgr.gen, "ag_code"):
            mgr_code = mgr.gen.ag_code
        else:  # pragma: no cover
            return None
        if mgr_code in unwrap_context_generator.registry:
            if context.inner_stack is not None:
                if context.inner_stack.frames:
                    return unwrap_context_generator(
                        context.inner_stack.frames[0], context
                    )
            else:
                try:
                    frame = _extract.extract_outermost(mgr.gen)
                except RuntimeError:  # no frames
                    pass
                else:
                    return unwrap_context_generator(frame, context)
        return None

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
            _extract.fill_context(child_context)
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
            context.inner_stack = _extract.extract_child(mgr._agen, for_task=False)
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


@builtin_glue("contextvars")
def glue_contextvars() -> None:
    import contextvars

    # Don't show Context.run() in tracebacks even if it's implemented in
    # Python (which is only true these days on pypy). On CPython it's
    # implemented in C and will be hidden with or without this logic.
    if isinstance(contextvars.Context.run, types.FunctionType):
        customize(contextvars.Context.run, hide=True)


@builtin_glue("trio")
def glue_trio() -> None:
    import threading
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

    @unwrap_stackitem.register(lowlevel.Task)
    def unwrap_task(task: lowlevel.Task) -> Any:
        return task.coro

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
        assert isinstance(context.obj, trio.Nursery)
        context.children = [
            _extract.extract_child(child_task, for_task=True)
            for child_task in context.obj.child_tasks
        ]

    @elaborate_frame.register(trio.to_thread.run_sync)
    def elaborate_to_thread_run_sync(frame: Frame, next_inner: object) -> object:
        thread_name = frame.pyframe.f_locals.get("thread_name")
        worker_fn = frame.pyframe.f_locals.get("worker_fn")
        sync_fn = frame.pyframe.f_locals.get("sync_fn")
        if not (thread_name and worker_fn and sync_fn):  # pragma: no cover
            # We're in the initial setup-y part, thread not running yet
            return None

        # Find the thread that's hosting the sync_fn
        for thread in threading.enumerate():
            if thread.name is thread_name:
                break
        else:  # pragma: no cover
            # Thread isn't running yet
            return None

        # Find the actual frames where the sync_fn and its callees are running
        inner_frame = sys._current_frames().get(thread.ident or 0)
        previous: types.FrameType | None = None
        current = inner_frame
        while current is not None and current.f_code is not worker_fn.__code__:
            previous = current
            current = current.f_back
        if current is None or previous is None:  # pragma: no cover
            # We either didn't find the worker_fn, or it didn't have a callee.
            # Thread isn't doing anything interesting yet.
            return None

        frame.hide = True
        thread_stack = StackSlice(inner=inner_frame, outer=previous)
        if (
            isinstance(next_inner, Frame)
            and next_inner.funcname == "wait_task_rescheduled"
        ):
            # Waiting for the thread to do something
            return thread_stack
        # Otherwise, the to_thread.run_sync task is most likely executing a
        # reentrant from_thread.run_* call made by the sync_fn. Include the
        # thread's stack but then switch back to the Trio task.
        return (thread_stack, next_inner)

    @elaborate_frame.register(trio.from_thread.run)
    def elaborate_from_thread_run(frame: Frame, next_inner: object) -> object:
        missing = object()
        token_provided = frame.pyframe.f_locals.get("token_provided")
        trio_token = frame.pyframe.f_locals.get("trio_token", missing)
        if trio_token is missing:  # pragma: no cover
            return None
        if token_provided is None:  # pragma: no cover
            # Trio v0.23.1 had token_provided; v0.24+ don't
            token_provided = trio_token is not None
        if not token_provided:
            # No trio_token specified, so this is a reentrant call
            # back into Trio from a to_thread.run_sync() function.
            # The in-Trio portion will show up in the stack of
            # to_thread.run_sync() so don't duplicate it here.
            # Prune the plumbing (_send_message_to_trio, Queue.get(), etc)
            # that's inward of here on the thread's stack.
            frame.hide = True
            return ()

        # If a trio_token was specified, then this is a call that did not
        # ultimately originate in Trio, so we should try to follow the link
        # back to the Trio task if possible.
        message = frame.pyframe.f_locals.get("message_to_trio")
        if (
            message is None
            and isinstance(next_inner, Frame)
            and next_inner.funcname == "_send_message_to_trio"
        ):  # pragma: no cover  # depends on Trio minor version
            message = next_inner.pyframe.f_locals.get("message_to_trio")
        if message is not None:  # pragma: no branch
            # Find the Trio runner that this token refers to
            runner = None
            try:
                runner_tlocals = trio._core._run.GLOBAL_RUN_CONTEXT  # type: ignore
            except AttributeError:  # pragma: no cover
                return None
            for ref in gc.get_referents(runner_tlocals):
                if not isinstance(ref, dict):
                    continue
                for key, value in ref.items():
                    if key == "runner" and value.trio_token is trio_token:
                        # PyPy style: each thread's thread-locals dict is
                        # directly referenced by the threading.local instance
                        runner = value
                        break
                    if isinstance(value, dict) and (  # pragma: no branch
                        runner := value.get("runner")
                    ):
                        # CPython style: each thread's thread-locals dict is
                        # a value in one big dict keyed by weakrefs
                        if runner.trio_token is trio_token:  # pragma: no branch
                            break
                else:  # pragma: no cover
                    continue
                break
            else:  # pragma: no cover
                return None

            # Find the system task that matches this call
            for task in runner.system_nursery.child_tasks:  # pragma: no branch
                if task.context is message.context:  # pragma: no branch
                    frame.hide = True
                    return task.coro

        return None  # pragma: no cover

    if trio_threads := getattr(trio, "_threads", None):  # pragma: no branch
        for clsname, fnname in (
            ("Run", "run"),
            ("Run", "run_system"),
            ("Run", "unprotected_afn"),
            ("RunSync", "run_sync"),
            ("RunSync", "unprotected_fn"),
        ):
            cls = getattr(trio_threads, clsname, None)
            fn = getattr(cls, fnname, None)
            if fn is not None:  # pragma: no branch
                customize(fn, hide=True)


@builtin_glue("pytest_trio.plugin")
def glue_pytest_trio() -> None:
    import pytest_trio.plugin  # type: ignore
    import trio

    @elaborate_frame.register(pytest_trio.plugin.TrioFixture.run)
    def elaborate_fixture(frame: Frame, next_inner: object) -> object:
        # When a pytest-trio fixture is implemented using an async generator,
        # any nurseries the fixture left open will be on the async generator's
        # stack, which is not directly on the fixture task's stack. Detect
        # this situation and fix it up.
        if not isinstance(next_inner, Frame):  # pragma: no cover
            return None
        # Do we have the fixture generator yet? (Might not if we're still waiting
        # for a depended-upon fixture to finish.)
        gen = frame.pyframe.f_locals.get("func_value")
        if gen is None:
            return None
        # Is it actually a generator? (Might not if this is a fixture that
        # does some async setup and just returns.)
        if not (hasattr(gen, "asend") or hasattr(gen, "__next__")):
            return None
        # Are we waiting for the test to finish? (Might not if we're still
        # running the setup phase of the fixture.)
        if next_inner.clsname == "Event" and next_inner.funcname == "wait":
            # Yes -- show the fixture generator's stack in place of Event.wait().
            frame.hide_line = True
            return gen
        return None

    @unwrap_context_generator.register(pytest_trio.plugin.TrioFixture._fixture_manager)
    def unwrap_fixture_manager(
        frame: Frame, context: Context
    ) -> None | AsyncContextManager[Any] | tuple[()]:
        # Replace the internal _fixture_manager context (which is itself
        # uninteresting) with the nursery that it contains (which might
        # be hosting relevant portions of the test logic that we don't
        # wish to hide). If that nursery has nothing in it, then hide it.
        if frame.contexts and isinstance(
            nursery := frame.contexts[0].obj, trio.Nursery
        ):
            if nursery.child_tasks:
                return cast(AsyncContextManager[Any], nursery)
            else:
                return ()
        else:  # pragma: no cover
            # There are no checkpoints in _fixture_manager() except
            # Nursery.__aexit__, so it is not generally possible to observe
            # it without the nursery on its context stack.
            return None


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
                    "Can't dump the stack of a greenlet running in another thread"
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
    def unwrap_greenback_async_context(manager: Any, context: Context) -> Any:
        return manager._cm
