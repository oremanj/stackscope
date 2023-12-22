from __future__ import annotations

# Basic logic for extracting stacks

import collections.abc
import functools
import gc
import inspect
import itertools
import sys
import threading
import types
import weakref
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Deque,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
    cast,
)

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from ._types import Stack, Frame, Context, StackSlice, StackItem
from ._code_dispatch import get_code
from ._customization import (
    unwrap_stackitem,
    elaborate_frame,
    unwrap_context,
    elaborate_context,
    FrameIterator,
    PRUNE,
)
from ._lowlevel import contexts_active_in_frame
from . import _glue


__all__ = [
    "extract",
    "extract_outermost",
    "extract_since",
    "extract_until",
    "extract_child",
    "fill_context",
]


def better_origin(candidate: object, fallback: object) -> object:
    """Returns *obj* if it is weak-referenceable and a better Frame.origin than
    *fallback*, else *fallback*."""
    try:
        weakref.ref(candidate)
    except TypeError:
        return fallback
    else:
        typelist = (types.CoroutineType, types.GeneratorType, types.AsyncGeneratorType)
        if isinstance(candidate, typelist) or not isinstance(fallback, typelist):
            return candidate
        return fallback


class ExtractOptions(threading.local):
    with_contexts: bool = cast(bool, None)
    recurse_child_tasks: bool = cast(bool, None)

    @contextmanager
    def push(self, *, with_contexts: bool, recurse_child_tasks: bool) -> Iterator[None]:
        prev = (self.with_contexts, self.recurse_child_tasks)
        self.with_contexts = with_contexts
        self.recurse_child_tasks = recurse_child_tasks
        try:
            yield
        finally:
            (self.with_contexts, self.recurse_child_tasks) = prev


current_options = ExtractOptions()


def extract_iter(
    stackitem: StackItem, save_errors: List[Exception]
) -> Generator[Frame, None, StackItem]:
    """Implements the common logic of extract() and extract_outermost().

    Yields a series of Frames, then returns the leaf object (if any) that
    remains after removing all the frames. Any exceptions encountered during
    traversal are saved to the list *save_errors*.
    """

    assert current_options.with_contexts is not None
    _glue.add_glue_as_needed()

    # to_unwrap items are (origin, stackitem, depth)
    to_unwrap: Deque[Tuple[Optional[StackItem], StackItem, int]] = collections.deque()
    # to_elaborate items are (frame or leaf, depth)
    to_elaborate: Deque[Tuple[Union[Frame, StackItem], int]] = collections.deque()
    # In both of these cases, depth is the number of layers of unwrapping
    # that were done to reach the item. It's used to ensure that
    # elaborate_frame() can only prune things that are logically 'inward'
    # of the frame it's looking at.

    to_unwrap.append((better_origin(stackitem, None), stackitem, 0))
    while to_unwrap or to_elaborate:
        # Unwrap until we have a frame and the thing that comes after it
        # (which might be 'nothing', if it's the innermost frame).
        # If our unwrapping didn't produce a frame, then unwrap everything
        # that remains so we can fill out .leaf properly.
        loops_since_progress = 0
        while to_unwrap and (
            len(to_elaborate) < 2 or not isinstance(to_elaborate[0], Frame)
        ):
            origin, current, depth = to_unwrap.popleft()
            if isinstance(current, types.FrameType):
                if not isinstance(
                    origin,
                    (
                        types.CoroutineType,
                        types.GeneratorType,
                        types.AsyncGeneratorType,
                    ),
                ):
                    origin = None
                current = Frame(pyframe=current, origin=origin)
            if isinstance(current, Frame):
                loops_since_progress = 0
                to_elaborate.append((current, depth))
                continue
            try:
                unwrapped = unwrap_stackitem(current)
                loops_since_progress += 1
                if loops_since_progress > 100:
                    raise RuntimeError(
                        f"{current!r} has been unwrapped more than 100 times "
                        f"without reaching something irreducible; probably an "
                        f"infinite loop? (next result is {unwrapped!r})"
                    )
            except Exception as ex:
                unwrapped = None
                save_errors.append(ex)
            if unwrapped is None:
                loops_since_progress = 0
                to_elaborate.append((current, depth))
                continue

            if isinstance(unwrapped, FrameIterator):
                it = unwrapped
                unwrapped = []
                while True:
                    try:
                        item = next(it)
                    except StopIteration:
                        break
                    except Exception as ex:
                        save_errors.append(ex)
                        break
                    else:
                        unwrapped.append(item)

            rev_items: Iterable[StackItem]
            if isinstance(unwrapped, collections.abc.Sequence):
                rev_items = reversed(unwrapped)
            else:
                rev_items = (unwrapped,)
            for item in rev_items:
                if item is not None:
                    to_unwrap.appendleft((better_origin(item, origin), item, depth + 1))

        if not to_elaborate:
            break

        if not isinstance(to_elaborate[0][0], Frame):
            # We've reached a leaf
            assert not to_unwrap
            if len(to_elaborate) > 1:
                return list(item[0] for item in to_elaborate)
            return to_elaborate[0][0]

        # Grab the first frame, and fill in context managers if requested
        frame, depth = to_elaborate.popleft()
        assert isinstance(frame, Frame)
        next_inner = to_elaborate[0][0] if to_elaborate else None

        if current_options.with_contexts:
            next_pyframe = next_inner.pyframe if isinstance(next_inner, Frame) else None
            try:
                frame.contexts = contexts_active_in_frame(
                    frame.pyframe, frame.origin, next_pyframe
                )
            except Exception as ex:  # pragma: no cover
                save_errors.append(ex)
            else:
                for context in frame.contexts:
                    try:
                        fill_context(context)
                    except Exception as ex:
                        save_errors.append(ex)

        # Elaborate the frame, see if we should redirect our attention
        # elsewhere
        try:
            replacement = elaborate_frame(frame, next_inner)
        except Exception as ex:
            save_errors.append(ex)
            frame.hide = False
            replacement = PRUNE

        yield frame
        if replacement is None:
            continue
        if isinstance(replacement, collections.abc.Sequence):
            items = replacement
        else:
            items = (replacement,)

        # We're replacing or augmenting the rest of the stack trace (at
        # this depth or below), so anything behind this in the elaboration
        # queue will need to move back to the unwrapping queue
        while to_elaborate:
            # Any origin info has already been stored on the Frame,
            # so we don't need to track it separately (that's the None)
            to_unwrap.appendleft((None, *to_elaborate.pop()))

        if not items or items[-1] is not next_inner:
            # Replacing the rest of the stack trace at this depth
            while to_unwrap and to_unwrap[0][2] >= depth:
                to_unwrap.popleft()
        else:
            # Only inserting new items into the stack trace; since
            # next_inner is in both `items` and `to_unwrap`, remove it
            # from the latter
            to_unwrap.popleft()
        for item in reversed(items):
            to_unwrap.appendleft((better_origin(item, None), item, depth))

    return None


def extract(
    stackitem: StackItem,
    *,
    with_contexts: bool = True,
    recurse_child_tasks: bool = False,
) -> Stack:
    """Extract a `Stack` from *stackitem*.

    *stackitem* may be anything that has a stack associated with it.
    If you want to dump the caller's stack or the stack starting or
    ending with some frame, then either pass a `StackSlice` or use
    :func:`extract_since` or :func:`extract_until`, which are shortcuts
    for passing a `StackSlice` to :func:`extract`. Besides that,
    stackscope also ships with support for threads, greenlets,
    generator iterators (sync and async), coroutine objects, and a few
    more obscure things that might be encountered while traversing
    those; and libraries may add support for more using the
    `unwrap_stackitem` hook.

    If *with_contexts* is True (the default), then each returned
    `Frame` will have a `~Frame.contexts` attribute specifying the
    context managers that are currently active in that frame. If you
    don't care about this information, then specifying ``with_contexts=False``
    will substantially simplify the stack extraction process.

    Some context managers might logically contain other tasks that
    each have their own `Stack`: `trio.Nursery`, `asyncio.TaskGroup`,
    etc. These will be listed as `Context.children` in the returned
    `Frame.contexts` for these context managers. If
    *recurse_child_tasks* is False (the default), then these "child
    tasks" will be rendered as stub `Stack` objects with only a
    `~Stack.root` (the child task object) but no `~Stack.frames`. If
    *recurse_child_tasks* is True, then the child stacks will be fully
    populated, including grandchildren and so on.

    :func:`extract` tries not to throw exceptions; any exception should
    be reported as a bug. Errors encountered during stack extraction
    are reported in the `~Stack.error` attribute of the returned
    object. If multiple errors are encountered, they will be wrapped in
    an `ExceptionGroup`.

    """
    with current_options.push(
        with_contexts=with_contexts, recurse_child_tasks=recurse_child_tasks
    ):
        return extract_child(stackitem, for_task=False)


def extract_child(stackitem: StackItem, *, for_task: bool) -> Stack:
    """Perform a recursive call equivalent to ``extract(stackitem)``, but
    reusing the options that were passed to the original ``extract()``.
    You should only call this from within a :ref:`customization hook
    <customizing>` such as :func:`elaborate_context`.

    If *for_task* is True, then this nested *stackitem* is considered to
    represent an async child task. Its stack will be fully extracted only
    if the outer ``extract()`` call specified ``recurse_child_tasks=True``;
    otherwise you will get a stub `Stack` with a `~Stack.root` but no
    `~Stack.frames`.
    """
    if current_options.recurse_child_tasks is None:
        raise RuntimeError(
            "extract_child() may only be called from within a customization "
            "hook invoked by extract()"
        )
    if for_task and not current_options.recurse_child_tasks:
        return Stack(root=stackitem, frames=[])

    errors: List[Exception] = []
    it = extract_iter(stackitem, errors)
    frames = []
    while True:
        try:
            frames.append(next(it))
        except StopIteration as ex:
            error: Optional[Exception]
            if len(errors) > 1:
                error = ExceptionGroup(
                    "multiple errors encountered while extracting stack", errors
                )
            else:
                error = errors[0] if errors else None
            return Stack(
                root=(None if isinstance(stackitem, StackSlice) else stackitem),
                frames=frames,
                leaf=ex.value,
                error=error,
            )


def extract_outermost(
    stackitem: StackItem,
    *,
    with_contexts: bool = True,
    recurse_child_tasks: bool = False,
) -> Frame:
    """Extract the outermost `Frame` from *stackitem*.

    :func:`extract_outermost` produces the same result as calling :func:`extract`
    and returning the first `Frame` of the returned stack, but might be faster
    since it can stop once it's extracted one frame. If the result has
    no `Frame`\\s, an exception will be thrown.
    """

    with current_options.push(
        with_contexts=with_contexts, recurse_child_tasks=recurse_child_tasks
    ):
        errors: List[Exception] = []
        try:
            return next(extract_iter(stackitem, errors))
        except StopIteration as ex:
            if len(errors) > 1:  # pragma: no cover
                # Rationale for 'no cover': as currently written, only one error can
                # be saved while unwrapping, and errors raised at other points wouldn't
                # reach here because a frame would be available
                raise ExceptionGroup(
                    "multiple errors encountered while extracting stack", errors
                )
            if errors:
                raise errors[0]
            else:
                raise RuntimeError(
                    f"Couldn't extract a frame from {stackitem!r}: unwrapping only "
                    f"reached {ex.value!r}"
                )


def extract_since(
    outer_frame: Optional[types.FrameType],
    *,
    with_contexts: bool = True,
    recurse_child_tasks: bool = False,
) -> Stack:
    """Return a `Stack` reflecting the currently-executing frames that were
    directly or indirectly called by *outer_frame*, including *outer_frame* itself.
    Equivalent to ``extract(StackSlice(outer=outer_frame))`` with more type checking.

    If *outer_frame* is a frame on the current thread's stack, the result
    will start with *outer_frame* and end with the immediate
    caller of :meth:`extract_since`.

    If *outer_frame* is a frame on some other thread's stack, and it remains
    there throughout the traceback extraction process, the resulting
    stack will start with *outer_frame* and end with some frame
    that was recently the innermost frame on that thread's stack.

    .. note:: If *other_frame* is not continuously on the same other thread's
       stack during the extraction process, you're likely to get
       a one-frame stack, maybe with an `~Stack.error`. It's not possible to prevent
       thread switching from within Python code, so we can't do better than
       this without a C extension.

    If *outer_frame* is None, the result contains all frames
    on the current thread's stack, starting with the outermost and ending
    with the immediate caller of :meth:`extract_since`.

    In any other case -- if *outer_frame* belongs to a suspended
    coroutine, generator, greenlet, or if it starts or stops running
    on another thread while :meth:`extract_since` is executing -- you will get
    a `Stack` containing information only on *outer_frame* itself;
    depending on the situation, its `~Stack.error` member might
    describe the reason more information can't be provided.
    """
    if outer_frame is not None and not isinstance(outer_frame, types.FrameType):
        raise TypeError(f"outer_frame must be a frame, not {type(outer_frame)!r}")
    return extract(
        StackSlice(outer=outer_frame),
        with_contexts=with_contexts,
        recurse_child_tasks=recurse_child_tasks,
    )


def extract_until(
    inner_frame: types.FrameType,
    *,
    limit: Union[int, types.FrameType, None] = None,
    with_contexts: bool = True,
    recurse_child_tasks: bool = False,
) -> Stack:
    """Return a `Stack` reflecting the currently executing frames that are
    direct or indirect callers of *inner_frame*, including
    *inner_frame* itself.

    If *inner_frame* belongs to a suspended coroutine or
    generator, or if it otherwise is not linked to other frames
    via its ``f_back`` attribute, then the returned traceback will
    contain only *inner_frame* and not any of its callers.

    If a *limit* is specified, only some of the callers of
    *inner_frame* will be returned. If the *limit* is a frame, then it
    must be an indirect caller of *inner_frame* and it will be the
    first frame in the result; any of its callers will be excluded.
    Otherwise, the *limit* must be a positive integer, and the
    traceback will start with the *limit*'th parent of *inner_frame*.

    Equivalent to ``extract(StackSlice(outer=outer_frame, limit=limit))`` or
    ``extract(StackSlice(outer=outer_frame, inner=limit))`` depending on the
    type of *limit*, except that :func:`extract_until` does more checking of
    its inputs (an exception will be raised if *limit* has an invalid type
    or is a frame that isn't an indirect caller of *inner_frame*).
    """
    opts = {"with_contexts": with_contexts, "recurse_child_tasks": recurse_child_tasks}
    if isinstance(limit, types.FrameType):
        outer_frame: Optional[types.FrameType] = inner_frame
        while (
            outer_frame is not limit
            and outer_frame is not None
            # This last condition catches suspended greenlets in PyPy,
            # whose f_back members form a cycle.
            and outer_frame.f_back is not inner_frame
        ):
            outer_frame = outer_frame.f_back
        if outer_frame is None:
            raise RuntimeError(f"{limit} is not an indirect caller of {inner_frame}")
        return extract(StackSlice(outer=outer_frame, inner=inner_frame), **opts)
    elif isinstance(limit, int) or limit is None:
        return extract(StackSlice(inner=inner_frame, limit=limit), **opts)
    else:
        raise TypeError(
            f"'limit' argument must be a frame or integer, not {type(limit)!r}"
        )


def fill_context(context: Context) -> None:
    """Augment the given newly-constructed `Context` object using the
    context manager hooks (:func:`unwrap_context` and :func:`elaborate_context`),
    calling both hooks in a loop until a steady state is reached.
    """
    if current_options.with_contexts is None:
        # Allow fill_context() to be used outside an extract() call, even
        # though the hooks it's calling might assume they're inside extract()
        with current_options.push(with_contexts=True, recurse_child_tasks=False):
            fill_context(context)
        return

    for _ in range(100):
        if TYPE_CHECKING:
            from typing import ContextManager, AsyncContextManager

            assert isinstance(context.obj, (ContextManager, AsyncContextManager))
        elaborate_context(context.obj, context)
        inner_mgr = unwrap_context(context.obj, context)
        if inner_mgr is None:
            break
        if inner_mgr == PRUNE:
            context.hide = True
            break
        context.obj = inner_mgr
        context.inner_stack = None
        context.children = ()
    else:
        inner_mgr = unwrap_context(context.obj, context)  # type: ignore
        raise RuntimeError(
            f"{context.obj!r} has been unwrapped more than 100 times "
            f"without reaching something irreducible; probably an "
            f"infinite loop? (next result is {inner_mgr!r})"
        )
