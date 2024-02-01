from __future__ import annotations

# Interfaces for customizing stack extraction to the needs of particular
# libraries that might be represented in the stack.

import dataclasses
import functools
import gc
import sys
import types
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
    ContextManager,
    AsyncContextManager,
    overload,
    TYPE_CHECKING,
)
from typing_extensions import ParamSpec

from ._code_dispatch import code_dispatch
from ._types import Stack, Frame, Context, StackItem

T = TypeVar("T")
P = ParamSpec("P")


__all__ = [
    "unwrap_stackitem",
    "elaborate_frame",
    "unwrap_context",
    "unwrap_context_generator",
    "elaborate_context",
    "customize",
    "yields_frames",
    "PRUNE",
]


PRUNE = ()


@dataclasses.dataclass
class FrameIterator(Iterator[T]):
    inner: Iterator[T]

    def __next__(self) -> T:
        return next(self.inner)


def yields_frames(fn: Callable[P, Iterator[T]]) -> Callable[P, FrameIterator[T]]:
    """Decorator for an :func:`unwrap_stackitem` implementation which allows
    the series of stack items to be yielded one-by-one instead of returned
    in a sequence.

    The main benefit of this approach is that previously-yielded
    frames can be preserved even if an exception is raised. It is used
    by the built-in glue that handles `StackSlice` objects. A
    decorator is needed to distinguish an iterator of stack items
    (which should be unpacked and treated one-by-one) from a generator
    iterator with a different purpose (which is a common stack item
    that we definitely should not iterate over).

    Put ``@yields_frames`` underneath ``@unwrap_stackitem.register``.

    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> FrameIterator[T]:
        return FrameIterator(fn(*args, **kwargs))

    return wrapper


@functools.singledispatch
def unwrap_stackitem(
    item: StackItem,
) -> Union[StackItem, Sequence[StackItem], FrameIterator[StackItem], None]:
    """Hook for turning something encountered during stack traversal into
    one or more objects that are easier to understand than it, eventually
    resulting in a Python frame. May return a single object, a sequence of
    objects, or None if we don't know how to unwrap any further. When
    extracting a stack, :func:`unwrap_stackitem` will be applied repeatedly
    until a series of frames results.

    The built-in unwrapping rules serve as good examples here:

    * Unwrapping a coroutine object, generator iterator, or async generator
      iterator produces a tuple ``(frame, next)`` where *frame* is a Python
      frame object and *next* is the thing being awaited or yielded-from
      (potentially another coroutine object, generator iterator, etc).

    * Unwrapping a `threading.Thread` produces the sequence of frames that
      form the thread's stack, from outermost to innermost.

    * Unwrapping an async generator ``asend()`` or ``athrow()`` awaitable
      produces the async generator it is operating on.

    * Unwrapping a ``coroutine_wrapper`` object produces the coroutine
      it is wrapping. This allows stackscope to look inside most
      awaitable objects.  (A "coroutine wrapper" is the object returned when
      you call ``coro.__await__()`` on a coroutine object, which acts
      like a coroutine object except that it also implements ``__next__``.)

    """
    return None


def _code_of_frame(frame: Frame) -> types.CodeType:
    return frame.pyframe.f_code


@code_dispatch(_code_of_frame)
def elaborate_frame(
    frame: Frame, next_inner: object
) -> Union[StackItem, Sequence[StackItem], None]:
    """Hook for providing additional information about a frame encountered
    during stack traversal. This hook uses `@code_dispatch
    <stackscope.lowlevel.code_dispatch>`, so it can be customized
    based on which function the frame is executing.

    *next_inner* is the thing that the *frame* is currently busy with:
    either the next `Frame` in the list of `Stack.frames`, or else
    the `Stack.leaf` (which may be ``None``) if there is no next frame.

    The :func:`elaborate_frame` hook may modify the attributes of *frame*,
    such as by setting `Frame.hide`. It may also redirect the remainder of
    the stack trace, by returning an object or sequence of objects that
    should be unwrapped to become the new *next_inner*. If the return value
    is a sequence and it ends with *next_inner*, then the items before
    *next_inner* are inserted before the remainder of the stack trace
    instead of replacing it. If you don't want to affect the rest of the
    stack trace, then return None (equivalent to *next_inner*). If you want to
    remove the rest of the stack trace and not replace it with anything,
    then return `PRUNE` (which is equivalent to an empty tuple).
    """
    if "__tracebackhide__" in frame.pyframe.f_locals:
        frame.hide = True
    return None


@functools.singledispatch
def unwrap_context(
    manager: ContextManager[Any] | AsyncContextManager[Any],
    context: Context,
) -> None | ContextManager[Any] | AsyncContextManager[Any] | tuple[()]:
    """Hook for extracting an inner context manager from another
    context manager that wraps it. The stackscope *context* object is
    also provided in case it's useful. Return None if there is no further
    unwrapping to do. Return `PRUNE` (equivalent to an empty tuple)
    to hide this context manager from the traceback. Unlike
    :func:`unwrap_stackitem`, it is not currently supported to let the
    result of unwrapping a context manager be a sequence of multiple
    context managers.

    .. note:: If the original context manager is currently exiting, the
       frames implementing its ``__exit__`` will appear on the stack
       regardless of any unwrapping you do here. You can customize
       :func:`elaborate_frame` for the appropriate ``__exit__`` if you
       want to affect the display there as well.
    """
    return None


@code_dispatch(_code_of_frame)
def unwrap_context_generator(
    frame: Frame, context: Context
) -> None | ContextManager[Any] | AsyncContextManager[Any] | tuple[()]:
    """Hook for extracting an inner context manager from the outermost
    *frame* of a generator-based context manager that wraps it.  This
    hook uses `@code_dispatch <stackscope.lowlevel.code_dispatch>`, so
    it can be customized based on the identity of the function that
    implements the context manager.  Apart from that, its semantics
    are equivalent to :func:`unwrap_context`.

    .. note:: If the context manager you're unwrapping uses ``yield from``,
       it's possible that you'll need to access callees of *frame* to
       implement your logic. You can find these using `Context.inner_stack`,
       or if that's None because the context is currently exiting, you can
       reconstruct it using ``stackscope.extract(context.obj.gen)``.
    """
    return None


@functools.singledispatch
def elaborate_context(
    manager: Union[ContextManager[Any], AsyncContextManager[Any]],
    context: Context,
) -> None:
    """Hook for providing additional information about a context manager
    encountered during stack traversal. It should modify the
    attributes of the provided *context* object (a `Context`) based on
    the provided *manager* (the actual context manager, i.e., the thing
    whose type has ``__enter__`` and ``__exit__`` attributes).
    """


@overload
def customize(
    *,
    hide: bool = False,
    hide_line: bool = False,
    prune: bool = False,
    elaborate: Optional[Callable[[Frame, object], object]] = None,
) -> Callable[[T], T]: ...


@overload
def customize(
    __target: T,
    *inner_names: str,
    hide: bool = False,
    hide_line: bool = False,
    prune: bool = False,
    elaborate: Optional[Callable[[Frame, object], object]] = None,
) -> T: ...


def customize(
    target: Any = None,
    *inner_names: str,
    hide: bool = False,
    hide_line: bool = False,
    prune: bool = False,
    elaborate: Optional[Callable[[Frame, object], object]] = None,
) -> Any:
    """Shorthand for common :func:`elaborate_frame` customizations
    which affect how stack extraction interacts with invocations of
    specific functions.

    ``(target, *inner_names)`` identifies a code object; all frames
    executing that code object will receive the customizations
    specified by the keyword arguments to :func:`customize`.
    Typically you would a function as *target*, with no *inner_names*,
    to customize frames that are executing that function. You only
    need *inner_names* if you're trying to name a nested function; see
    :func:`~stackscope.lowlevel.get_code` for details.

    If you don't specify a *target*, then :func:`customize` returns a
    partially bound invocation of itself so that you can use it as a
    decorator.  The target in that case is the decorated function; the
    decorator returns that function unchanged. (Note the distinction:
    ``@customize`` decorates the function whose frames get the custom
    behavior, while ``@elaborate_frame.register`` decorates the function
    that implements the custom behavior.)

    The customizations are specified by the keyword arguments you pass:

    * If *elaborate* is specified, then it will be registered as an
      `elaborate_frame` hook for the matching frames.

    * If *hide* is True, then the matching frames will have their
      `Frame.hide` attribute set to True, indicating that they should
      not be shown by default when printing the stack.

    * If *hide_line* is True, then the matching frames will have their
      `Frame.hide_line` attribute set to True, indicating that the executing
      line should not be shown by default when printing the stack (but the
      function info and contexts still will be).

    * If *prune* is True, then direct and indirect callees of
      the matching frames will not be included when extracting the
      stack. This option only has effect if *elaborate* either is
      unspecified or returns None.
    """
    if target is None:
        return functools.partial(customize, hide=hide, prune=prune, elaborate=elaborate)

    @elaborate_frame.register(target, *inner_names)
    def customize_it(frame: Frame, next_inner: object) -> Any:
        if hide:
            frame.hide = True
        if elaborate:
            replacement = elaborate(frame, next_inner)
            if replacement is not None:  # pragma: no branch
                return replacement
        return PRUNE if prune else None

    return target
