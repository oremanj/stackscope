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
    TypeVar,
    Union,
    ContextManager,
    AsyncContextManager,
    overload,
    TYPE_CHECKING,
)
from typing_extensions import ParamSpec

from ._code_dispatch import code_dispatch
from ._types import Frame, Context, StackItem

T = TypeVar("T")
P = ParamSpec("P")


__all__ = [
    "unwrap_stackitem",
    "elaborate_frame",
    "unwrap_context",
    "elaborate_context",
    "customize",
    "fill_context",
    "yields_frames",
    "PRUNE",
]


class _Prune:
    def __repr__(self) -> str:
        return "stackscope.customization.PRUNE"


PRUNE = _Prune()


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
def elaborate_frame(frame: Frame, next_inner: object) -> Union[StackItem, _Prune, None]:
    """Hook for providing additional information about a frame encountered
    during stack traversal. This hook uses `@code_dispatch
    <stackscope.lowlevel.code_dispatch>`, so it can be customized
    based on which function the frame is executing.

    *next_inner* is the thing that the *frame* is currently busy with:
    either the next `Frame` in the list of `Stack.frames`, or else
    the `Stack.leaf` (which may be ``None``) if there is no next frame.

    The :func:`elaborate_frame` hook may modify the attributes of *frame*,
    such as by setting `Frame.hide`. It may also redirect the remainder of
    the stack trace, by returning an object that should be unwrapped to
    become the new *next_inner*. If you don't want to replace the rest of
    the stack trace, then return None. If you want to remove the rest of
    the stack trace and not replace it with anything, then return `PRUNE`.

    """
    if "__tracebackhide__" in frame.pyframe.f_locals:
        frame.hide = True
    return None


@functools.singledispatch
def unwrap_context(
    manager: Union[ContextManager[Any], AsyncContextManager[Any]]
) -> Optional[Union[ContextManager[Any], AsyncContextManager[Any]]]:
    """Hook for extracting an inner context manager from another
    context manager that wraps it. Return None if there is no further
    unwrapping to do.
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


def fill_context(context: Context) -> None:
    """Augment the given newly-constructed `Context` object using the
    context manager hooks (:func:`unwrap_context` and :func:`elaborate_context`),
    calling both hooks in a loop until a steady state is reached.
    """
    for _ in range(100):
        if TYPE_CHECKING:
            from typing import ContextManager, AsyncContextManager

            assert isinstance(context.obj, (ContextManager, AsyncContextManager))
        elaborate_context(context.obj, context)
        inner_mgr = unwrap_context(context.obj)
        if inner_mgr is None:
            break
        context.obj = inner_mgr
    else:
        inner_mgr = unwrap_context(context.obj)  # type: ignore
        raise RuntimeError(
            f"{context.obj!r} has been unwrapped more than 100 times "
            f"without reaching something irreducible; probably an "
            f"infinite loop? (next result is {inner_mgr!r})"
        )


@overload
def customize(
    *,
    hide: bool = False,
    prune: bool = False,
    elaborate: Optional[Callable[[Frame, object], object]] = None,
) -> Callable[[T], T]:
    ...


@overload
def customize(
    __target: T,
    *inner_names: str,
    hide: bool = False,
    prune: bool = False,
    elaborate: Optional[Callable[[Frame, object], object]] = None,
) -> T:
    ...


def customize(
    target: Any = None,
    *inner_names: str,
    hide: bool = False,
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
