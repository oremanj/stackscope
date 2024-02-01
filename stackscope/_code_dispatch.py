from __future__ import annotations

# Tools for describing customizations that apply to particular code objects.

import functools
import inspect
import sys
import types
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Tuple,
    TypeVar,
    Union,
    Optional,
    cast,
    overload,
    TYPE_CHECKING,
)
from typing_extensions import Concatenate, ParamSpec

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

if not TYPE_CHECKING and sys.implementation.name == "pypy":
    # typing_extensions uses some hacks to make ParamSpec work like a TypeVar;
    # they don't work on pypy
    P = TypeVar("P")

    class _ConcatenateForm:
        def __getitem__(self, args):
            return list(args)

    Concatenate = _ConcatenateForm()  # noqa: F811


__all__ = ["IdentityDict", "get_code", "code_dispatch"]


class IdentityDict(MutableMapping[K, V]):
    """A dict that hashes objects by their identity, not their contents.

    We use this to track code objects, since they have an expensive-to-compute
    hash which is not cached. You can probably think of other uses too.

    Single item lookup, assignment, deletion, and ``setdefault()`` are
    thread-safe because they are each implented in terms of a single call to
    a method of an underlying native dictionary object.
    """

    __slots__ = ("_data",)

    def __init__(self, items: Iterable[Tuple[K, V]] = ()):
        self._data = {id(k): (k, v) for k, v in items}

    def __repr__(self) -> str:
        return "IdentityDict([{}])".format(
            ", ".join(f"({k!r}, {v!r})" for k, v in self._data.values())
        )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return (k for k, v in self._data.values())

    def __getitem__(self, key: K) -> V:
        return self._data[id(key)][1]

    def __setitem__(self, key: K, value: V) -> None:
        self._data[id(key)] = key, value

    def __delitem__(self, key: K) -> None:
        del self._data[id(key)]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IdentityDict):
            return self._data == other._data
        return super().__eq__(other)

    _marker = object()

    @overload
    def pop(self, key: K) -> V: ...

    @overload
    def pop(self, key: K, default: Union[V, T] = ...) -> Union[V, T]: ...

    def pop(self, key: K, default: object = _marker) -> object:
        try:
            return self._data.pop(id(key))[1]
        except KeyError:
            if default is self._marker:
                raise KeyError(key) from None
            return default

    def popitem(self) -> Tuple[K, V]:
        return self._data.popitem()[1]

    def clear(self) -> None:
        self._data.clear()

    def setdefault(self, key: K, default: V = cast(V, None)) -> V:
        return self._data.setdefault(id(key), (key, default))[1]


def get_code(thing: object, *nested_names: str) -> types.CodeType:
    """Return the code object that implements the behavior of *thing* or of
    a function nested inside it.

    *thing* should be a code object or a callable: a function, method,
    or `functools.partial` object. If *thing* is a callable, it will be
    unwrapped if necessary to yield a function (looking inside methods,
    following decorator-produced wrappers to the original decorated function,
    and so forth), and then the function's code object will be extracted.
    (If *thing* is already a code object, it is used as-is at this stage.)

    If no *nested_names* are provided, then the code object obtained
    through the actions in the previous paragraph is returned
    directly. Otherwise, each of the *nested_names* is used to look up
    a function or class whose definition is nested inside the function
    we're working with. The code object that results at the end of
    this traversal is returned from :func:`get_code`.

    A somewhat contrived example::

        def make_calc(mult, add):
            class C:
                def __init__(self, addend):
                    self.addend = addend
                def calculate(val):
                    return val * mult + self.addend
            return C(add)

        calc_code = get_code(make_calc, "C", "calculate")
        calc_obj = make_calc(5, 2)
        assert calc_obj.calculate.__func__.__code__ is calc_code
        assert calc_obj.calculate(3) == (3 * 5) + 2 == 17
    """

    while True:
        if isinstance(thing, functools.partial):
            thing = thing.func
            continue
        if isinstance(thing, (types.MethodType, classmethod, staticmethod)):
            thing = thing.__func__
            continue
        if hasattr(thing, "__wrapped__"):
            thing = inspect.unwrap(cast(types.FunctionType, thing))
            continue
        break

    code: types.CodeType
    if isinstance(thing, types.FunctionType):
        code = thing.__code__
    elif isinstance(thing, types.CodeType):
        code = thing
    else:
        raise TypeError(f"Don't know how to extract a code object from {thing!r}")

    top_name = code.co_name
    for idx, name in enumerate(nested_names):
        for const in code.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == name:
                code = const
                break
        else:
            raise ValueError(
                f"Couldn't find a function or class named {name!r} in "
                + ".".join([top_name, *nested_names[:idx]])
            )

    return code


class _CodeDispatcher(Generic[T, P, R]):
    """Interface definition for the thing returned by :func:`code_dispatch`."""

    if TYPE_CHECKING:
        registry: Mapping[types.CodeType, Callable[Concatenate[T, P], R]]

        @overload
        def register(
            self,
            code: Union[types.CodeType, Callable[..., Any]],
            *nested_names: str,
            func: Callable[Concatenate[T, P], R],
        ) -> Callable[Concatenate[T, P], R]: ...

        @overload
        def register(
            self,
            code: Union[types.CodeType, Callable[..., Any]],
            *nested_names: str,
            func: None = None,
        ) -> Callable[
            [Callable[Concatenate[T, P], R]], Callable[Concatenate[T, P], R]
        ]: ...

        def register(
            self,
            code: Union[types.CodeType, Callable[..., Any]],
            *nested_names: str,
            func: Optional[Callable[Concatenate[T, P], R]] = None,
        ) -> Callable[..., Any]:
            raise NotImplementedError

        def dispatch(self, arg: T) -> Callable[Concatenate[T, P], R]: ...

        def __call__(self, __first_arg: T, *args: P.args, **kwargs: P.kwargs) -> R: ...


def code_dispatch(
    code_from_arg: Callable[[T], types.CodeType]
) -> Callable[[Callable[Concatenate[T, P], R]], _CodeDispatcher[T, P, R]]:
    """Decorator for a function that should dispatch to different
    specializations depending on the code object associated with its
    first argument. Similar to `functools.singledispatch`, except that
    `~functools.singledispatch` dispatches on the type of its first
    argument rather than the implementation of something associated
    with its first argument.

    "Associated with" is determined by the required *code_from_arg* argument,
    a callable which will be used to extract a code object from the first
    argument of the decorated function each time it is called. For example,
    a function that operates on Python frames, and wants to operate differently
    depending on what function those frames are executing, might pass
    ``lambda frame: frame.f_code`` as its *code_from_arg*.

    Example use: creating a registry of "should this frame be hidden?" logic::

        # The argument to code_dispatch() is used to obtain a code object
        # from the first argument of each call to should_hide_frame()
        @code_dispatch(lambda frame: frame.f_code)
        def should_hide_frame(frame: types.FrameType) -> bool:
            return "__tracebackhide__" in frame.f_locals

        # Hide any outcome.capture() or outcome.acapture() frames
        @should_hide_frame.register(outcome.capture)
        @should_hide_frame.register(outcome.acapture)
        def hide_captures(frame: types.FrameType) -> bool:
            return True

    As with `~functools.singledispatch`, the decorated function has some
    additional attributes in addition to being callable:

    * Use ``@func.register(target, *names)`` as a decorator to register
      specializations. ``(target, *names)`` identifies a code object as
      documented under :func:`get_code`. ``register()`` also supports
      invocation as a non-decorator ``func.register(target, *names, impl)``.

    * Use ``func.dispatch(first_arg)`` to return the function that will
      be invoked for calls to ``func(first_arg, ...)``.

    * ``func.registry`` is a read-only mapping whose keys are code objects
      and whose values are the corresponding specializations of the
      `@code_dispatch <code_dispatch>`-decorated function.

    """

    def decorate(
        default_impl: Callable[Concatenate[T, P], R]
    ) -> _CodeDispatcher[T, P, R]:
        registry = IdentityDict[types.CodeType, Callable[Concatenate[T, P], R]]()

        def dispatch(arg: T) -> Callable[Concatenate[T, P], R]:
            code = code_from_arg(arg)
            try:
                return registry[code]
            except KeyError:
                return default_impl

        def register(
            code: Union[types.CodeType, Callable[..., Any]],
            *nested_names: str,
            func: Optional[Callable[Concatenate[T, P], R]] = None,
        ) -> object:
            if func is None and nested_names and callable(nested_names[-1]):
                func = cast(Callable[..., Any], nested_names[-1])
                nested_names = nested_names[:-1]
            if func is None:
                return lambda fn: register(code, *nested_names, func=fn)

            actual_code = get_code(code, *nested_names)
            registry[actual_code] = func
            return func

        def wrapper(__first_arg: T, *args: P.args, **kwargs: P.kwargs) -> R:
            return dispatch(__first_arg)(__first_arg, *args, **kwargs)

        wrapper.register = register  # type: ignore
        wrapper.dispatch = dispatch  # type: ignore
        wrapper.registry = types.MappingProxyType(registry)  # type: ignore
        functools.update_wrapper(wrapper, default_impl)
        return cast(_CodeDispatcher[T, "P", R], wrapper)

    return decorate
