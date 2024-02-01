from __future__ import annotations

import collections
import dis
import gc
import inspect
import sys
import threading
import traceback
import types
import warnings
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    cast,
)
from ._types import Context


__all__ = [
    "FrameDetails",
    "ExitingContext",
    "inspect_frame",
    "currently_exiting_context",
    "describe_assignment_target",
    "analyze_with_blocks",
    "contexts_active_in_frame",
    "set_trickery_enabled",
]


class InspectionWarning(RuntimeWarning):
    """Warning raised if something goes awry during frame inspection."""


@dataclass
class FrameDetails:
    """A collection of internal interpreter details relating to a currently
    executing or suspended frame.
    """

    @dataclass
    class FinallyBlock:
        """Information about a currently active exception-catching context
        within the frame.

        On CPython 3.11+, these are inferred from the "zero-cost
        exception handling" ``co_exceptiontable`` attribute of the
        code object. On earlier CPython and all PyPy, they are
        directly tracked at runtime by the frame object.
        """

        handler: int
        level: int

    blocks: List[FinallyBlock] = field(default_factory=list)
    stack: List[object] = field(default_factory=list)


def inspect_frame(frame: types.FrameType) -> FrameDetails:
    """Return a `FrameDetails` object describing the exception handlers
    and evaluation stack for the currently executing or suspended
    frame *frame*.

    There are three implementations of this function: one for CPython 3.8-3.10,
    one for CPython 3.11+, and one for PyPy when using the "incminimark" garbage
    collector. The appropriate one will be chosen automatically.
    """
    # Overwrite this function with the version that's applicable to the
    # running interpreter
    global inspect_frame
    if sys.implementation.name == "cpython":
        if sys.version_info < (3, 11):
            from ._lowlevel_cpython_310 import inspect_frame
        else:
            from ._lowlevel_cpython_311 import inspect_frame
    elif sys.implementation.name == "pypy":
        from ._lowlevel_pypy import inspect_frame
    else:
        raise NotImplementedError("frame details not supported on this interpreter")
    return inspect_frame(frame)


@dataclass
class ExitingContext:
    """Information about the sync or async context manager that's
    currently being exited in a frame.
    """

    is_async: bool
    cleanup_offset: int


if sys.version_info >= (3, 11):

    def _parse_varint(it: Iterator[int]) -> int:
        b = next(it)
        val = b & 63
        while b & 64:
            val <<= 6
            b = next(it)
            val |= b & 63
        return val

    def _parse_exception_table(
        code: types.CodeType,
    ) -> Iterator[Tuple[int, int, int, int, bool]]:
        it = iter(code.co_exceptiontable)
        try:
            while True:
                start = _parse_varint(it) * 2
                length = _parse_varint(it) * 2
                end = start + length - 2  # Present as inclusive, not exclusive
                target = _parse_varint(it) * 2
                dl = _parse_varint(it)
                depth = dl >> 1
                lasti = bool(dl & 1)
                yield start, end, target, depth, lasti
        except StopIteration:
            return


def currently_exiting_context(frame: types.FrameType) -> Optional[ExitingContext]:
    """If *frame* is currently suspended waiting for one of its context
    managers' ``__exit__`` or ``__aexit__`` methods to complete, then
    return an object indicating which context manager is exiting and
    whether it's async or not.  Otherwise return None.

    This function uses some rather involved bytecode introspection,
    but only via public interfaces, and should always be safe to call
    even if something is incorrect in its output. It is used in
    the "referents" mode of :func:`contexts_active_in_frame` as well as
    "trickery" mode, because an exiting context manager is no longer
    referenced by its frame's value stack.

    For the curious, the implementation of this function contains
    extensive comments about the bytecode sequences to which context
    managers compile on different Python versions.
    """
    code = frame.f_code.co_code
    op = dis.opmap
    offs = frame.f_lasti
    if offs < 0:
        return None

    # Our task here is twofold:
    # - figure out whether `frame` is in the middle of a call to a context
    #   manager __exit__ or __aexit__
    # - if so, figure out *which* context manager, in terms that
    #   can be matched up with the result of analyze_with_blocks()
    #
    # This is rather challenging, because the block stack gets popped
    # before the __exit__ method is called, so we can't just consult
    # the block stack like we do for the context managers that aren't
    # currently exiting (on Pythons before 3.11 that had a block
    # stack).  But we can do it if we know something about how the
    # Python bytecode compiler compiles 'with' and 'async with'
    # blocks.
    #
    # There are basically three ways to exit a 'with' block:
    # - falling off the bottom
    # - jumping out (using return, break, or continue)
    # - unwinding due to an exception
    #
    # On 3.7 and earlier, "jumping out" uses the exception-unwinding
    # mechanism, and falling off the bottom falls through into the
    # exception handling block, so there is only one bytecode location
    # where __exit__ or __aexit__ is called. It looks like:
    #
    #     POP_BLOCK          \__ these may be absent if fallthrough is impossible
    #     LOAD_CONST None    /
    #     WITH_CLEANUP_START <-- block stack for exception unwinding points here
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     WITH_CLEANUP_FINISH
    #     END_FINALLY
    #
    # f_lasti will be at WITH_CLEANUP_START for a synchronous call,
    # LOAD_CONST None for an async call on CPython, or YIELD_FROM for an
    # async call on pypy.  Note that the LOAD_CONST may have some
    # EXTENDED_ARGs before it, in weird cases where None is not one of
    # the first 256 constants.
    #
    # On 3.8, falling off the bottom still falls through into the
    # exception handling block, which looks like:
    #
    #     POP_BLOCK          \__ these may be absent if fallthrough is impossible
    #     BEGIN_FINALLY      /
    #     WITH_CLEANUP_START <-- block stack for exception unwinding points here
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     WITH_CLEANUP_FINISH
    #     END_FINALLY
    #
    # But, now each instance of jumping-out inlines its own cleanup. The cleanup
    # sequence is the same as the terminal sequence except that it may have a
    # ROT_TWO after POP_BLOCK (for non-constant 'return' jumps only, not 'break' or
    # 'continue') and it ends with POP_FINALLY rather than END_FINALLY.
    # Since there can be multiple WITH_CLEANUP_START opcodes that clean up
    # the same 'with' block, we can't assume the WITH_CLEANUP_START near f_lasti is
    # the one whose offset is named in the SETUP_WITH that analyze_with_blocks()
    # found. Instead, we'll build a basic control-flow graph to see what offset
    # was in the block that just got POP_BLOCK'ed.
    #
    # On 3.9, this was further split up so that "falling off the bottom" and
    # "unwinding due to an exception" execute different code. Falling off the bottom:
    #
    #     POP_BLOCK
    #     LOAD_CONST None
    #     DUP_TOP
    #     DUP_TOP
    #     CALL_FUNCTION 3    <-- calls __exit__(None, None, None)
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     POP_TOP            <-- return value of __exit__ ignored
    #
    # Jumping out: same as falling off the bottom, except with possible ROT_TWO
    # after POP_BLOCK.
    #
    # Unwinding on exception:
    #
    #     WITH_EXCEPT_START  <-- block stack points here
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     POP_JUMP_IF_TRUE x <-- jumps over the RERAISE
    #     RERAISE
    #     POP_TOP
    #     POP_TOP
    #     POP_TOP
    #     POP_EXCEPT
    #     POP_TOP
    #
    # On 3.11, the block stack was removed (replaced with "zero-cost"
    # exception unwind tables) and a lot of other opcodes were changed
    # as part of general CPython optimization efforts. Falling off the
    # bottom and jumping out of the block now both look like:
    #
    #     LOAD_CONST None
    #     LOAD_CONST None
    #     LOAD_CONST None
    #     PRECALL 2          \__ only on 3.11, not 3.12+
    #     (CACHE)            /
    #     CALL 2             <-- calls __exit__(None, None, None)
    #     (CACHE x3-4)
    #     GET_AWAITABLE 2    \
    #     LOAD_CONST None    |
    #     SEND 3             |
    #     (CACHE)            | CACHE only on 3.12a6+
    #     YIELD_VALUE [X]    |-- only if 'async with'; Y_V has oparg on 3.12+
    #     RESUME 3           |
    #     JUMP_BACKWARD_NO_IN/ERRUPT 4
    #     POP_TOP            <-- return value of __exit__ ignored
    #
    # f_lasti will point at the CALL for a synchronous exit,
    # or at SEND or YIELD_VALUE (depending on whether running
    # or suspended) for an aexit. Unwinding on exception looks like:
    #
    #     PUSH_EXC_INFO      <-- unwind table points here
    #     WITH_EXCEPT_START  <-- calls __exit__
    #     GET_AWAITABLE 2    \
    #     LOAD_CONST None    |
    #     SEND (3-4)         |
    #     (CACHE)            |
    #     YIELD_VALUE        |-- only if 'async with'
    #     RESUME 3           |
    #     JUMP_BACKWARD_NO_IN/ERRUPT 4
    #     CLEANUP_THROW      only on 3.12+
    #     SWAP 2             \__ only if 'async with' on 3.12.0a6+
    #     POP_TOP            /
    #     POP_JUMP_FORWARD_IF_TRUE 4   (jumps over the RERAISEs)
    #     RERAISE 2          reraise new_exc with lasti
    # (*) COPY 3
    #     POP_EXCEPT
    #     RERAISE 1
    #     POP_TOP
    #     POP_EXCEPT
    #     POP_TOP
    #     POP_TOP
    #
    # There are three unwind table entries created for a with block:
    # - From the end of __enter__/__aenter__ to the start of __exit__
    #   (the first non-covered instruction is the LOAD_CONST None),
    #   jump to PUSH_EXC_INFO.
    # - From PUSH_EXC_INFO through the first RERAISE, jump to COPY 3
    #   (starred). The block starting at COPY 3 just restores the
    #   previous exc_info.
    # - At the POP_TOP after the second RERAISE, jump to COPY 3 also.
    #   This one seems like it might be an artifact of some sort.
    #
    # Armed with that context, the below code will hopefully make a bit more sense!

    # See if we're at a context manager __exit__ or __aexit__ call
    is_async = False
    if sys.version_info < (3, 11):
        if code[offs] == op["YIELD_FROM"] or (
            offs + 2 < len(code) and code[offs + 2] == op["YIELD_FROM"]
        ):
            # Async calls have lasti pointing at YIELD_FROM or LOAD_CONST
            is_async = True
            if code[offs] == op["YIELD_FROM"]:
                # If lasti points to YIELD_FROM (pypy convention), move backward
                # to point to LOAD_CONST (cpython convention)
                offs -= 2
    else:  # 3.11 and later
        # Async calls have lasti pointing at YIELD_VALUE or SEND
        if code[offs] == op["YIELD_VALUE"] and offs >= 2:
            offs -= 2
            # SEND can have a CACHE after it in 3.12
            while code[offs] == op["CACHE"] and offs >= 2:
                offs -= 2
            is_async = True
        if code[offs] == op["SEND"]:
            offs -= 2
            is_async = True
            if code[offs] != op["LOAD_CONST"]:  # pragma: no cover
                warnings.warn(
                    f"Surprise during analysis of {frame.f_code!r}: SEND at "
                    f"{offs} not preceded by LOAD_CONST -- please file a bug",
                    InspectionWarning,
                )
                return None
        elif is_async:
            # YIELD_VALUE preceded by something other than SEND is not
            # an aexit
            return None

    def backtrack_over_load_none() -> bool:
        nonlocal offs
        if code[offs] != op["LOAD_CONST"]:
            return False
        arg = code[offs + 1]
        shift = 8
        offs -= 2
        while offs and code[offs] == op["EXTENDED_ARG"]:
            # If LOAD_CONST had an EXTENDED_ARG then skip over those.
            # This is very unlikely -- would require none of the first
            # 256 constants used in a function to be None.
            arg |= code[offs + 1] << shift
            shift += 8
            offs -= 2
        return frame.f_code.co_consts[arg] is None

    if is_async:
        # Backtrack one more to find GET_AWAITABLE
        if not backtrack_over_load_none():  # pragma: no cover
            warnings.warn(
                f"Surprise during analysis of {frame.f_code!r}: LOAD_CONST None "
                f"not found at {offs} before await -- please file a bug",
                InspectionWarning,
            )
            return None
        if code[offs] != op["GET_AWAITABLE"] or (
            sys.version_info >= (3, 11) and code[offs + 1] != 2
        ):
            # Non-awaity use of 'yield from' --> must not be an __aexit__ call
            # we're in. 3.11+ annotate GET_AWAITABLE with an oparg describing
            # the reason; 2 is aexit.
            return None
        # And finally go back one more to reach a CALL_FUNCTION,
        # WITH_CLEANUP_START, or WITH_EXCEPT_START, which can be handled
        # the same as in the synchronous case
        offs -= 2

    if sys.version_info < (3, 9):
        # 3.8: they all use WITH_CLEANUP_START, but there might be multiple instances;
        # backtrack to the preceding POP_BLOCK
        if offs < 4 or code[offs] != op["WITH_CLEANUP_START"]:
            return None
        if code[offs - 2] != op["BEGIN_FINALLY"]:
            # Every jumping-out exit uses BEGIN_FINALLY before WITH_CLEANUP_START,
            # but it's possible for the end-of-block exit to not have a preceding
            # BEGIN_FINALLY. If we're in that situation, then we're in the
            # exception handler, so we already know its offset.
            return ExitingContext(is_async=is_async, cleanup_offset=offs)
        offs -= 4
        if offs and code[offs] == op["ROT_TWO"]:
            offs -= 2
    elif sys.version_info < (3, 11):
        # 3.9 and 3.10: either WITH_EXCEPT_START at the handler
        # offset, or LOAD_CONST DUP_TOP DUP_TOP CALL_FUNCTION
        # somewhere else (that particular sequence is not produced by
        # anything else, as far as I can tell)
        if code[offs] == op["WITH_EXCEPT_START"]:
            return ExitingContext(is_async=is_async, cleanup_offset=offs)
        if offs < 8 or code[offs - 6 : offs + 2 : 2] != bytes(
            [op["LOAD_CONST"], op["DUP_TOP"], op["DUP_TOP"], op["CALL_FUNCTION"]]
        ):
            return None
        # Backtrack from CALL_FUNCTION to the preceding POP_BLOCK
        offs -= 8
        while offs and code[offs] == op["EXTENDED_ARG"]:
            offs -= 2
        if offs and code[offs] == op["ROT_TWO"]:
            offs -= 2
    else:
        # 3.11: either WITH_EXCEPT_START at the handler offset,
        # or LOAD_CONST LOAD_CONST LOAD_CONST [PRECALL] CALL somewhere else
        if code[offs] == op["WITH_EXCEPT_START"]:
            offs -= 2  # back up to PUSH_EXC_INFO
            return ExitingContext(is_async=is_async, cleanup_offset=offs)
        while offs and code[offs] == op["CACHE"]:
            offs -= 2
        if code[offs : offs + 2] != bytes([op["CALL"], 2]):
            # The __exit__ call is encoded as a 2-argument call; the
            # first None is treated as 'self'. This is a pretty good
            # distinguisher of a call to __exit__ from a call to any
            # other function with 3 None arguments.
            return None
        offs -= 2
        if sys.version_info < (3, 12):
            # 3.11 has PRECALL, 3.12+ doesn't
            while offs > 4 and code[offs] == op["CACHE"]:
                offs -= 2
            if code[offs] != op["PRECALL"]:
                return None
            offs -= 2
        if code[offs] != op["LOAD_CONST"]:
            return None
        for _ in range(3):
            if not backtrack_over_load_none():
                return None
        # offs is now the instruction right before the first LOAD_CONST.
        # We expect this to be the last instruction that is covered
        # by the exception handler block that unwinds to call this context's
        # __exit__ in the exception case. Possible exceptions to that rule:
        # - sometimes there's a SWAP before the LOAD_CONSTs
        # - if the with stmt has no body, there might be a NOP to attach
        #   line number information to
        # Neither of these are covered by the exception handler block.
        for _, end, target, *_ in _parse_exception_table(frame.f_code):
            if end == offs or (
                end == offs - 2 and code[offs] in (op["SWAP"], op["NOP"])
            ):
                return ExitingContext(is_async=is_async, cleanup_offset=target)
        warnings.warn(
            f"Surprise during analysis of {frame.f_code!r}: couldn't find an "
            f"exception table entry ending at {offs} just before the call to "
            f"__exit__ -- please file a bug",
            InspectionWarning,
        )
        return None

    # If we get here, we're on 3.8-3.10 and offs is the offset of a
    # POP_BLOCK opcode that popped the context manager block whose offset
    # we want to return.
    if code[offs] != op["POP_BLOCK"]:  # pragma: no cover
        warnings.warn(
            f"Surprise during analysis of {frame.f_code!r}: __exit__ call at {offs} "
            f"not preceded by POP_BLOCK -- please file a bug",
            InspectionWarning,
        )
        return None

    pop_block_offs = offs

    # The block stack on CPython 3.8+ is pretty simple: there's only one
    # type of block used outside exception handling, it's
    # pushed by any of SETUP_FINALLY, SETUP_WITH, SETUP_ASYNC_WITH
    # and popped by POP_BLOCK. This is the block type that will
    # let us match up our POP_BLOCK with its corresponding
    # SETUP_WITH or SETUP_ASYNC_WITH, so it's the only one we need
    # to worry about.
    #
    # PyPy is still using a richer block stack as of 3.10, where there
    # are different types of blocks for try, except, with, and for:
    # https://foss.heptapod.net/pypy/pypy/-/blob/branch/py3.10/pypy/interpreter/pyframe.py#L899
    # This winds up not actually complicating the implementation much though;
    # we just need to recognize SETUP_EXCEPT as well as the others.

    # We represent the state of the block stack at each bytecode offset
    # as a list of the bytecode offsets of exception handlers for
    # each 'finally:'.
    BlockStack = List[int]

    # List of (bytecode offset, block stack that exists just before the
    # instruction at that offset is executed)
    todo: Deque[Tuple[int, BlockStack]] = collections.deque([(0, [])])

    # Bytecode offsets we've already visited
    seen: Set[int] = set()

    # Jumps are denominated in instructions in 3.10+, in bytes previously
    jmul = 2 if sys.version_info >= (3, 10) else 1

    while todo:  # pragma: no branch
        offs, stack = todo.popleft()
        if offs in seen:
            continue
        seen.add(offs)
        arg = code[offs + 1]
        while code[offs] == op["EXTENDED_ARG"]:
            offs += 2
            arg = (arg << 8) | code[offs + 1]
        if code[offs] in dis.hasjabs:
            todo.append((arg * jmul, stack[:]))
        if code[offs] in dis.hasjrel:
            todo.append((offs + 2 + arg * jmul, stack[:]))
            # All three SETUP_* opcodes are in the hasjrel list,
            # because they indicate a possible jump to the handler
            # whose relative offset is named in their argument.
            # That handler is entered with the finally block already
            # popped, so it's correct that we record it in todo
            # before updating the block stack.
            if code[offs] in (
                op.get("SETUP_EXCEPT"),  # pypy only
                op["SETUP_FINALLY"],
                op["SETUP_WITH"],
                op["SETUP_ASYNC_WITH"],
            ):
                stack.append(offs + 2 + arg * jmul)
        if code[offs] == op["POP_BLOCK"]:
            if offs == pop_block_offs:
                # We found the one we're looking for!
                return ExitingContext(is_async=is_async, cleanup_offset=stack[-1])
            stack.pop()
        if code[offs] not in (
            op["JUMP_FORWARD"],
            op["JUMP_ABSOLUTE"],
            op["RETURN_VALUE"],
            op["RAISE_VARARGS"],
            op.get("RERAISE"),  # 3.9 only
        ):
            # The above are the unconditional control transfer opcodes.
            # If we're not one of those, then we'll continue to the following
            # line at least sometimes.
            todo.append((offs + 2, stack))

    warnings.warn(
        f"Surprise during analysis of {frame.f_code!r}: POP_BLOCK at offset "
        f"{pop_block_offs} doesn't appear reachable -- please file a bug",
        InspectionWarning,
    )
    return None


def describe_assignment_target(
    insns: List[dis.Instruction],
    start_idx: int,
) -> Optional[str]:
    """Given that ``insns[start_idx]`` and beyond constitute a series of
    instructions that assign the top-of-stack value somewhere, this
    function returns a string description of where it's getting
    assigned, or None if we can't figure it out.  Understands simple
    names, attributes, subscripting, unpacking, and
    positional-only function calls.
    """
    if start_idx >= len(insns) or insns[start_idx].opname == "POP_TOP":
        return None
    if insns[start_idx].opname == "STORE_FAST":
        return cast(str, insns[start_idx].argval)

    def format_tuple(values: Sequence[str]) -> str:
        if len(values) == 1:
            return "({},)".format(values[0])
        return "({})".format(", ".join(values))

    idx = start_idx

    def next_target() -> str:
        nonlocal idx
        stack: List[str] = []
        while True:
            insn = insns[idx]
            idx += 1
            if insn.opname == "EXTENDED_ARG":
                continue
            if insn.opname in (
                # fmt: off
                "LOAD_GLOBAL", "LOAD_FAST", "LOAD_NAME", "LOAD_DEREF",
                "STORE_GLOBAL", "STORE_FAST", "STORE_NAME", "STORE_DEREF",
                "LOAD_FAST_CHECK",
                # fmt: on
            ):
                stack.append(insn.argval)
            elif insn.opname in (
                # LOOKUP_METHOD is pypy-only
                "LOAD_ATTR",
                "LOAD_METHOD",
                "LOOKUP_METHOD",
                "STORE_ATTR",
            ):
                obj = stack.pop()
                stack.append(f"{obj}.{insn.argval}")
            elif insn.opname == "LOAD_CONST":
                stack.append(insn.argrepr)
            elif insn.opname in ("BINARY_SUBSCR", "STORE_SUBSCR"):
                index = stack.pop()
                container = stack.pop()
                stack.append(f"{container}[{index}]")
            elif insn.opname in ("BINARY_SLICE", "STORE_SLICE"):
                if insn.arg == 3:
                    steprepr = f":{stack.pop()}"
                else:
                    steprepr = ""
                end = stack.pop()
                start = stack.pop()
                container = stack.pop()
                stack.append(f"{container}[{start}:{end}{steprepr}]")
            elif insn.opname == "UNPACK_SEQUENCE":
                values = [next_target() for _ in range(insn.argval)]
                stack.append(format_tuple(values))
            elif insn.opname == "UNPACK_EX":
                before = [next_target() for _ in range(insn.argval & 0xFF)]
                rest = next_target()
                after = [next_target() for _ in range(insn.argval >> 8)]
                stack.append(format_tuple(before + [f"*{rest}"] + after))
            elif insn.opname in ("CALL_FUNCTION", "CALL_METHOD", "CALL"):
                if insn.argval == 0:
                    args = []
                else:
                    args = stack[-insn.argval :]
                    del stack[-insn.argval :]
                func = stack.pop()
                stack.append("{}({})".format(func, ", ".join(args)))
            elif insn.opname == "DUP_TOP":
                # Walrus assignments get here
                stack.append(stack[-1])
            elif insn.opname == "POP_TOP":  # pragma: no cover
                # No known way to get here -- POP_TOP as sole insn is
                # handled at the top of this function
                stack.pop()
            elif insn.opname in ("PRECALL", "CACHE"):
                pass
            else:
                raise ValueError(f"{insn.opname} in assignment target not supported")
            if insn.opname.startswith(("STORE_", "UNPACK_")):
                break
        if len(stack) != 1:
            # Walrus assignments can get here
            raise ValueError("Assignment occurred at unsupported stack depth")
        return stack[0]

    try:
        return next_target()
    except (ValueError, IndexError):
        return None


_can_use_trickery: Optional[bool] = None
_trickery_lock = threading.Lock()


def set_trickery_enabled(enabled: Optional[bool]) -> None:
    """Choose which of the two available implementations of
    :func:`contexts_active_in_frame` should be used. This is a global setting.

    The "trickery" implementation (*enabled* = True) uses `ctypes`
    frame object introspection and bytecode analysis to determine all
    available information about context managers. It works on
    currently-executing frames and there are presently no known
    situations in which it can be fooled.

    The "referents" implementation (*enabled* = False) uses
    :func:`gc.get_referents` to locate ``__exit__`` and ``__aexit__``
    methods referenced by the frame. It doesn't support executing
    frames on CPython (but does on PyPy, and supports suspended frames
    on CPython such as in a generator object/coroutine).  It can't
    tell which line a context manager started on or what name it was
    assigned to; you just get the context manager object and an
    is-async flag. It can be fooled by context managers whose
    ``__exit__`` methods are not implemented by functions that know
    their name is ``__exit__``, and by frames that keep direct
    references to methods called ``__exit__`` for reasons unrelated to
    an active context manager in that frame.  In exchange for these
    limitations, you get increased portability and robustness: it
    should be impossible by construction to crash the interpreter
    using this implementation, while with the "trickery"
    implementation you're putting more trust in our level of testing
    and caution.

    The default on CPython and (PyPy with the default "incminimark"
    garbage collector) is to attempt trickery-based analysis of a simple
    function the first time context managers need to be extracted, and
    to use trickery as long as that works. On other Python implementations
    the "referents" implementation is used. You may request a return to this
    dynamic default by passing *enabled* = None.

    """

    global _can_use_trickery
    with _trickery_lock:
        _can_use_trickery = enabled


def _check_trickery_available() -> bool:
    global _can_use_trickery
    if _can_use_trickery is not None:
        return _can_use_trickery
    with _trickery_lock:
        if _can_use_trickery is not None:  # pragma: no cover
            return _can_use_trickery
        _can_use_trickery = sys.implementation.name == "cpython" or (
            sys.implementation.name == "pypy"
            and sys.pypy_translation_info["translation.gc"]  # type: ignore
            == "incminimark"
            and sys.platform != "win32"  # seeing a segfault of unknown cause in CI
        )
        if _can_use_trickery:
            from contextlib import contextmanager

            @contextmanager
            def noop() -> Iterator[None]:
                yield

            noop_cm = noop()
            contexts_live = []

            def fn() -> Generator[None, None, None]:
                with noop_cm as xyzzy:  # noqa: F841
                    contexts_live.extend(_contexts_active_by_trickery(sys._getframe(0)))
                    yield

            gen = fn()
            try:
                gen.send(None)
                contexts = _contexts_active_by_trickery(gen.gi_frame)
                assert contexts == contexts_live
                assert len(contexts) == 1
                assert contexts[0].varname == "xyzzy" and contexts[0].obj is noop_cm
            except Exception as ex:
                warnings.warn(
                    "Inspection trickery doesn't work on this interpreter: {!r}. "
                    "Information about context managers will be less detailed. "
                    "Please file a bug.".format(ex),
                    InspectionWarning,
                )
                traceback.print_exc()
                _can_use_trickery = False
        else:
            warnings.warn(
                "Inspection trickery is not supported on this interpreter: "
                "need either CPython, or PyPy with incminimark GC on not-Windows. "
                "Information about context managers will be less detailed. ",
                InspectionWarning,
            )
    return _can_use_trickery


def analyze_with_blocks(code: types.CodeType) -> Dict[int, Context]:
    """Analyze the bytecode of the given code object, returning a
    partially filled-in `~stackscope.Context` object for each ``with`` or
    ``async with`` block.

    Each key in the returned mapping uniquely identifies one ``with``
    or ``async with`` block in the function, by specifying the
    bytecode offset of the ``WITH_CLEANUP_START`` (3.8 and earlier),
    ``WITH_EXCEPT_START`` (3.9 and 3.10), or ``PUSH_EXC_INFO`` (3.11+)
    instruction that begins its associated exception handler.  The
    corresponding value is a `~stackscope.Context` object appropriate
    to that block, with the *is_async*, *varname*, and *start_line*
    fields filled in.
    """
    with_block_info: Dict[int, Context] = {}
    current_line = -1
    if sys.version_info >= (3, 11):
        start_to_handler = {
            start: target for start, _, target, *_ in _parse_exception_table(code)
        }
    insns = list(dis.Bytecode(code))
    for idx, insn in enumerate(insns):
        if insn.starts_line is not None:
            current_line = insn.starts_line
        if insn.opname in ("SETUP_WITH", "SETUP_ASYNC_WITH"):
            store_to = describe_assignment_target(insns, idx + 1)
            cleanup_offset = insn.argval
            with_block_info[cleanup_offset] = Context(
                obj=None,
                is_async=(insn.opname == "SETUP_ASYNC_WITH"),
                varname=store_to,
                start_line=current_line,
            )
        elif sys.version_info >= (3, 11) and insn.opname in (
            "BEFORE_WITH",
            "BEFORE_ASYNC_WITH",
        ):
            is_async = insn.opname == "BEFORE_ASYNC_WITH"
            # 7: BEFORE_ASYNC_WITH, GET_AWAITABLE 1, LOAD_CONST None, SEND 3,
            #    YIELD_VALUE, RESUME 3, JUMP_BACKWARD_NO_INTERRUPT 4
            skip_insns = 7 if is_async else 1
            # Allow for EXTENDED_ARG(s) before LOAD_CONST None
            while is_async and insns[idx + skip_insns - 5].opname == "EXTENDED_ARG":
                skip_insns += 1
            if is_async:
                if (
                    sys.version_info >= (3, 12)
                    and insns[idx + skip_insns].opname == "CLEANUP_THROW"
                ):
                    # This can show up before END_SEND in some cases such as an
                    # async CM inside a finally block. It is not reachable
                    # except via an exception; the prior SEND arg jumps over it.
                    skip_insns += 1
                if sys.version_info >= (3, 12, 0, "beta", 1):
                    # After 411b169281 there is an END_SEND bytecode after
                    # the jump, to deal with changed SEND stackeffect more
                    # efficiently than the below
                    skip_insns += 1
                elif sys.version_info >= (3, 12, 0, "alpha", 6):  # pragma: no cover
                    # 160f2fe2b9 changed SEND stackeffect, resulting in
                    # an extra SWAP 2 + POP_TOP
                    skip_insns += 2
            store_to = describe_assignment_target(insns, idx + skip_insns)
            cleanup_offset = start_to_handler[insns[idx + skip_insns].offset]
            with_block_info[cleanup_offset] = Context(
                obj=None,
                is_async=is_async,
                varname=store_to,
                start_line=current_line,
            )
    return with_block_info


def contexts_active_in_frame(
    frame: types.FrameType,
    origin: Any = None,
    next_inner: Optional[types.FrameType] = None,
) -> List[Context]:
    """Inspects the given *frame* to try to determine which context
    managers are currently active; returns a list of
    `stackscope.Context` objects describing the active context
    managers from outermost to innermost.

    This is the entry point to the frame-analysis functions in
    `stackscope.lowlevel` from the rest of the library.  All the others
    are indirectly called from this one.

    There are two implementations of this function with different
    tradeoffs.  By default, the most capable one that appears to work
    in your environment will be chosen; you can override this choice
    using :func:`set_trickery_enabled`. See the documentation of that
    function for more information.

    If *frame* is the frame of a generator or coroutine, then you are
    encouraged to pass that generator or coroutine as the *origin* parameter.
    This is required in order to get context manager information on CPython 3.11
    and later when using the (fallback/safer) "referents" implementation.

    *next_inner* should be the frame that *frame* is currently calling, if any.
    This is necessary to set the `stackscope.Context.obj` attribute correctly
    on a context manager that is currently exiting.

    """
    ret: List[Context] = []
    if _check_trickery_available():
        try:
            ret = _contexts_active_by_trickery(frame)
        except Exception as ex:
            warnings.warn(
                "Inspection trickery failed on frame {!r}: {!r}. "
                "Information about context managers will be less detailed. "
                "Please file a bug.".format(frame, ex),
                InspectionWarning,
            )
            traceback.print_exc()
            ret = _contexts_active_by_referents(frame, origin)
    else:
        ret = _contexts_active_by_referents(frame, origin)

    if ret and ret[-1].is_exiting and next_inner is not None:
        # Infer the context manager being exited based on the
        # self argument to its __exit__ or __aexit__ method in
        # the next frame
        args = inspect.getargvalues(next_inner)
        if args.args:
            ret[-1].obj = args.locals[args.args[0]]

    return ret


def _contexts_active_by_referents(frame: types.FrameType, origin: Any) -> List[Context]:
    """Version of `contexts_active_in_frame` that relies only on
    `gc.get_referents`, and thus can be used on any Python interpreter
    that supports the `gc` module.

    On CPython, this doesn't support running frames -- only those that
    are suspended at a yield or await. It can't determine the
    `~Context.varname` or `~Context.start_line` members of
    `Context`, and it's possible to fool it in some unlikely
    circumstances (e.g., if you have a local variable that points
    directly to an ``__exit__`` or ``__aexit__`` method, or if a context
    manager's ``__exit__`` method is a static method or thinks its name
    is something other than ``__exit__``).
    """
    ret: List[Context] = []
    root: Any = frame
    if sys.version_info >= (3, 11) and isinstance(
        origin, (types.GeneratorType, types.CoroutineType, types.AsyncGeneratorType)
    ):
        # On 3.11+ the stack of a generator's frame is 'owned' for GC purposes
        # by the generator, not the frame object
        root = origin

    for referent in gc.get_referents(root):
        if isinstance(referent, types.MethodType) and referent.__func__.__name__ in (
            "__exit__",
            "__aexit__",
        ):
            # 'with' and 'async with' statements push a reference to the
            # __exit__ or __aexit__ method that they'll call when exiting.
            ret.append(
                Context(
                    is_async="a" in referent.__func__.__name__,
                    obj=referent.__self__,
                )
            )
    exiting = currently_exiting_context(frame)
    if exiting is not None:
        ret.append(Context(obj=None, is_async=exiting.is_async, is_exiting=True))
    return ret


def _contexts_active_by_trickery(frame: types.FrameType) -> List[Context]:
    """Version of `contexts_active_in_frame` that provides full information
    on tested versions of CPython and PyPy by accessing the block stack.
    This is an internal implementation detail so it may stop working as
    Python's internals change. The inspectors use lots of assertions so
    such failures will hopefully downgrade to the by_referents version,
    but there are no guarantees -- they might just segfault if we get
    really unlucky.
    """
    with_block_info = analyze_with_blocks(frame.f_code)
    frame_details = inspect_frame(frame)
    with_blocks = [
        block for block in frame_details.blocks if block.handler in with_block_info
    ]
    exiting = currently_exiting_context(frame)
    ret = [
        replace(
            with_block_info[block.handler],
            obj=frame_details.stack[block.level - 1].__self__,  # type: ignore
        )
        for block in with_blocks
    ]
    if exiting is not None:
        ret.append(replace(with_block_info[exiting.cleanup_offset], is_exiting=True))
    locals_by_id = {}
    for name, value in frame.f_locals.items():
        locals_by_id[id(value)] = name
    for idx, info in enumerate(ret):
        if info.obj is not None and info.varname is None:
            ret[idx] = replace(info, varname=locals_by_id.get(id(info.obj)))
    return ret
