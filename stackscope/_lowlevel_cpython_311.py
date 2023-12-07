import bisect
import ctypes
import dis
import gc
import sys
from types import FrameType
from typing import List, Tuple, Type, Generator, Optional

assert sys.version_info >= (3, 11)

from ._lowlevel import FrameDetails


# Reference for the frame changes in 3.11:
# https://github.com/python/cpython/blob/3.11/Objects/frame_layout.md


wordsize = ctypes.sizeof(ctypes.c_size_t)


class InterpreterFrame(ctypes.Structure):
    # 3.11 func, globals, builtins, locals, code, frame_obj, previous
    # 3.12 code, previous, func, globals, builtins, locals, frame_obj

    _fields_: List[Tuple[str, Type["ctypes._CData"]]] = []
    if sys.version_info >= (3, 12):
        _fields_ += [
            ("f_code", ctypes.c_size_t),  # PyCodeObject*
            ("previous", ctypes.c_size_t),  # _PyInterpreterFrame*
            ("f_func", ctypes.c_size_t),  # PyObject*, now called f_funcobj
            ("f_globals", ctypes.c_size_t),
            ("f_builtins", ctypes.c_size_t),
            ("f_locals", ctypes.c_size_t),
            ("frame_obj", ctypes.c_size_t),  # PyFrameObject*
            ("prev_instr", ctypes.POINTER(ctypes.c_ushort)),
            ("stacktop", ctypes.c_int),  # offset of TOS from localsplus, or -1
            ("yield_offset", ctypes.c_ushort),
            ("owner", ctypes.c_byte),
        ]
    else:  # 3.11
        _fields_ += [
            ("f_func", ctypes.c_size_t),  # PyObject*
            ("f_globals", ctypes.c_size_t),
            ("f_builtins", ctypes.c_size_t),
            ("f_locals", ctypes.c_size_t),
            ("f_code", ctypes.c_size_t),  # PyCodeObject*
            ("frame_obj", ctypes.c_size_t),  # PyFrameObject*
            ("previous", ctypes.c_size_t),  # _PyInterpreterFrame*
            ("prev_instr", ctypes.POINTER(ctypes.c_ushort)),
            ("stacktop", ctypes.c_int),  # offset of TOS from localsplus, or -1
            ("is_entry", ctypes.c_byte),
            ("owner", ctypes.c_byte),
        ]
    if (3, 12) <= sys.version_info < (3, 12, 0, "alpha", 6):  # pragma: no cover
        # Fields were reordered in 3.12.0a6; handle previous alpha releases
        # temporarily
        f_code = _fields_.pop(0)
        previous = _fields_.pop(0)
        _fields_.insert(4, f_code)
        _fields_.insert(6, previous)
        del f_code, previous

    # localsplus: locals and stack start here, inline after the above header


class FrameObject(ctypes.Structure):
    _fields_: List[Tuple[str, Type["ctypes._CData"]]] = [
        ("ob_refcnt", ctypes.c_size_t),  # reference count
        ("ob_type", ctypes.c_size_t),  # PyTypeObject*
        ("f_back", ctypes.c_size_t),  # PyFrameObject*, may be null
        ("f_frame", ctypes.POINTER(InterpreterFrame)),
        ("f_trace", ctypes.c_size_t),  # PyObject*
        ("f_lineno", ctypes.c_int),
        # There are more fields after this, but they diverge between versions
        # and we don't care about them.
    ]

    extra_header_bytes = object().__sizeof__() - 2 * wordsize
    if extra_header_bytes:  # pragma: no cover
        _fields_.insert(0, ("ob_debug", ctypes.c_byte * extra_header_bytes))


FRAME_OWNED_BY_THREAD = 0
FRAME_OWNED_BY_GENERATOR = 1
FRAME_OWNED_BY_FRAME_OBJECT = 2


def assert_frame_cases() -> None:
    this_frame = sys._getframe(0)
    this_iframe = FrameObject.from_address(id(this_frame)).f_frame.contents
    assert this_iframe.owner == FRAME_OWNED_BY_THREAD
    assert this_iframe.f_func == id(assert_frame_cases)
    assert this_iframe.frame_obj == id(this_frame)

    def gen(x: int) -> Generator[None, None, None]:
        yield  # pragma: no cover

    gi = gen(42)
    gen_iframe = FrameObject.from_address(id(gi.gi_frame)).f_frame.contents
    assert gen_iframe.owner == FRAME_OWNED_BY_GENERATOR
    assert gen_iframe.frame_obj == id(gi.gi_frame)
    # Make sure locals start at the InterpreterFrame offset we think they do
    assert ctypes.c_size_t.from_address(
        ctypes.addressof(gen_iframe) + ctypes.sizeof(InterpreterFrame)
    ).value == id(gi.gi_frame.f_locals["x"])

    def outlived() -> FrameType:
        return sys._getframe(0)

    outlived_frame = outlived()
    outlived_iframe = FrameObject.from_address(id(outlived_frame)).f_frame.contents
    assert outlived_iframe.owner == FRAME_OWNED_BY_FRAME_OBJECT
    # frame_obj is filled in only when the iframe is not embedded inside it
    assert outlived_iframe.frame_obj == 0


assert_frame_cases()


def inspect_frame(frame: FrameType) -> FrameDetails:
    assert sys.implementation.name == "cpython" and sys.version_info >= (3, 11)

    details = FrameDetails()

    # Basic sanity checks cross-referencing the values we can get from Python
    # with their Python values
    frame_raw = FrameObject.from_address(id(frame))
    refcnt = frame_raw.ob_refcnt
    assert refcnt + 1 == sys.getrefcount(frame)
    assert frame_raw.ob_type == id(type(frame))
    # The f_back struct member is generally null, so not good for
    # sanity checks; the f_back Python attribute does a loop over
    # f_frame.previous.

    # The interpreter frame object has a fixed-length part followed by
    # 'localsplus' which is a variable-length array of PyObject*. The
    # array contains co_nlocalplus slots for local/free/cell vars,
    # followed by co_stacksize slots for the bytecode interpreter stack.
    # Figure out where localsplus and the value stack start.
    co = frame.f_code
    localsplus_offset = ctypes.sizeof(InterpreterFrame)
    # NB: unlike on previous versions, it's not necessarily the case
    # that the localsplus are laid out in locals-cellvars-freevars
    # order.  (Newly in 3.11, arguments that are closed over appear in
    # both varnames and cellvars.)  You need to use the undocumented
    # CodeType._varname_from_oparg function to get the name of the
    # thing at a given index.
    stack_start_offset = localsplus_offset + wordsize * (
        len(set(co.co_varnames + co.co_cellvars)) + len(co.co_freevars)
    )
    end_offset = stack_start_offset + wordsize * co.co_stacksize

    from ._lowlevel import _parse_exception_table

    # Obtain a consistent snapshot of lasti + stack. This might require
    # more than one attempt if the frame we're looking at is currently
    # executing on another thread.
    for _ in range(10):
        lasti_before = frame.f_lasti
        for start, end, _, depth, _ in _parse_exception_table(co):
            if start <= lasti_before <= end:
                handler_depth = depth
                break
        else:
            handler_depth = 0

        try:
            # Unavoidable hazard (unless we write a C extension): If
            # this frame is currently executing on another thread, and
            # returns while we're looking at it, the frame object will
            # remain valid but the InterpreterFrame will move (to be
            # inside the frame object). We thus risk a dangling
            # pointer and should be extremely paranoid about anything
            # we read from iframe_raw. All accesses to the
            # InterpreterFrame object are kept within this
            # consistency-checked loop for that reason.
            iframe_raw = frame_raw.f_frame.contents
            assert iframe_raw.f_globals == id(frame.f_globals)
            assert iframe_raw.f_builtins == id(frame.f_builtins)
            assert iframe_raw.f_code == id(frame.f_code)
            # frame_obj is null if this iframe is owned by the frame object (thus
            # physically contained within it), to avoid a circular reference
            assert iframe_raw.frame_obj in (0, id(frame))

            # Figure out what portion of the stack is actually valid
            stacktop_copy = iframe_raw.stacktop
            if stacktop_copy == -1:
                # Frames that are currently executing have stacktop == -1.
                # Trim the stack at the depth it would be popped to before
                # executing the current exception handler, which we know is
                # a safe depth.
                stack_top_offset = stack_start_offset + wordsize * handler_depth
            else:
                stack_top_offset = localsplus_offset + wordsize * stacktop_copy
                assert stack_start_offset <= stack_top_offset <= end_offset

            frame_owner = iframe_raw.owner  # one of the FRAME_OWNED_BY_* constants

            stack_len = (stack_top_offset - stack_start_offset) // wordsize
            stack_ptr = (ctypes.py_object * stack_len).from_address(
                ctypes.addressof(iframe_raw) + stack_start_offset
            )
            assert frame.f_lasti == lasti_before

            # Extract object pointers for it. This is by far the most
            # delicate part of our routine if the frame is executing
            # on another thread, because we need to avoid taking a
            # reference to a PyObject* that's been destroyed.
            # (gc.get_referents() would be safer if it worked, but it
            # only walks the locals, and only if the frame isn't
            # executing and isn't owned by a generator -- so, useless
            # for our purposes.) It is relevant here that Python will
            # only consider switching threads before making a backward
            # jump or after making a call.
            details.stack = []
            if frame_owner != FRAME_OWNED_BY_FRAME_OBJECT:
                for i in range(stack_len):
                    # Assert that the extent of stack validity still matches
                    # what we thought before. (Note it's fine if the function
                    # has continued execution and happened to wind up in the
                    # same place, because it will have the same stack depth.)
                    #
                    # Note this also suffices to check that the frame remains
                    # pinned on the thread stack if it was before, because
                    # finishing execution would change lasti.
                    assert frame.f_lasti == lasti_before

                    try:
                        # Read the PyObject* from memory and take a reference to it,
                        # in one atomic operation
                        obj = stack_ptr[i]
                    except ValueError:
                        # ctypes raises this if a PyObject* is NULL. We'll record
                        # those as None.
                        obj = None

                    details.stack.append(obj)

            assert frame.f_lasti == lasti_before

        except AssertionError:
            if frame.f_lasti == lasti_before:
                raise
            # otherwise this was probably a concurrent modification, try again
            continue

        # we got a consistent snapshot
        lasti = lasti_before
        break
    else:
        raise RuntimeError(
            "Could not obtain a consistent stack snapshot. Probably this frame "
            "is running in another thread and is too complex for us to scan the "
            "stack before we get preempted."
        )

    # Figure out the active context managers and finally blocks, by
    # using the exception table to repeatedly simulate raising an exception
    # from the location of the previous handler.
    handlers = list(_parse_exception_table(co))
    current = lasti
    while True:
        # current + 1 is an invalid bytecode offset (the next would be
        # current + 2); but it definitely comes after any handlers
        # that start at current, and before any that start at current
        # + 2, without worrying about the rest of the tuple
        idx = bisect.bisect_left(handlers, (current + 1, 0))
        if idx == 0:
            break
        start, end, target, depth, *_ = handlers[idx - 1]
        if start <= current <= end:
            details.blocks.append(
                FrameDetails.FinallyBlock(handler=target, level=depth)
            )
            current = target
        else:
            break
    # The above loop produced blocks in inside-out order; swap to make outside-in
    details.blocks.reverse()

    return details
