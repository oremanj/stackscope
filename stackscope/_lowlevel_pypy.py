import ctypes
import dis
import gc
import sys
from types import FrameType
from typing import Dict, Iterator, List, Optional, Sequence, Type, TYPE_CHECKING, cast
from ._lowlevel import FrameDetails


# Every interpreter-level type in pypy has a description, index, and id.
# The description is something human-readable, possibly with spaces.
# The id is a unique half-word value (2 bytes on 32-bit, 4 bytes on 64-bit)
# stored in the bottom half of the first word in the instance representation;
# it indicates the offset of the type-info structure in a global array.
# The index is a denser unique value used in some other places.
#
# More details on pypy object representation:
# https://github.com/oremanj/asynctb/issues/1

_pypy_type_desc_from_index: List[str] = []
_pypy_type_index_from_id: Dict[int, int] = {}


if TYPE_CHECKING:
    # typeshed doesn't include the pypy-specific gc methods
    from . import _pypy_gc_stubs as pgc
else:
    pgc = gc


def _fill_pypy_typemaps() -> None:
    assert sys.implementation.name == "pypy"
    import zlib

    # The first few lines of get_typeids_z(), after decompression, look like:
    # member0    ?
    # member1    GcStruct JITFRAME { jf_frame_info, jf_descr, jf_force_descr, [...] }
    # member2    GcStruct rpy_string { hash, chars }
    # member3    GcStruct rpy_unicode { hash, chars }
    for line in zlib.decompress(pgc.get_typeids_z()).decode("ascii").splitlines():
        memberNNN, rest = line.split(None, 1)
        header, brace, fields = rest.partition(" { ")
        _pypy_type_desc_from_index.append(header)

    for idx, typeid in enumerate(pgc.get_typeids_list()):
        _pypy_type_index_from_id[typeid] = idx


if sys.implementation.name == "pypy":  # pragma: no branch
    _fill_pypy_typemaps()


def _pypy_typename(obj: object) -> str:
    """Return the pypy interpreter-level type name of the type of *obj*.

    *obj* may be an ordinary Python object, or may be a gc.GcRef to inspect
    something that is not manipulatable at app-level.
    """
    return _pypy_type_desc_from_index[pgc.get_rpy_type_index(obj)]


def _pypy_typename_from_first_word(first_word: int) -> str:
    """Return the pypy interpreter-level type name of the type of the instance
    whose first word in memory (8 bytes on 64-bit, 4 bytes on 32-bit) has the
    value *first_word*.
    """
    if sys.maxsize > 2**32:
        mask = 0xFFFFFFFF
    else:
        mask = 0xFFFF
    return _pypy_type_desc_from_index[_pypy_type_index_from_id[first_word & mask]]


def inspect_frame(frame: FrameType) -> FrameDetails:
    assert sys.implementation.name == "pypy"

    # Somewhere in the list of immediate referents of the frame is its
    # code object.
    frame_refs = pgc.get_rpy_referents(frame)
    (code_idx,) = [idx for idx, ref in enumerate(frame_refs) if ref is frame.f_code]

    # The two referents immediately before the code object are
    # the last entry in the block list, followed by the value stack.
    # These are interp-level objects so we see them as opaque GcRefs.
    # We locate them by reference to the code object because the
    # earlier references might or might not be present (e.g., one depends
    # on whether the frame's f_locals have been accessed yet or not).
    assert code_idx >= 1
    valuestack_ref = frame_refs[code_idx - 1]
    assert isinstance(valuestack_ref, pgc.GcRef)

    lastblock_ref: Optional[pgc.GcRef] = None
    if code_idx >= 2:  # pragma: no branch
        # Rationale for "no branch": there's no known way to get a
        # frame with neither a generator weakref nor an f_back, which
        # is what would be required to have the code object as the 2nd
        # referent. If we do wind up in that case, then there's nothing
        # on the block stack and the first referent is the value stack.
        candidate = frame_refs[code_idx - 2]
        typename = _pypy_typename(candidate)
        if "Block" not in typename and "SysExcInfoRestorer" not in typename:
            # There are no blocks active in this frame. lastblock was
            # skipped when getting referents because it's null, so the
            # previous field (generator weakref or f_back) bled through.
            assert _pypy_typename(
                candidate
            ) == "GcStruct weakref" or "Frame" in _pypy_typename(candidate)
        else:
            assert isinstance(candidate, pgc.GcRef)
            lastblock_ref = candidate

    # The value stack's referents are everything on the value stack.
    # Unfortunately we can't rely on the indices here because 'del x'
    # leaves a null (not None) that will be skipped. We'll fill them
    # in from ctypes later. Note that this includes locals/cellvars/
    # freevars (at the start, in that order).
    valuestack = pgc.get_rpy_referents(valuestack_ref)

    # The block list is a linked list in PyPy, unlike in CPython where
    # it's an array. The head of the list is the newest block.
    # Iterate through and unroll it into a list of GcRefs to blocks.
    blocks: List[pgc.GcRef] = []
    if lastblock_ref is not None:
        blocks.append(lastblock_ref)
        while True:
            assert len(blocks) < 100
            more = pgc.get_rpy_referents(blocks[-1])
            if not more:
                break
            for ref in more:
                assert isinstance(ref, pgc.GcRef)
                blocks.append(ref)
        assert all(
            "Block" in name or "SysExcInfo" in name
            for blk in blocks
            for name in [_pypy_typename(blk)]
        )
        # Reverse so the oldest block is at the beginning
        blocks = blocks[::-1]
        # Remove those that aren't FinallyBlocks -- those are the
        # only ones we care about (used for context managers too)
        blocks = [blk for blk in blocks if "FinallyBlock" in _pypy_typename(blk)]

    # With the default (incminimark) GC, id() of a young object
    # returns the address it will live in after the next minor
    # collection, but doesn't move it there. Perform a minor
    # collection so we can look at the object representations using
    # ctypes below.
    #
    # Annoyingly, there's no way to directly perform a minor
    # collection.  gc.collect() performs a major collection, which is
    # quite slow to do on every frame traversal. gc.collect_step()
    # usually does at least a minor collection, but for the last step
    # of each major collection it runs finalizers instead. So we try
    # one collect_step(), and if it was the last step in a major
    # collection then we do another one.
    if pgc.collect_step().major_is_done:
        pgc.collect_step()

    # We cast ctypes.pointer[ctypes.c_ulong] to Sequence[int] because
    # typeshed incorrectly thinks indexing the pointer produces a c_ulong.
    # In reality it produces an int.
    def pointer(val: ctypes.c_ulong) -> Sequence[int]:
        return cast(Sequence[int], ctypes.pointer(val))

    def unwrap_gcref(ref: pgc.GcRef) -> Sequence[int]:
        ref_p = pointer(ctypes.c_ulong.from_address(id(ref)))
        assert "W_GcRef" in _pypy_typename_from_first_word(ref_p[0])
        return pointer(ctypes.c_ulong.from_address(ref_p[1]))

    # Fill in nulls in the value stack. This requires inspecting the
    # memory that backs the list object. An RPython list is two words
    # (typeid, length) followed by one word per element.
    def build_full_stack(refs: Sequence[object]) -> List[object]:
        assert isinstance(valuestack_ref, pgc.GcRef)
        stackdata_p = unwrap_gcref(valuestack_ref)
        assert _pypy_typename_from_first_word(stackdata_p[0]) in (
            "GcArray of * GcStruct object",
            "GcArray of * GCREF (gcopaque) ",
        )
        ref_iter = iter(refs)
        result: List[object] = []
        for idx in range(stackdata_p[1]):
            if stackdata_p[2 + idx] == 0:
                result.append(None)
            else:
                try:
                    result.append(next(ref_iter))
                except StopIteration:  # pragma: no cover
                    # The value stack has more entries than GC knows about.
                    # I haven't been able to produce an example of this,
                    # and I don't think it should be possible, but there's
                    # no harm in trying to continue on -- every object we're
                    # returning is a real Python object obtained through
                    # non-sketchy means, so even if our assumptions are
                    # wrong we shouldn't segfault.
                    break
        return result

    details = FrameDetails(stack=list(build_full_stack(valuestack)))
    for block_ref in blocks:
        block_p = unwrap_gcref(block_ref)
        assert _pypy_typename_from_first_word(block_p[0]) == (
            "GcStruct pypy.interpreter.pyopcode.FinallyBlock"
        )
        details.blocks.append(
            FrameDetails.FinallyBlock(handler=block_p[1], level=block_p[3])
        )
    return details
