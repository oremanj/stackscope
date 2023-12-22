import sys
import threading
import types
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Optional

import pytest

import stackscope


def test_identity_dict() -> None:
    from stackscope.lowlevel import IdentityDict

    @dataclass(eq=True)
    class Cell:
        value: int

    Cell_name = Cell.__qualname__
    c1 = Cell(10)
    c2 = Cell(10)
    idict: IdentityDict[Cell, Optional[int]] = IdentityDict()
    idict.update([(c1, 1), (c2, 2)])
    rev = IdentityDict([(c2, 2), (c1, 1)])

    # equality, order preservation, repr
    assert idict == IdentityDict([(c1, 1), (c2, 2)])
    assert IdentityDict([(1, 2), (3, 4)]) == {1: 2, 3: 4}
    assert idict == rev
    assert idict != IdentityDict([(c2, 1), (c1, 2)])
    assert (
        repr(idict)
        == f"IdentityDict([({Cell_name}(value=10), 1), ({Cell_name}(value=10), 2)])"
    )
    assert len(idict) == 2
    assert list(map(id, iter(idict))) == [id(c1), id(c2)]

    # item lookup
    assert idict[c1] == rev[c1] == 1
    assert idict[c2] == rev[c2] == 2
    with pytest.raises(KeyError):
        idict[Cell(10)]
    c1.value = 100
    assert idict[c1] == 1

    # item assignment & deletion
    idict[c1] = 11
    assert idict != rev
    assert list(idict.values()) == [11, 2]
    del idict[c2]
    assert len(idict) == 1
    assert list(idict.items()) == [(Cell(100), 11)]
    assert next(iter(idict.items()))[0] is c1
    idict.clear()
    assert len(idict) == 0

    idict.setdefault(c2, 10)
    idict.setdefault(c2, 20)
    idict.setdefault(Cell(100))
    assert list(idict.items()) == [(c2, 10), (Cell(100), None)]
    assert idict.popitem() == (Cell(100), None)
    assert idict.pop("nope", 42) == 42  # type: ignore
    assert idict.pop(c2, 99) == 10
    with pytest.raises(KeyError):
        idict.pop(c2)


# This elaborate_frame implementation pretends that every frame it's registered on
# that has a local or argument named "magic_arg" is actually the runner for
# a generator called "fake_target". The generator receives the value of magic_arg
# as its argument "arg".
def simple_elaborate_frame(frame, next_inner):
    if "magic_arg" in frame.pyframe.f_locals:  # pragma: no branch

        def fake_target(arg):
            yield

        gen = fake_target(frame.pyframe.f_locals["magic_arg"])
        gen.send(None)
        return gen


def current_frame_uses_registered_elaborate():
    tb = stackscope.extract_since(sys._getframe(1))
    return tb.frames[1].funcname == "fake_target"


def test_registration_through_functools_wraps_or_partial(local_registry):
    def example(magic_arg):
        return stackscope.extract_since(sys._getframe(0))

    @wraps(example)
    def wrapper(something):
        return example(something)

    bound_example = partial(wrapper, 10)
    stackscope.elaborate_frame.register(bound_example)(simple_elaborate_frame)

    tb = bound_example()
    assert len(tb.frames) == 2
    assert [f.funcname for f in tb.frames] == ["example", "fake_target"]
    assert tb.frames[-1].pyframe.f_locals["arg"] == 10


def test_registration_through_code_object(local_registry):
    def code_example(magic_arg):
        return stackscope.extract_since(sys._getframe(0))

    stackscope.elaborate_frame.register(code_example.__code__, simple_elaborate_frame)
    tb = code_example(100)
    assert [f.funcname for f in tb.frames] == ["code_example", "fake_target"]


def test_registration_through_unsupported():
    with pytest.raises(TypeError, match="extract a code object"):
        stackscope.customize(42)


def test_registration_through_method(local_registry):
    class C:
        @stackscope.customize(elaborate=simple_elaborate_frame)
        def instance(self, magic_arg):
            assert current_frame_uses_registered_elaborate()
            return "instance", self

        @stackscope.customize(elaborate=simple_elaborate_frame)
        @staticmethod
        def static(magic_arg):
            assert current_frame_uses_registered_elaborate()
            return "static", None

        @stackscope.customize(elaborate=simple_elaborate_frame)
        @classmethod
        def class_(cls, magic_arg):
            assert current_frame_uses_registered_elaborate()
            return "class", cls

    c = C()
    assert c.instance(1) == C.instance(c, 2) == ("instance", c)
    assert c.class_(1) == C.class_(2) == ("class", C)
    assert c.static(1) == C.static(2) == ("static", None)


def test_registration_through_nested(local_registry):
    def outer_fn(magic_arg):
        def middle_fn():
            def inner_fn():
                assert magic_arg == 1
                return current_frame_uses_registered_elaborate()

            assert magic_arg == 1
            assert current_frame_uses_registered_elaborate()
            return inner_fn

        assert not current_frame_uses_registered_elaborate()
        return middle_fn

    stackscope.customize(outer_fn, "middle_fn", elaborate=simple_elaborate_frame)
    with pytest.raises(ValueError, match="function or class named"):
        stackscope.customize(outer_fn, "magic_arg")
    with pytest.raises(ValueError, match="function or class named"):
        stackscope.customize(outer_fn, "nope")

    middle_fn = outer_fn(1)
    inner_fn = middle_fn()
    assert not inner_fn()
    stackscope.customize(
        outer_fn, "middle_fn", "inner_fn", elaborate=simple_elaborate_frame
    )
    assert inner_fn()


def test_module_beats_local(local_registry, monkeypatch):
    def subject_fn():
        return not stackscope.extract_since(sys._getframe(0)).frames[0].hide

    record = []

    def builtin_glue():  # pragma: no cover
        record.append("builtin")
        stackscope.customize(subject_fn, hide=True)

    def module_glue():
        record.append("module")
        stackscope.customize(subject_fn, hide=True)

    assert subject_fn()

    monkeypatch.setitem(
        stackscope._glue.builtin_glue_pending, "stackscope_test_mod", builtin_glue
    )
    module = types.ModuleType("stackscope_test_mod")
    module._stackscope_install_glue_ = module_glue
    monkeypatch.setitem(sys.modules, "stackscope_test_mod", module)

    assert not subject_fn()
    assert record == ["module"]


def test_glue_error(local_registry, monkeypatch):
    def glue_whoops():
        raise ValueError("guess not")

    monkeypatch.setitem(stackscope._glue.builtin_glue_pending, "test_one", glue_whoops)
    monkeypatch.setitem(sys.modules, "test_one", None)

    with pytest.warns(
        RuntimeWarning,
        match="Failed to initialize stackscope-builtin glue for test_one",
    ):
        stackscope._glue.add_glue_as_needed()


def test_prune():
    repr(stackscope.PRUNE)
