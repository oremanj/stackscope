import gc
import types
import pytest
from contextlib import contextmanager, ExitStack
from stackscope import _customization as cust
from stackscope._glue import add_glue_as_needed, builtin_glue_pending


def unwrap_mappingproxy(dct):
    if isinstance(dct, types.MappingProxyType):
        (dct,) = gc.get_referents(dct)
    assert hasattr(dct, "clear")
    return dct


@contextmanager
def save_restore_contents(dct):
    dct = unwrap_mappingproxy(dct)
    prev_contents = list(dct.items())
    try:
        yield
    finally:
        dct.clear()
        dct.update(prev_contents)


def each_singledispatch():
    yield cust.unwrap_stackitem
    yield cust.unwrap_context
    yield cust.elaborate_context


@pytest.fixture
def local_registry():
    add_glue_as_needed()
    with ExitStack() as stack:
        stack.enter_context(save_restore_contents(builtin_glue_pending))
        stack.enter_context(save_restore_contents(cust.elaborate_frame.registry))
        stack.enter_context(
            save_restore_contents(cust.unwrap_context_generator.registry)
        )
        for hook in each_singledispatch():
            stack.callback(hook._clear_cache)
            stack.enter_context(save_restore_contents(hook.registry))
        yield


@pytest.fixture
def trickery_disabled():
    from stackscope.lowlevel import set_trickery_enabled

    set_trickery_enabled(False)
    yield
    set_trickery_enabled(None)
