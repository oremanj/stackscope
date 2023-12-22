Customizing stackscope for your library
=======================================

.. currentmodule:: stackscope

.. _customizing:

`stackscope` contains several customization hooks that allow it to be
adapted to provide good stack traces for context managers, awaitables,
and control-flow primitives that it doesn't natively know anything about.
These are implemented either as `@functools.singledispatch <functools.singledispatch>`
functions (which dispatch to a different implementation depending on the type of
their first argument) or as `@stackscope.lowlevel.code_dispatch
<stackscope.lowlevel.code_dispatch>` functions (which dispatch to a different
implementation depending on the identity of the Python code object associated
with their first argument). Implementations of these hooks that support a
particular library are referred to as stackscope "glue" for that library.

If you're working with a library that could be better-supported by stackscope,
you have two options for implementing that support:

* If you maintain the library that the customizations are intended to
  support, then define a function named ``_stackscope_install_glue_``
  at top level in any of your library's module(s), which takes no
  arguments and returns None. The body of the function should register
  customization hooks appropriate to your library, using the
  stackscope APIs described in the rest of this section. As long as you
  only write ``import stackscope`` inside the body of the glue installation
  function, this won't require that users of your library install stackscope,
  but they will benefit from your glue if they do.

* If you're contributing glue for a library you don't maintain, you
  can put the glue in stackscope instead. We have glue for several
  modules and a system that avoids registering it unless the module
  has been imported. See ``stackscope/_glue.py``, and feel free to
  submit a PR.

If the same module has glue implemented using both of these methods,
then the glue provided by the module will be used; the glue
shipped with stackscope is ignored. This allows for a module's glue to
start out being shipped with stackscope and later "graduate" to being
maintained upstream.

Overview of customization hooks
-------------------------------

In order to understand the available customization hooks, it's helpful
to know how stackscope's stack extraction works internally. There are
separate systems for frames and for context managers, and they can
interact with each other recursively.

Frames
~~~~~~

There are two hooks that are relevant in determining the frames in the
returned stack: :func:`unwrap_stackitem` and :func:`elaborate_frame`.

:func:`unwrap_stackitem` receives a "stack item", which may have been
passed to :func:`stackscope.extract` or returned from an
:func:`elaborate_frame` or :func:`unwrap_stackitem` hook.  It is
dispatched based on the Python type of the stack item. Abstractly
speaking, a stack item should be something that logically has Python
frame objects associated with it, and the job of
:func:`unwrap_stackitem` is to turn it into something that is closer
to those frame objects. :func:`unwrap_stackitem` may return a single
stack item, a sequence thereof, or None if it can't do any unwrapping.
Each returned stack item is recursively unwrapped in the same way until
no further unwrapping can be done. The resulting frame objects become the
basis for the `Stack.frames` and any non-frame-objects go in `Stack.leaf`.

Example: built-in glue for unwrapping a generator iterator::

    @unwrap_stackitem.register(types.GeneratorType)
    def unwrap_geniter(gen: types.GeneratorType[Any, Any, Any]) -> Any:
        if gen.gi_running:
            return StackSlice(outer=gen.gi_frame)
        return (gen.gi_frame, gen.gi_yieldfrom)

:func:`elaborate_frame` operates after unwrapping is complete, and is
dispatched based on the *code object identity* of the executing frame,
so it's useful for customizations that are specific to a particular
function. (You get approximately one code object per source file
location of a function definition.) It receives the frame it is
elaborating as well as the next inner frame-or-leaf for context. It
can customize the `Frame` object, such as by setting the `Frame.hide`
attribute or modifying the `Frame.contexts`. It can also redirect the
rest of the stack extraction, by returning a stack item or sequence of
stack items that will be used in place of the otherwise-next frame and
all of its callees. (If it returns a sequence that ends with the
otherwise-next frame, then the preceding elements are inserted before
the rest of the stack trace rather than replacing it.)

Example: glue for elaborating :func:`greenback.await_`::

    @elaborate_frame.register(greenback.await_)
    def elaborate_greenback_await(
        frame: Frame, next_inner: object
    ) -> object:
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

Contexts
~~~~~~~~

There are three hooks relevant in determining the context
managers in the returned stack: :func:`unwrap_context`,
:func:`unwrap_context_generator`, and
:func:`elaborate_context`. They are less complex than the frame hooks
since they only operate on one context manager at a time.

:func:`unwrap_context` handles context managers that wrap other context
managers. It receives the "outer" context manager and returns the "inner"
one, or returns None to indicate no further unwrapping is needed. It is
dispatched based on the type of the outer context manager. When using
:func:`unwrap_context`, the "outer" context manager is totally lost; it
appears to stackscope, and to clients of :func:`extract`, as though only
the "inner" one ever existed. If you prefer to include both, you can
assign to the `Context.children` attribute in :func:`elaborate_context`
instead.

Example: glue for elaborating `greenback.async_context` objects::

    @unwrap_context.register(greenback.async_context)
    def unwrap_greenback_async_context(manager: Any) -> Any:
        return manager._cm

:func:`unwrap_context_generator` is a specialization of
:func:`unwrap_context` for generator-based context managers
(`@contextmanager <contextlib.contextmanager>` and
`@asynccontextmanager <contextlib.asynccontextmanager>`).  It works
exactly like :func:`unwrap_context` except that it takes as its first
argument the generator's `Stack` rather than the context manager
object. Like :func:`elaborate_frame`, :func:`unwrap_context_generator`
is dispatched based on the code object identity of the function that
implements the context manager, so you can unwrap different generator-based
context managers in different ways even though their context manager
objects all have the same type.

:func:`elaborate_context` is called once for each context manager
before trying to unwrap it, and again after each successful
unwrapping.  It is dispatched based on the context manager type and
fills in attributes of the `Context` object, such as
`Context.description`, `Context.children`, and
`Context.inner_stack`. These `Context` attributes might be filled out
using calls to :func:`fill_context` or :func:`extract`, which will
recursively execute context/frame hooks as
needed. :func:`elaborate_context` can also change the `Context.obj`
which may influence further unwrapping attempts.

Example: glue for handling generator-based `@contextmanager <contextlib.contextmanager>`\s::

    @elaborate_context.register(contextlib._GeneratorContextManagerBase)
    def elaborate_generatorbased_contextmanager(mgr: Any, context: Context) -> None:
        # Don't descend into @contextmanager frames if the context manager
        # is currently exiting, since we'll see them later in the traceback
        # anyway
        if not context.is_exiting:
            context.inner_stack = stackscope.extract(mgr.gen)
        context.description = f"{mgr.gen.__qualname__}(...)"


Utilities for use in customization hooks
----------------------------------------

.. autofunction:: extract_child
.. autofunction:: fill_context


Customization hooks reference
-----------------------------

.. autofunction:: unwrap_stackitem
.. autofunction:: yields_frames
.. autofunction:: elaborate_frame
.. data:: PRUNE

   Sentinel value which may be returned by :func:`elaborate_frame` to indicate
   that the remainder of the stack (containing the direct and indirect callees
   of the frame being elaborated) should not be included in the extracted
   `stackscope.Stack`.

.. autofunction:: customize
.. autofunction:: unwrap_context
.. autofunction:: unwrap_context_generator
.. autofunction:: elaborate_context
