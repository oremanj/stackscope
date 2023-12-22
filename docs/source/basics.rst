Extracting and inspecting stacks
================================

.. module:: stackscope

Extracting a stack
------------------

The main entry point for stackscope is the :func:`stackscope.extract` function.
It comes in several variants:

* :func:`extract` accepts a "stack item", which is anything stackscope
  knows how to turn into a series of frame objects. This might be a
  `StackSlice`, coroutine object, generator iterator, thread,
  greenlet, `trio.lowlevel.Task`, or anything else for which a library
  author (maybe you) have added support through the :func:`unwrap_stackitem`
  customization hook.

* :func:`extract_since` and :func:`extract_until` obtain a stack for
  the callees or callers (respectively) of a currently-executing
  frame. Use ``extract_since(None)`` to get the full stack of the
  thread that made the ``extract_since()`` call, like :func:`inspect.stack`.
  These are aliases for invoking :func:`extract` with a `StackSlice`.

* :func:`extract_outermost` returns the outermost `Frame` that
  :func:`extract` would return, without computing the whole stack.

.. autofunction:: extract
.. autofunction:: extract_since
.. autofunction:: extract_until
.. autofunction:: extract_outermost

.. autoclass:: StackSlice

   .. autoattribute:: outer

      The outermost frame to extract. If unspecified, start from
      `inner` (or the caller of :func:`stackscope.extract` if
      `inner` is None) and iterate outward until top-of-stack
      is reached or `limit` frames have been extracted.

   .. autoattribute:: inner

      The innermost frame to extract. If unspecified, extract all the
      currently-executing callees of `outer`, up to `limit` frames
      total.

   .. autoattribute:: limit

      The maximum number of frames to extract. If None, there is no limit.


.. autoclass:: StackItem

.. autoexception:: InspectionWarning

Working with stacks
-------------------

This section documents the `Stack` object that :func:`extract`
returns, as well as the `Frame` and `Context` objects that it refers
to. All of these are `dataclasses`. Their primary purpose is to
organize the data returned by :func:`extract`.

.. autoclass:: Stack

   .. autoattribute:: frames

      The series of frames (individual calls) in the call stack,
      from outermost/oldest to innermost/newest.

   .. autoattribute:: root

      The object that was originally passed to :func:`extract` to produce
      this stack trace, or `None` if the trace was created from a
      `StackSlice` (which doesn't carry any information beyond the frames).

   .. autoattribute:: leaf

      An object that provides additional context on what this call stack
      is doing, after you peel away all the frames.

      If this callstack comes from a generator that is yielding from an
      iterator which is not itself a generator, or comes from an
      async function that is awaiting an awaitable which is not
      itself a coroutine, then *leaf* will be that iterator or awaitable.

      Library glue may provide additional semantics for *leaf*; for
      example, the call stack of an async task that is waiting on an
      event might set *leaf* to that event.

      If there is no better option, *leaf* will be None.

   .. autoattribute:: error

      The error encountered walking the stack, if any.
      (``stackscope`` does its best to not actually raise exceptions
      out of :func:`~stackscope.extract`.)

   .. automethod:: format
   .. automethod:: format_flat
   .. automethod:: as_stdlib_summary

.. autoclass:: Frame

   .. autoattribute:: pyframe

      The Python `frame object
      <https://docs.python.org/3/reference/datamodel.html#frame-objects>`__
      that this frame describes.

   .. autoattribute:: lineno

      A line number in `pyframe`.

      This is the currently executing line number, or the line number at which
      it will resume execution if currently suspended by a ``yield`` statement or
      greenlet switch, as captured when the `Frame` object was constructed.

   .. autoattribute:: origin

      The innermost weak-referenceable thing that we looked inside to
      find this frame, or None. Frames themselves are not
      weak-referenceable, but ``extract_outermost(frame.origin).pyframe``
      will recover the original ``frame.pyframe``. For example, when
      traversing an async call stack, `origin` might be a coroutine
      object or generator iterator.

      This is exposed for use in async debuggers, which might want a way to
      get ahold of a previously-reported frame if it's still running, without
      keeping it pinned in memory if it's finished.

   .. autoattribute:: contexts

      The series of contexts (``with`` or ``async with`` blocks) that
      are active in this frame, from outermost to innermost. A context is
      considered "active" for this purpose from the point where its manager's
      ``__enter__`` or ``__aenter__`` method returns until the point where
      its manager's ``__exit__`` or ``__aexit__`` method returns.

   .. autoattribute:: hide

      If true, this frame relates to library internals that are likely
      to be more distracting than they are useful to see in a traceback.
      Analogous to the ``__tracebackhide__`` variable supported by pytest.
      Hidden frames are suppressed by default when printing stacks, but
      this can be controlled using the *show_hidden_frames* argument
      to :meth:`format`.

   .. autoattribute:: hide_line

      Limited version of `hide` which by default suppresses display of the
      executing line, but not of the function information or context managers
      associated with the frame. As with `hide`, you can force the hidden
      information to be displayed by specifying the *show_hidden_frames*
      argument to :meth:`format`.

   .. autoproperty:: filename
   .. autoproperty:: funcname
   .. autoproperty:: clsname
   .. autoproperty:: modname
   .. autoproperty:: linetext
   .. automethod:: format
   .. automethod:: as_stdlib_summary
   .. automethod:: as_stdlib_summary_with_contexts

.. autoclass:: Context

   .. autoattribute:: obj

      The object that best describes what this context is doing.
      By default, this is the context manager object (the thing with the
      ``__enter__``/``__aenter__`` and ``__exit__``/``__aexit__`` methods),
      but library glue may override it to provide something more helpful.
      For example, an ``async with trio.open_nursery():`` block will put
      the `trio.Nursery` object here instead of the context manager that
      wraps it.

   .. autoattribute:: is_async

      True for an async context manager, False for a sync context manager.

   .. autoattribute:: is_exiting

      True if this context manager is currently exiting, i.e., the next
      thing in the traceback is a call to its ``__exit__`` or ``__aexit__``.

   .. autoattribute:: varname

      The name that the result of the context manager was assigned to.
      In ``with foo() as bar:``, this is the string ``"bar"``.
      This may be an expression representing any valid assignment
      target, not just a simple identifier, although a simple identifier
      is by far the most common case. If the context manager result was
      not assigned anywhere, or if its assignment target was too complex
      for us to reconstruct, *name* will be None.

   .. autoattribute:: start_line

      The line number on which the ``with`` or ``async with`` block
      started, or ``None`` if we couldn't determine it. (In order to
      determine the corresponding filename, you need to know which
      `Frame` this `Context` is associated with.)

   .. autoattribute:: description

      A description of the context manager suitable for human-readable
      output. By default this is None, meaning we don't know how to
      do better than ``repr(obj)``, but library glue
      may augment it in some cases, such as to provide the arguments
      that were passed to a ``@contextmanager`` function.

   .. autoattribute:: inner_stack

      The call stack associated with the implementation of this context
      manager, if applicable. For a ``@contextmanager`` function, this
      will typically contain a single frame, though it might be more if
      the function uses ``yield from``. In most other cases there are
      no associated frames so *stack* will be None.

   .. autoattribute:: children

      The other context managers or child task stacks that are
      logically nested inside this one, if applicable.  For example,
      an `~contextlib.ExitStack` will have one entry here per thing
      that was pushed on the stack, and a `trio.Nursery` will have one
      entry per child task running in the nursery.

   .. autoattribute:: hide

      If true, this context manager relates to library internals that are likely
      to be more distracting than they are useful to see in a traceback.
      Analogous to the ``__tracebackhide__`` variable supported by pytest.
      Hidden context managers are suppressed by default when printing stacks,
      but this can be controlled using the *show_hidden_frames* argument
      to :meth:`format`.

   .. automethod:: format
