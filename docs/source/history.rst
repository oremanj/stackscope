Release history
===============

.. currentmodule:: stackscope

.. towncrier release notes start

stackscope 0.2.1 (2024-02-02)
-----------------------------

Bugfixes
~~~~~~~~

- Fixed inspection of async context managers that contain a ``CLEANUP_THROW``
  bytecode instruction in their ``__aenter__`` sequence. This can appear on 3.12+
  if you write an async context manager inside an ``except`` or ``finally`` block,
  and would previously produce an inspection warning. (`#11 <https://github.com/oremanj/stackscope/issues/11>`__)
- The first invocation of :func:`stackscope.extract` no longer leaves a
  partially-exhausted async generator object to be garbage collected,
  which previously could confuse async generator finalization hooks. (`#12 <https://github.com/oremanj/stackscope/issues/12>`__)


stackscope 0.2.0 (2023-12-22)
-----------------------------

With this release, stackscope can print full Trio task trees out-of-the-box.
Try ``print(stackscope.extract(trio.lowlevel.current_root_task(),
recurse_child_tasks=True))``.

Backwards-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The :func:`unwrap_context` hook now
  accepts an additional `Context` argument. This saves on duplicated effort
  between :func:`elaborate_context` and :func:`unwrap_context`, avoiding
  exponential time complexity in some pathological cases.
- Removed support for Python 3.7.

User-facing improvements to core logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for representing child tasks in structured concurrency libraries,
  by allowing `Context.children` to contain `Stack`\s in addition to the
  existing support for child `Context`\s. By default, the child tasks will
  not have their frames filled out, but you can override this with the new
  *recurse_child_tasks* parameter to :func:`extract`.
  (`#9 <https://github.com/oremanj/stackscope/issues/9>`__)
- Added `Frame.hide_line` and `Context.hide` attributes for more precise
  control of output.
- Added a new attribute `Stack.root` which preserves the original "stack item"
  object that was passed to :func:`extract`. For stacks generated from async
  child tasks, this will be the ``Task`` object.
- Added support for Python 3.12.

Library support ("glue") improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- stackscope can now trace seamlessly across Trio/thread boundaries when
  extracting a stack that includes calls to :func:`trio.to_thread.run_sync`
  and/or :func:`trio.from_thread.run`. The functions running in the
  cross-thread child will appear in the same way that they would if they
  had been called directly without a thread transition.
  (`#8 <https://github.com/oremanj/stackscope/issues/8>`__)
- Added glue to support ``pytest-trio``.
  (`#4 <https://github.com/oremanj/stackscope/issues/4>`__)
- Updated Trio glue to support unwrapping `trio.lowlevel.Task`\s and filling
  in the child tasks of a `trio.Nursery`.

Improvements for glue developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A library can now ship its own :ref:`stackscope customizations <customizing>`
  without requiring that all of its users install stackscope. Any module may
  define a function called ``_stackscope_install_glue_()``, which stackscope will
  call when it is first used to extract a stack trace after the module has been
  imported. (`#7 <https://github.com/oremanj/stackscope/issues/7>`__)
- Added :func:`unwrap_context_generator` hook for more specific customization
  of generator-based context managers.
- Modified the :func:`elaborate_frame` hook to be able to return a sequence
  of stack items rather than just a single one. This permits more expressive
  augmentation rules, such as inserting elements into the stack trace without
  removing what would've been there if the hook were not present.
- Added a new function :func:`extract_child` for use in customization hooks.
  It is like :func:`extract` except that it reuses the options that were
  specified for the outer :func:`extract` call, and contains some additional
  logic to prune child task frames if the outer :func:`extract` didn't ask
  for them.
- :func:`elaborate_frame` now runs after `Frame.contexts` is populated,
  so it has the chance to modify the detected context managers.

stackscope 0.1.0 (2023-04-12)
-----------------------------

Initial release.
