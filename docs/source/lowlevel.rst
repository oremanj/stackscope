Low-level introspection tools and utilities
===========================================

.. module:: stackscope.lowlevel

In order to implement its context manager analysis, `stackscope` includes some
fairly arcane bits of Python introspection lore, including bytecode analysis
and inspection of raw frame objects using `ctypes`. The `stackscope.lowlevel`
module provides direct access to these lower-level bits, in case you want to
use them for a different purpose. It also collects a few of the utilities
used to track code objects for the :func:`stackscope.elaborate_frame`
customization hook, in the hope that they might find some broader use.

These are supported public APIs just like the rest of the library;
their membership in the `~stackscope.lowlevel` module is primarily
because the problems they solve aren't directly relevant to the
typical end user.

Extracting context managers
---------------------------

.. autofunction:: contexts_active_in_frame
.. autofunction:: set_trickery_enabled

Frame analysis pieces
---------------------

.. autofunction:: analyze_with_blocks
.. autofunction:: inspect_frame
.. autoclass:: FrameDetails

   .. autoclass:: stackscope.lowlevel::FrameDetails.FinallyBlock

      .. autoattribute:: stackscope.lowlevel::FrameDetails.FinallyBlock.handler

         The bytecode offset to which control will be transferred if an
         exception is raised.

      .. autoattribute:: stackscope.lowlevel::FrameDetails.FinallyBlock.level

         The value stack depth at which the exception handler begins execution.

   .. autoattribute:: blocks

      Currently active exception-catching contexts in this frame
      (includes context managers too) in order from outermost to
      innermost

   .. autoattribute:: stack

      All values on this frame's evaluation stack. This may be truncated at the
      position where an exception would unwind to, if the frame is currently
      executing and we don't know its actual stack depth. Null pointers are
      rendered as None and local variables (including cellvars/freevars) are
      not included.

.. autofunction:: currently_exiting_context
.. autoclass:: ExitingContext

   .. autoattribute:: is_async

      True for an async context manager, False for a sync context manager.

   .. autoattribute:: cleanup_offset

      The bytecode offset of the WITH_CLEANUP_START or WITH_EXCEPT_START instruction
      that begins the exception handler associated with this context manager.

.. autofunction:: describe_assignment_target

Code-object-based dispatch utilities
------------------------------------

.. autofunction:: get_code

.. autofunction:: code_dispatch

.. autoclass:: IdentityDict
   :show-inheritance:
