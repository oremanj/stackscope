from __future__ import annotations

# Basic type definitions used in the interface of the stackscope library.

import linecache
import traceback
import types
import weakref
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    ContextManager,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
    cast,
)


__all__ = ["Stack", "Frame", "Context", "StackSlice", "StackItem"]


# Alias used in type hints for increased expressiveness.
# StackItem is anything that can be passed to extract().
if TYPE_CHECKING:
    StackItem = object  # so mypy doesn't complain
else:

    class StackItem:  # so it shows as StackItem in sphinx docs, etc
        """Placeholder used in type hints to mean "anything you can pass to
        :func:`extract`."
        """


@dataclass
class FormatOptions:
    ascii_only: bool
    show_contexts: bool
    show_hidden_frames: bool


class Formattable:
    def __str__(self) -> str:
        return "".join(self.format())

    def format(
        self,
        *,
        ascii_only: bool = False,
        show_contexts: bool = True,
        show_hidden_frames: bool = False,
    ) -> List[str]:
        """Return a list of newline-terminated strings describing
        this object, which may be printed for human consumption.
        ``str(obj)`` is equivalent to ``"".join(obj.format())``.

        Args:
          ascii_only: Use only ASCII characters in the output.
              By default, Unicode line-drawing characters are used.
          show_contexts: Include information about context managers in the output.
              This is the default; pass False for a shorter stack trace that
              only includes frames in the main series.
          show_hidden_frames: Include frames in the output even if they are
              marked as hidden. By default, hidden frames will be suppressed.
              See `Frame.hide` for more details.
        """
        return self._format(
            FormatOptions(
                ascii_only=ascii_only,
                show_contexts=show_contexts,
                show_hidden_frames=show_hidden_frames,
            )
        )

    @abstractmethod
    def _format(self, opts: FormatOptions) -> List[str]:
        raise NotImplementedError


@dataclass
class Stack(Formattable):
    """Representation of a generalized call stack.

    In addition to the attributes described below, you can treat a `Stack`
    like an iterable over its `frames`, and its ``len()`` is the number
    of frames it contains.
    """

    root: object
    frames: Sequence[Frame]
    leaf: object = None
    error: Optional[Exception] = None

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def _format_header(self) -> str:
        if self.root is not None:
            return f"stackscope.Stack of {self.root!r} (most recent call last):\n"
        else:
            return "stackscope.Stack (most recent call last):\n"

    def _format(self, opts: FormatOptions) -> List[str]:
        start_frame = "+ " if opts.ascii_only else "╠ "
        continue_frame = "| " if opts.ascii_only else "║ "
        start_leaf = "+ " if opts.ascii_only else "╚ "

        lines = [self._format_header()]
        for frame in self.frames:
            if frame.hide and not opts.show_hidden_frames:
                continue
            for idx, line in enumerate(frame._format(opts)):
                marker = start_frame if idx == 0 else continue_frame
                lines.append(marker + line)
        if self.leaf is not None:
            lines.append(f"{start_leaf}{self.leaf!r}\n")
        if self.error is not None:
            lines.extend(self._format_error())
        return lines

    def format_flat(self, *, show_contexts: bool = False) -> List[str]:
        """Return a list of newline-terminated strings providing a flattened
        representation of this stack.

        This will be formatted similarly to a standard Python
        traceback, such as might be produced if an exception were
        raised at the point where the stack was extracted.
        Context manager information is not included by default, but can be
        requested using the *show_contexts* parameter.
        """
        lines = [self._format_header()]
        if self.frames:
            lines.extend(self.as_stdlib_summary(show_contexts=show_contexts).format())
        if self.leaf is not None:
            lines.append(f"  Target of innermost frame: {self.leaf!r}\n")
        if self.error is not None:
            lines.extend(self._format_error())
        return lines

    def _format_error(self) -> Iterator[str]:
        assert self.error is not None
        yield "  Error while extracting stack:\n"
        for line in traceback.format_exception(
            type(self.error), self.error, self.error.__traceback__
        ):
            if line != "Traceback (most recent call last):\n":
                for subline in line.splitlines(True):
                    yield "  " + subline

    def as_stdlib_summary(
        self,
        *,
        show_contexts: bool = False,
        show_hidden_frames: bool = False,
        capture_locals: bool = False,
    ) -> traceback.StackSummary:
        """Return a representation of this stack as a standard
        `traceback.StackSummary`. Unlike the `Stack` object, a
        `~traceback.StackSummary` can be pickled and will not keep
        frames alive, at the expense of some loss of information.

        If *show_contexts* is True, then additional frame summaries
        will be emitted describing the context managers active in each
        frame. See the documentation of
        :meth:`Frame.as_stdlib_summary_with_contexts` for details.

        By default, hidden frames (`Frame.hide`) are not included in
        the output. You can use the *show_hidden_frames* parameter to
        override this.

        *capture_locals* is passed through to the
        :meth:`Frame.as_stdlib_summary` calls for each stack frame;
        see that method's documentation for details on its semantics.
        """
        return traceback.StackSummary.from_list(
            self._frame_summaries(show_contexts, show_hidden_frames, capture_locals)
        )

    def _frame_summaries(
        self, show_contexts: bool, show_hidden_frames: bool, capture_locals: bool
    ) -> Iterator[traceback.FrameSummary]:
        for frame in self.frames:
            if frame.hide and not show_hidden_frames:
                continue
            if show_contexts:
                yield from frame.as_stdlib_summary_with_contexts(
                    show_hidden_frames=show_hidden_frames, capture_locals=capture_locals
                )
            else:
                yield frame.as_stdlib_summary(capture_locals=capture_locals)


@dataclass
class Frame(Formattable):
    """Representation of one call frame within a generalized call stack."""

    pyframe: types.FrameType
    lineno: int = field(default=-1)
    origin: Optional[StackItem] = None
    contexts: Sequence[Context] = ()
    hide: bool = False
    hide_line: bool = False

    def __post_init__(self) -> None:
        if self.lineno == -1:
            self.lineno = self.pyframe.f_lineno

    @property
    def filename(self) -> str:
        """The filename of the Python file from which the code executing in
        this frame was imported."""
        return self.pyframe.f_code.co_filename

    @property
    def funcname(self) -> str:
        """The name of the function executing in this frame."""
        return self.pyframe.f_code.co_name

    @property
    def clsname(self) -> Optional[str]:
        """The name of the class that contains the function executing in this frame,
        or None if we couldn't determine one.

        This is determined heuristically, based on the executing function having a
        first argument named ``self`` or ``cls``, so it can be fooled.
        """
        code = self.pyframe.f_code
        if code.co_argcount == 0:
            return None
        try:
            if code.co_varnames[0] == "self":
                self_type = type(self.pyframe.f_locals["self"])
            elif code.co_varnames[0] == "cls":
                self_type = self.pyframe.f_locals["cls"]
            else:
                return None
        except Exception:
            return None
        return getattr(self_type, "__qualname__", None) or getattr(
            self_type, "__name__", None
        )

    @property
    def modname(self) -> Optional[str]:
        """The name of the module that contains the function executing in this frame,
        or None if we couldn't determine one.

        This is looked up using the ``__name__`` attribute of the frame's globals
        namespace. It can be fooled, but usually won't be. Another option, which
        is possibly more reliable but definitely much slower, would be to iterate
        through `sys.modules` looking for a module whose ``__file__`` matches
        this frame's `filename`.
        """
        try:
            return cast(str, self.pyframe.f_globals["__name__"])
        except Exception:
            return None

    @property
    def linetext(self) -> str:
        """The text of the line of source code that this stack entry
        describes. The result has leading and trailing whitespace
        stripped, and does not end in a newline.
        """
        if self.lineno == 0 or self.hide_line:
            return ""
        return linecache.getline(
            self.filename, self.lineno, self.pyframe.f_globals
        ).strip()

    def _format(self, opts: FormatOptions) -> List[str]:
        start_context = ". " if opts.ascii_only else "├ "
        continue_context = "  " if opts.ascii_only else "│ "
        start_child_context = "  " if opts.ascii_only else "├─"
        child_context_indicator = ". " if opts.ascii_only else "─ "
        start_code = "` " if opts.ascii_only else "└ "

        function = self.funcname
        clsname = self.clsname
        if clsname is not None:
            function = f"{clsname}.{function}"

        lines = [
            f"{function} in {self.modname or 'unknown module'} "
            f"at {self.filename}:{self.lineno}\n"
        ]
        if opts.show_contexts:
            for context in self.contexts:
                for idx, line in enumerate(context._format(opts, self)):
                    if idx == 0:
                        lines.append(start_context + line)
                    elif line.startswith(child_context_indicator):
                        lines.append(start_child_context + line)
                    else:
                        lines.append(continue_context + line)
        # If the last context manager is currently exiting, then don't
        # include an additional traceback entry for the frame it's
        # running in; it would either be the same 'with' statement we
        # just yielded (3.10+) or unhelpfully the last line of the
        # with block (earlier versions)
        if not (self.contexts and self.contexts[-1].is_exiting):
            linetext = self.linetext
            if linetext:
                lines.append(start_code + linetext + "\n")
        return lines

    def as_stdlib_summary(
        self, *, capture_locals: bool = False
    ) -> traceback.FrameSummary:
        """Return a representation of this frame entry as a standard
        `traceback.FrameSummary` object. Unlike the `Frame` object, a
        `~traceback.FrameSummary` can be pickled and will not keep frames alive,
        at the expense of some loss of information.

        If *capture_locals* is True, then the returned
        `~traceback.FrameSummary` will contain the stringified object
        representations of local variables in the frame, just like
        passing ``capture_locals=True`` to
        :meth:`traceback.StackSummary.extract`.
        """
        if capture_locals:
            save_locals = {
                name: repr(value) for name, value in self.pyframe.f_locals.items()
            }
        else:
            save_locals = None
        return traceback.FrameSummary(
            self.filename,
            self.lineno,
            self.funcname,
            locals=save_locals,
        )

    def as_stdlib_summary_with_contexts(
        self, *, show_hidden_frames: bool = False, capture_locals: bool = False
    ) -> Iterator[traceback.FrameSummary]:
        """Return a representation of this frame and its context managers as
        a series of standard `traceback.FrameSummary` objects.

        The last yielded `~traceback.FrameSummary` matches what
        :meth:`as_stdlib_summary` would return. Before that, one or
        more `~traceback.FrameSummary` objects will be yielded for
        each of the active `contexts` in this frame. Each context will
        get one `~traceback.FrameSummary` introducing it (pointing to
        the start of the ``with`` or ``async with`` block), followed
        by zero or more frames containing any relevant substructure,
        such as elements in an `~contextlib.ExitStack` or nested
        context managers within a `@contextmanager
        <contextlib.contextmanager>` function.  The order of
        `~traceback.FrameSummary` objects is intended to hew as
        closely as possible to the (reverse) path that an exception
        would take if it were to propagate up the call stack.
        That is, the result of :meth:`as_stdlib_summary_with_contexts`
        should ideally look pretty similar to what you would see when
        printing out a traceback after an exception.

        A `~traceback.FrameSummary` that introduces a context will
        append some additional information (the type of the context
        manager and the name that its result was assigned to) to the
        function name in the returned object, in parentheses after a
        space. This results in reasonable output from
        :meth:`traceback.StackSummary.format`.

        By default, hidden frames (`Frame.hide`) encountered during
        context manager traversal are not included in the output. You
        can use the *show_hidden_frames* parameter to override this.
        The frame on which you called :meth:`as_stdlib_summary_with_contexts`
        will be included unconditionally.

        If *capture_locals* is True, then the local ``repr``\\s
        will be included in each `~traceback.FrameSummary`,
        as with :meth:`as_stdlib_summary`. Frame summaries that
        introduce a context will include the stringified context
        manager object as a fictitious local called ``"<context manager>"``.

        """
        for context in self.contexts:
            yield from context._frame_summaries(
                self, show_hidden_frames, capture_locals
            )
        # If the last context manager is currently exiting, then don't
        # include an additional traceback entry for the frame it's
        # running in; it would either be the same 'with' statement we
        # just yielded (3.10+) or unhelpfully the last line of the
        # with block (earlier versions)
        if not (self.contexts and self.contexts[-1].is_exiting):
            yield self.as_stdlib_summary(capture_locals=capture_locals)


@dataclass
class Context(Formattable):
    """Information about a context manager active within a frame."""

    obj: object
    is_async: bool
    is_exiting: bool = False
    varname: Optional[str] = None
    start_line: Optional[int] = None
    description: Optional[str] = None
    inner_stack: Optional[Stack] = None
    children: Sequence[Context | Stack] = ()
    hide: bool = False

    def _name_and_type(self) -> str:
        if self.obj is not None:
            typename = type(self.obj).__name__
            varname = self.varname or "_"
            return f"{varname}: {typename}"
        elif self.varname is not None:
            return f"{self.varname}"
        else:
            return ""

    def _frame_summaries(
        self,
        parent: Frame,
        show_hidden_frames: bool,
        capture_locals: bool,
        override_line: Optional[str] = None,
    ) -> Iterator[traceback.FrameSummary]:
        if self.hide and not show_hidden_frames:
            return
        if capture_locals:
            save_locals = {"<context manager>": self.description or repr(self.obj)}
        else:
            save_locals = None
        info = self._name_and_type()
        yield traceback.FrameSummary(
            parent.filename,
            self.start_line or parent.lineno,
            parent.funcname + (f" ({info})" if info else ""),
            locals=save_locals,
            line=override_line or ("" if self.start_line is None else None),
        )
        if self.inner_stack is not None:
            yield from self.inner_stack._frame_summaries(
                show_contexts=True,
                show_hidden_frames=show_hidden_frames,
                capture_locals=capture_locals,
            )
        for subctx in self.children:
            if isinstance(subctx, Context):
                yield from subctx._frame_summaries(
                    parent,
                    show_hidden_frames,
                    capture_locals,
                    "# " + (subctx.description or repr(subctx)),
                )

    def _format(
        self,
        opts: FormatOptions,
        parent: Optional[Frame] = None,
        *,
        show_lineno: bool = True,
    ) -> List[str]:
        if self.hide and not opts.show_hidden_frames:
            return []

        start_child = ". " if opts.ascii_only else "─ "
        continue_child = "  "

        linetext = ""
        if self.start_line is not None and parent is not None:
            linetext = linecache.getline(
                parent.filename, self.start_line, parent.pyframe.f_globals
            ).strip()
        if not linetext:
            if self.description:
                linetext = self.description
            else:
                linetext = "async with <???>:" if self.is_async else "with <???>:"

        comment_parts = []
        info = self._name_and_type()
        if info:
            comment_parts.append(info)
        if show_lineno and self.start_line is not None:
            comment_parts.append(f"(line {self.start_line})")
        if comment_parts:
            linetext += "  # " + " ".join(comment_parts)

        lines = [linetext + "\n"]
        if self.inner_stack is not None:
            lines.extend(self.inner_stack._format(opts)[1:])
        did_blank = False
        for child in self.children:
            if isinstance(child, Context):
                sublines = child._format(opts, show_lineno=False)
            else:  # child task stack
                sublines = child._format(opts)
                if child.root is not None:
                    sublines[0] = f"{child.root!r}\n"
                else:
                    sublines[0] = "<unidentified child>\n"
                if child.frames:
                    # Add a blank line on each side of child task stacks
                    if not did_blank:
                        lines.append(continue_child + "\n")
                    sublines.append("\n")
            did_blank = bool(sublines and not sublines[-1].strip())
            for idx, line in enumerate(sublines):
                marker = start_child if idx == 0 else continue_child
                lines.append(marker + line)
        return lines


@dataclass
class StackSlice:
    """Identifies a contiguous series of frames that we want to analyze.

    `StackSlice` has no logic on its own; its only use is as something to
    pass to :func:`extract` or return from an :func:`unwrap_stackitem` hook.

    This can be used in three different ways:

    * If `inner` is not None, then the `StackSlice` logically
      contains currently-executing frames that are direct or
      indirect callers of `inner`, ending with `inner` itself.
      Iteration begins with `inner` and proceeds outward via
      ``frame.f_back`` links until the frame `outer` is reached
      or `limit` frames have been extracted. If neither of those is
      specified, then the stack slice starts with the outermost frame
      of the thread on which `inner` is running.

      If `inner` belongs to a suspended coroutine or
      generator, or if it otherwise is not linked to other frames
      via its ``f_back`` attribute, then the returned traceback will
      contain only `inner` and not any of its callers.

    * If `inner` is None but `outer` is not, the `StackSlice`
      contains `outer` followed by its currently-executing direct
      and indirect callees, up to `limit` frames total.

      If `outer` is executing on the current thread, then the
      `StackSlice` ends with the frame that called :func:`stackscope.extract`
      (unless it is cut off before that by reaching its `limit`).
      If it is executing on some other thread, and remains so throughout
      the stack extraction process, then the `StackSlice` ends with the
      innermost frame on that thread. In any other case -- if `outer`
      belongs to a suspended coroutine, generator, greenlet, or if it starts
      or stops running on another thread while :func:`stackscope.extract`
      is executing -- the returned stack will contain information only on
      `outer` itself.

    * If `inner` and `outer` are both None, the `StackSlice`
      contains the entirety of the current thread's stack, ending with
      the frame that made the call to :func:`stackscope.extract`.

    """

    outer: Optional[types.FrameType] = None
    inner: Optional[types.FrameType] = None
    limit: Optional[int] = None
