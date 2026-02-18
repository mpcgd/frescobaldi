"""
Full test suite for the Frescobaldi codebase.

Covers:
  - Syntax validity for every .py file
  - Regression tests for all known bug fixes (AST-based, no Qt needed)
  - signals.py  — pure-Python signal/slot system (imported directly)
  - variables.py — pure parsing functions (exec'd in isolation)
  - util.py      — pure utility functions (exec'd in isolation)
  - snippet/snippets.py — pure parsing functions (exec'd in isolation)
  - job/queue.py — pure data-structure classes (exec'd in isolation)
  - backup.py    — backupName logic (exec'd in isolation)
  - preferences/import_export.py — indentXml, toBool (exec'd in isolation)
  - SVG / XML file integrity
  - Source-code analysis (no bare print() in non-debug modules, etc.)

All tests run without PyQt6 or a display.
"""

import ast
import bisect
import codecs
import contextlib
import glob
import io
import itertools
import os
import re
import sys
import types
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
FRESCOBALDI = ROOT / "frescobaldi"

# ─── add frescobaldi/ to sys.path so we can import pure-Python modules ────────
if str(FRESCOBALDI) not in sys.path:
    sys.path.insert(0, str(FRESCOBALDI))

# ─── Pure-function / class extractor ─────────────────────────────────────────

def _stdlib_ns():
    ns = {"__builtins__": __builtins__}
    for mod in ("re", "codecs", "io", "os", "itertools", "bisect",
                "contextlib", "types", "weakref", "sys",
                "collections", "unicodedata", "pathlib", "heapq",
                "enum", "time"):
        try:
            ns[mod] = __import__(mod)
        except ImportError:
            pass
    from enum import Enum
    ns["Enum"] = Enum
    return ns


def _load_functions(rel_path, *names, extra_ns=None, src_patch=None):
    """Extract named top-level functions *and* classes from a source file.

    Uses ast.get_source_segment() to pull each definition's text out of the
    file, then exec()s them into a minimal stdlib-only namespace — no Qt
    needed.  Module-level assignments (constants, compiled regexes …) that
    appear *before* the definitions are also exec'd first.

    src_patch: optional list of (old_str, new_str) string replacements
               applied to the raw source before parsing (for stripping Qt).
    """
    src = Path(ROOT / rel_path).read_text()
    if src_patch:
        for old, new in src_patch:
            src = src.replace(old, new)

    tree = ast.parse(src)

    ns = _stdlib_ns()
    if extra_ns:
        ns.update(extra_ns)

    # exec module-level simple assignments (e.g. _LINES = 5, regex compiles)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            seg = ast.get_source_segment(src, node)
            if seg:
                try:
                    exec(compile(seg, rel_path, "exec"), ns)
                except Exception:
                    pass

    # exec requested function/class definitions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, rel_path, "exec"), ns)

    return {name: ns[name] for name in names if name in ns}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Syntax validity: every .py file must parse cleanly
# ─────────────────────────────────────────────────────────────────────────────

def _all_py_files():
    return (
        list(FRESCOBALDI.rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
    )

def test_all_py_files_parse():
    """Every .py file in the project must parse without SyntaxError."""
    errors = []
    for path in _all_py_files():
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as e:
            errors.append(f"{path.relative_to(ROOT)}: {e}")
    assert not errors, "Syntax errors found:\n" + "\n".join(errors)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Regression: fix/return-in-finally (SnippetModel.removeRows)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_return_in_finally_removerows():
    """removeRows must not have 'return' inside a finally block."""
    source = (FRESCOBALDI / "snippet/model.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "removeRows":
            for child in ast.walk(node):
                if isinstance(child, ast.Try) and child.finalbody:
                    for stmt in child.finalbody:
                        assert not isinstance(stmt, ast.Return), (
                            "removeRows still has 'return' inside a finally block"
                        )
            return
    raise AssertionError("removeRows not found in snippet/model.py")


def test_return_true_at_function_level_removerows():
    """removeRows must still return True — outside the finally block."""
    source = (FRESCOBALDI / "snippet/model.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "removeRows":
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    val = stmt.value
                    assert isinstance(val, ast.Constant) and val.value is True
                    return
    raise AssertionError("No top-level 'return True' in removeRows")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Regression: fix/svg-font-cross-platform
# ─────────────────────────────────────────────────────────────────────────────

_SVG_PATHS = (
    glob.glob(str(FRESCOBALDI / "symbols/*.svg"))
    + glob.glob(str(FRESCOBALDI / "icons/*.svg"))
)

def test_no_bare_century_schoolbook_in_svgs():
    """'Century Schoolbook L' must not appear without cross-platform fallbacks."""
    for path in _SVG_PATHS:
        content = Path(path).read_text()
        if "Century Schoolbook L" not in content:
            continue
        assert "Palatino" in content or "serif" in content, (
            f"{path}: Century Schoolbook L with no fallback"
        )

def test_all_svgs_are_valid_xml():
    """Every SVG must remain well-formed XML."""
    for path in _SVG_PATHS:
        try:
            ET.parse(path)
        except ET.ParseError as e:
            raise AssertionError(f"{path} invalid XML: {e}") from e

def test_oldfontsdialog_platform_aware_default():
    """oldfontsdialog.py must use a platform-aware _DEFAULT_ROMAN constant."""
    source = (FRESCOBALDI / "fonts/oldfontsdialog.py").read_text()
    assert "_DEFAULT_ROMAN" in source
    assert 'roman", "Century Schoolbook L"' not in source
    assert "sys.platform" in source


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Regression: fix/editor-font-singleton-mutation (app.py)
# ─────────────────────────────────────────────────────────────────────────────

def test_editor_font_returns_copy_not_reference():
    """editor_font() must copy the singleton with QFont(), not alias it."""
    source = (FRESCOBALDI / "app.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "editor_font":
            src_seg = ast.get_source_segment(source, node)
            # Must contain QFont(_editor_font) to create a copy
            assert "QFont(_editor_font)" in src_seg, (
                "editor_font() does not copy the singleton via QFont(_editor_font)"
            )
            # Must NOT contain a bare 'font = _editor_font' alias assignment
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    seg = ast.get_source_segment(source, child)
                    assert seg != "font = _editor_font", (
                        "editor_font() contains a bare alias 'font = _editor_font' "
                        "that would mutate the singleton"
                    )
            return
    raise AssertionError("editor_font() not found in app.py")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Regression: fix/util-encode-uses-wrong-variable
# ─────────────────────────────────────────────────────────────────────────────

def test_encode_uses_enc_not_encoding():
    """util.encode() must call text.encode(enc), not text.encode(encoding)."""
    source = (FRESCOBALDI / "util.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "encode":
            # Must NOT call encode(encoding) with the raw parameter
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if (isinstance(child.func, ast.Attribute)
                            and child.func.attr == "encode"
                            and child.args):
                        arg = child.args[0]
                        assert not (isinstance(arg, ast.Name)
                                    and arg.id == "encoding"), (
                            "encode() passes raw 'encoding' param to text.encode() "
                            "— should use the resolved 'enc' variable"
                        )
            return
    raise AssertionError("encode() not found in util.py")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Regression: fix/job-manager-signal-and-qobject-init
# ─────────────────────────────────────────────────────────────────────────────

def test_job_manager_finished_uses_self_job():
    """JobManager._finished() must emit self._job, not the module-level job()."""
    source = (FRESCOBALDI / "job/manager.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_finished":
            seg = ast.get_source_segment(source, node)
            assert "self.finished(self._job," in seg, (
                "_finished() does not pass self._job to self.finished()"
            )
            assert "self.finished(job," not in seg, (
                "_finished() still passes the module-level job function"
            )
            return
    raise AssertionError("_finished() not found in job/manager.py")


def test_global_job_queue_calls_super_init():
    """GlobalJobQueue.__init__ must call super().__init__() first."""
    source = (FRESCOBALDI / "job/queue.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GlobalJobQueue":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    # first statement must be super().__init__()
                    first = item.body[0]
                    seg = ast.get_source_segment(source, first)
                    assert "super().__init__()" in seg, (
                        "GlobalJobQueue.__init__() does not call super().__init__() "
                        "as its first statement"
                    )
                    return
    raise AssertionError("GlobalJobQueue.__init__ not found in job/queue.py")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Regression: fix/import-theme-allstyles-key-error
# ─────────────────────────────────────────────────────────────────────────────

def test_import_theme_uses_style_tag_as_key():
    """importTheme() must use style.tag (string) not style (XML element) as dict key."""
    source = (FRESCOBALDI / "preferences/import_export.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "importTheme":
            seg = ast.get_source_segment(source, node)
            assert "if style.tag not in tfd.allStyles" in seg, (
                "importTheme() does not check style.tag; uses XML element as key"
            )
            assert "tfd.allStyles[style.tag] = {}" in seg, (
                "importTheme() does not initialise tfd.allStyles[style.tag]"
            )
            assert "if style not in tfd.allStyles" not in seg, (
                "importTheme() still uses XML element for membership test"
            )
            return
    raise AssertionError("importTheme() not found in preferences/import_export.py")


def test_import_export_no_duplicate_xml_import():
    """The try/except that imported the same module twice must be gone."""
    source = (FRESCOBALDI / "preferences/import_export.py").read_text()
    # Both branches importing the same thing means the except block is dead.
    # After the fix there should be exactly one import of ElementTree.
    import_count = source.count("import xml.etree.ElementTree")
    assert import_count == 1, (
        f"Expected 1 xml.etree.ElementTree import, found {import_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — signals.py (pure Python — imported directly)
# ─────────────────────────────────────────────────────────────────────────────

from signals import Signal, SignalContext  # noqa: E402 (after sys.path insert)


# Signal tests must NOT connect list.append directly — list objects do not
# support weakrefs, so MethodListener.add() raises TypeError.  Use a plain
# function that closes over a list, or a small helper class, instead.

class _Collector:
    """Weakref-safe callable that accumulates received values."""
    def __init__(self):
        self.data = []
    def __call__(self, *args):
        self.data.append(args[0] if len(args) == 1 else args)


def test_signal_basic_emit():
    """A connected function must be called when the signal fires."""
    sig = Signal()
    received = []
    def slot(v):
        received.append(v)
    sig.connect(slot)
    sig("hello")
    assert received == ["hello"]


def test_signal_multiple_listeners():
    """All connected functions must be called."""
    sig = Signal()
    a, b = _Collector(), _Collector()
    sig.connect(a)
    sig.connect(b)
    sig(42)
    assert a.data == [42] and b.data == [42]


def test_signal_disconnect():
    """disconnect() must stop future calls to that slot."""
    sig = Signal()
    called = []
    def slot(v):
        called.append(v)
    sig.connect(slot)
    sig.disconnect(slot)
    sig("x")
    assert called == []


def test_signal_clear():
    """clear() must remove all listeners."""
    sig = Signal()
    called = []
    def slot(v):
        called.append(v)
    sig.connect(slot)
    sig.clear()
    sig("x")
    assert called == []


def test_signal_no_duplicate_connect():
    """Connecting the same function twice must not cause double calls."""
    sig = Signal()
    called = []
    def slot(v):
        called.append(v)
    sig.connect(slot)
    sig.connect(slot)
    sig(1)
    assert len(called) == 1


def test_signal_priority_order():
    """Lower-priority slots are called first (bisect insertion order)."""
    sig = Signal()
    order = []
    sig.connect(lambda: order.append("low"),  priority=0)
    sig.connect(lambda: order.append("high"), priority=10)
    sig()
    assert order == ["low", "high"]


def test_signal_blocked_context_manager():
    """Emissions inside a 'with sig.blocked()' block must be suppressed."""
    sig = Signal()
    called = []
    def slot(v):
        called.append(v)
    sig.connect(slot)
    with sig.blocked():
        sig("suppressed")
    assert called == []
    sig("visible")
    assert called == ["visible"]


def test_signal_method_listener_weakref():
    """When the listener object is garbage-collected the signal disconnects."""
    sig = Signal()
    class Obj:
        def handler(self, v):
            pass
    obj = Obj()
    sig.connect(obj.handler)
    assert len(sig.listeners) == 1
    del obj
    # Trigger GC by emitting — dead weakrefs are pruned on call
    sig(1)
    assert len(sig.listeners) == 0


def test_signal_descriptor_per_instance():
    """Each instance gets its own independent Signal when used as descriptor."""
    class MyClass:
        changed = Signal()
    a, b = MyClass(), MyClass()
    ca, cb = _Collector(), _Collector()
    a.changed.connect(ca)
    b.changed.connect(cb)
    a.changed(1)
    b.changed(2)
    assert ca.data == [1] and cb.data == [2]


def test_signal_accepts_fewer_args():
    """A slot that accepts fewer arguments than emitted should still work."""
    sig = Signal()
    called = []
    sig.connect(lambda: called.append(True))
    sig("extra", "args", "ignored")
    assert called == [True]


def test_signal_emit_method():
    """emit() is equivalent to __call__."""
    sig = Signal()
    c = _Collector()
    sig.connect(c)
    sig.emit("via emit")
    assert c.data == ["via emit"]


def test_signal_chain():
    """A signal can be connected to another signal (forwarding)."""
    sig1 = Signal()
    sig2 = Signal()
    c = _Collector()
    sig2.connect(c)
    sig1.connect(sig2)
    sig1("forwarded")
    assert c.data == ["forwarded"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — variables.py pure functions (exec'd in isolation)
# ─────────────────────────────────────────────────────────────────────────────

_var_fns = _load_functions(
    "frescobaldi/variables.py",
    "variables", "positions", "prepare",
)
_variables  = _var_fns["variables"]
_positions  = _var_fns["positions"]
_prepare    = _var_fns["prepare"]


def test_variables_empty_text():
    assert _variables("") == {}


def test_variables_single_line():
    text = "% -*- coding: utf-8; -*-"
    result = _variables(text)
    assert result.get("coding") == "utf-8"


def test_variables_multiline_header():
    text = "% -*- coding: latin1; indent-tabs-mode: nil; -*-\n\nsome music"
    result = _variables(text)
    assert result["coding"] == "latin1"
    assert result["indent-tabs-mode"] == "nil"


def test_variables_no_variables():
    text = "\\version \"2.24\"\n\n{ c d e f }"
    assert _variables(text) == {}


def test_prepare_bool_true():
    for val in ("true", "yes", "on", "t", "1", "True", "YES"):
        assert _prepare(val, False) is True


def test_prepare_bool_false():
    for val in ("false", "no", "off", "f", "0", "False", "NO"):
        assert _prepare(val, True) is False


def test_prepare_int():
    assert _prepare("42", 0) == 42


def test_prepare_int_invalid_returns_default():
    assert _prepare("abc", 0) == 0


def test_prepare_string_passthrough():
    assert _prepare("hello", "default") == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — util.py pure functions (exec'd in isolation)
# ─────────────────────────────────────────────────────────────────────────────

# variables() is needed by encode()/decode(), so inject it
_util_fns = _load_functions(
    "frescobaldi/util.py",
    "get_bom", "decode", "encode",
    "universal_newlines", "platform_newlines",
    "naturalsort", "filenamesort",
    "next_file", "uniq", "group_files",
    extra_ns={"variables": _var_fns},  # inject the variables module shim
)

# variables module shim: provide variables.variables()
class _VariablesShim:
    variables = staticmethod(_variables)
_util_extra = {"variables": _VariablesShim()}

_util_fns = _load_functions(
    "frescobaldi/util.py",
    "get_bom", "decode", "encode",
    "universal_newlines", "platform_newlines",
    "naturalsort", "filenamesort",
    "next_file", "uniq", "group_files",
    extra_ns={"variables": _VariablesShim()},
)

get_bom            = _util_fns["get_bom"]
decode             = _util_fns["decode"]
encode             = _util_fns["encode"]
universal_newlines = _util_fns["universal_newlines"]
platform_newlines  = _util_fns["platform_newlines"]
naturalsort        = _util_fns["naturalsort"]
filenamesort       = _util_fns["filenamesort"]
next_file          = _util_fns["next_file"]
uniq               = _util_fns["uniq"]
group_files        = _util_fns["group_files"]


# get_bom
def test_get_bom_utf8():
    data = codecs.BOM_UTF8 + b"hello"
    enc, rest = get_bom(data)
    assert enc == "utf-8" and rest == b"hello"

def test_get_bom_utf16_le():
    data = codecs.BOM_UTF16_LE + b"\x41\x00"
    enc, rest = get_bom(data)
    assert enc == "utf_16_le"

def test_get_bom_none():
    data = b"no bom here"
    enc, rest = get_bom(data)
    assert enc is None and rest == data

# decode
def test_decode_utf8_bom():
    data = codecs.BOM_UTF8 + "héllo".encode("utf-8")
    assert decode(data) == "héllo"

def test_decode_explicit_encoding():
    data = "café".encode("latin1")
    assert decode(data, encoding="latin1") == "café"

def test_decode_fallback_latin1():
    data = bytes([0xE9])  # é in latin-1
    result = decode(data)
    assert isinstance(result, str)

def test_decode_from_coding_variable():
    text = "% -*- coding: utf-8; -*-\ncafé"
    data = text.encode("utf-8")
    assert decode(data) == text

# encode
def test_encode_default_utf8():
    assert encode("hello") == b"hello"

def test_encode_explicit_encoding():
    assert encode("café", encoding="latin1") == "café".encode("latin1")

def test_encode_from_coding_variable():
    text = "% -*- coding: latin1; -*-\ncafé"
    result = encode(text)
    # should have used latin1 from coding variable
    assert result == text.encode("latin1")

def test_encode_uses_enc_not_encoding_runtime():
    """The 'encoding' param being None must not cause TypeError when coding var present."""
    text = "% -*- coding: utf-8; -*-\nhello"
    # This would raise TypeError before the fix (text.encode(None))
    result = encode(text, encoding=None)
    assert result == text.encode("utf-8")

def test_encode_bad_encoding_falls_back_to_default():
    """If coding variable has an invalid encoding, fall back to utf-8."""
    text = "% -*- coding: nonexistent-codec; -*-\nhello"
    result = encode(text)
    assert result == text.encode("utf-8")

# universal_newlines
def test_universal_newlines_cr():
    assert universal_newlines("a\rb") == "a\nb"

def test_universal_newlines_crlf():
    assert universal_newlines("a\r\nb") == "a\nb"

def test_universal_newlines_lf():
    assert universal_newlines("a\nb") == "a\nb"

def test_universal_newlines_mixed():
    assert universal_newlines("a\rb\r\nc") == "a\nb\nc"

# naturalsort
def test_naturalsort_numeric():
    names = ["item10", "item2", "item1"]
    assert sorted(names, key=naturalsort) == ["item1", "item2", "item10"]

def test_naturalsort_strings():
    names = ["banana", "apple", "cherry"]
    assert sorted(names, key=naturalsort) == ["apple", "banana", "cherry"]

def test_naturalsort_version():
    versions = ["2.22.1", "2.10.0", "2.9.0"]
    assert sorted(versions, key=naturalsort) == ["2.9.0", "2.10.0", "2.22.1"]

# next_file
def test_next_file_no_number():
    assert next_file("score.ly") == "score-1.ly"

def test_next_file_with_number():
    assert next_file("score-3.ly") == "score-4.ly"

def test_next_file_preserves_extension():
    result = next_file("my-file.pdf")
    assert result.endswith(".pdf")

# uniq
def test_uniq_removes_duplicates():
    assert list(uniq([1, 2, 2, 3, 1])) == [1, 2, 3]

def test_uniq_preserves_order():
    assert list(uniq([3, 1, 2, 1, 3])) == [3, 1, 2]

def test_uniq_empty():
    assert list(uniq([])) == []

# group_files
def test_group_files_basic():
    names = ["a.ly", "b.pdf", "c.midi", "d.ly"]
    ly, pdf = list(group_files(names, ["ly", "pdf"]))
    assert ly == ["a.ly", "d.ly"]
    assert pdf == ["b.pdf"]

def test_group_files_negation():
    names = ["a.ly", "b.pdf", "c.midi"]
    other, = list(group_files(names, ["!ly pdf"]))
    assert "c.midi" in other

def test_group_files_case_insensitive():
    names = ["A.LY", "b.pdf"]
    ly, _ = list(group_files(names, ["ly", "pdf"]))
    assert "A.LY" in ly


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — snippet/snippets.py pure functions (exec'd in isolation)
# ─────────────────────────────────────────────────────────────────────────────

_snip_fns = _load_functions(
    "frescobaldi/snippet/snippets.py",
    "parse", "maketitle", "expand",
)
_parse     = _snip_fns["parse"]
_maketitle = _snip_fns["maketitle"]
_expand    = _snip_fns["expand"]


def test_parse_no_variables():
    tv = _parse("hello world")
    assert tv.text == "hello world"
    assert tv.variables == {}

def test_parse_single_variable():
    tv = _parse("-*- name: value;\nhello")
    assert tv.variables["name"] == "value"
    assert tv.text == "hello"

def test_parse_multiple_variables():
    tv = _parse("-*- a: 1; b: 2;\ntext")
    assert tv.variables == {"a": "1", "b": "2"}

def test_parse_boolean_variable():
    tv = _parse("-*- indent-ly;\ntext")
    assert tv.variables.get("indent-ly") is True

def test_parse_strips_header_lines():
    tv = _parse("-*- x: y;\nactual text")
    assert tv.text == "actual text"
    assert "-*-" not in tv.text

def test_maketitle_single_line():
    assert _maketitle("hello world") == "hello world"

def test_maketitle_strips_leading_blank_lines():
    assert _maketitle("\n\nhello") == "hello"

def test_maketitle_multiline_elides():
    result = _maketitle("first\nsecond\nthird")
    assert "first" in result
    assert "third" in result
    assert "..." in result

def test_maketitle_replaces_expansions():
    result = _maketitle("before $VAR after")
    assert "$VAR" not in result
    assert "..." in result

def test_expand_plain_text():
    parts = list(_expand("hello"))
    assert parts == [("hello", "")]

def test_expand_dollar_name():
    parts = list(_expand("Hello $NAME!"))
    assert any(exp == "NAME" for _, exp in parts)

def test_expand_double_dollar():
    parts = list(_expand("cursor $$"))
    assert any(exp == "$" for _, exp in parts)

def test_expand_braced():
    parts = list(_expand("${some text}"))
    assert any(exp == "some text" for _, exp in parts)

def test_expand_escaped_brace():
    parts = list(_expand("${a\\}b}"))
    assert any(exp == "a}b" for _, exp in parts)

def test_expand_multiple():
    parts = list(_expand("$A and $B"))
    expansions = [exp for _, exp in parts if exp]
    assert "A" in expansions
    assert "B" in expansions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — job/queue.py pure data structures (exec'd in isolation)
# ─────────────────────────────────────────────────────────────────────────────

# Queue, Stack, PriorityQueue inherit from AbstractStackQueue / AbstractQueue(QObject).
# We strip QObject and load the full hierarchy so subclasses resolve correctly.
_queue_fns = _load_functions(
    "frescobaldi/job/queue.py",
    "AbstractQueue", "AbstractStackQueue", "Queue", "Stack", "PriorityQueue",
    src_patch=[("(QObject)", "(object)")],
)
_Queue         = _queue_fns["Queue"]
_Stack         = _queue_fns["Stack"]
_PriorityQueue = _queue_fns["PriorityQueue"]


class _DummyJob:
    """Minimal job stub for queue tests."""
    def __init__(self, priority=1):
        self._priority = priority
    def priority(self):
        return self._priority


def test_queue_fifo():
    """Queue is first-in-first-out."""
    q = _Queue()
    j1, j2, j3 = _DummyJob(), _DummyJob(), _DummyJob()
    q.push(j1); q.push(j2); q.push(j3)
    assert q.pop() is j1
    assert q.pop() is j2
    assert q.pop() is j3

def test_queue_empty():
    q = _Queue()
    assert q.empty()
    q.push(_DummyJob())
    assert not q.empty()

def test_queue_length():
    q = _Queue()
    for _ in range(5):
        q.push(_DummyJob())
    assert q.length() == 5

def test_queue_clear():
    q = _Queue()
    q.push(_DummyJob()); q.push(_DummyJob())
    q.clear()
    assert q.empty()

def test_stack_lifo():
    """Stack is last-in-first-out."""
    s = _Stack()
    j1, j2, j3 = _DummyJob(), _DummyJob(), _DummyJob()
    s.push(j1); s.push(j2); s.push(j3)
    assert s.pop() is j3
    assert s.pop() is j2
    assert s.pop() is j1

def test_priority_queue_highest_first():
    """PriorityQueue pops the job with the *lowest* priority number first."""
    pq = _PriorityQueue()
    low  = _DummyJob(priority=1)
    high = _DummyJob(priority=5)
    pq.push(high)
    pq.push(low)
    # heapq pops smallest, and priority=1 < priority=5
    assert pq.pop() is low

def test_priority_queue_fifo_within_same_priority():
    """Jobs with equal priority must be served FIFO."""
    pq = _PriorityQueue()
    j1 = _DummyJob(priority=1)
    j2 = _DummyJob(priority=1)
    pq.push(j1)
    pq.push(j2)
    assert pq.pop() is j1

def test_priority_queue_empty():
    pq = _PriorityQueue()
    assert pq.empty()

def test_priority_queue_clear():
    pq = _PriorityQueue()
    pq.push(_DummyJob())
    pq.clear()
    assert pq.empty()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — backup.py pure logic (exec'd in isolation)
# ─────────────────────────────────────────────────────────────────────────────

_backup_src = (FRESCOBALDI / "backup.py").read_text()
_backup_ns  = {"__builtins__": __builtins__}
exec(compile(_backup_src.replace("from PyQt6.QtCore import QSettings", ""),
             "backup.py", "exec"), _backup_ns)
_backupName = _backup_ns["backupName"]

# Patch scheme() for testing
def _scheme_tilde():
    return "FILE~"

def _scheme_bak():
    return "FILE.bak"

_backup_ns["scheme"] = _scheme_tilde


def test_backup_name_tilde_scheme():
    assert _backupName("/path/to/score.ly") == "/path/to/score.ly~"

def test_backup_name_bak_scheme():
    _backup_ns["scheme"] = _scheme_bak
    assert _backupName("/path/to/score.ly") == "/path/to/score.ly.bak"
    _backup_ns["scheme"] = _scheme_tilde

def test_backup_name_preserves_full_path():
    result = _backupName("/home/user/music/piece.ly")
    assert result.startswith("/home/user/music/piece.ly")

def test_backup_name_replaces_file_token():
    """The backup scheme replaces FILE, not just the basename."""
    _backup_ns["scheme"] = lambda: "FILE.backup"
    result = _backupName("test.ly")
    assert result == "test.ly.backup"
    _backup_ns["scheme"] = _scheme_tilde


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — preferences/import_export.py — indentXml and toBool
# ─────────────────────────────────────────────────────────────────────────────

_ie_fns = _load_functions(
    "frescobaldi/preferences/import_export.py",
    "indentXml", "toBool",
)
_indentXml = _ie_fns["indentXml"]
_toBool    = _ie_fns["toBool"]


def test_toBool_true():
    assert _toBool("True") is True

def test_toBool_false():
    assert _toBool("False") is False

def test_toBool_invalid():
    try:
        _toBool("maybe")
        raise AssertionError("Expected KeyError for invalid toBool value")
    except KeyError:
        pass

def test_indentXml_adds_indentation():
    root = ET.fromstring("<root><child/></root>")
    _indentXml(root)
    # After indentation, the child's tail should contain a newline
    child = list(root)[0]
    assert child.tail is not None and "\n" in child.tail

def test_indentXml_nested():
    root = ET.fromstring("<a><b><c/></b></a>")
    _indentXml(root)
    b = list(root)[0]
    c = list(b)[0]
    assert c.tail is not None

def test_indentXml_empty_element():
    root = ET.fromstring("<root/>")
    _indentXml(root)  # Must not raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — Source-code analysis
# ─────────────────────────────────────────────────────────────────────────────

# Modules where print() calls are known/intentional (debug helpers, CLI tools,
# test scripts, parsers that deliberately dump to stdout).
_ALLOWED_PRINT_MODULES = {
    # explicit debug / developer-tool modules
    "debug.py", "debuginfo.py",
    # i18n command-line scripts
    "i18n/messages.py", "i18n/md2pot.py", "i18n/mo-gen.py",
    "i18n/molint.py", "i18n/po-update.py",
    # VCS manual test script
    "vcs/test.py",
    # i18n Qt translator helper (uses print for diagnostics)
    "i18n/qtranslator.py",
    # Legacy debug prints — known pre-existing issues, tracked for removal
    "app.py",
    "hyphenator.py",
    "simplestate.py",
    "svgview/view.py",
    "preferences/paths.py",
    "midifile/parser.py",
    "quickinsert/buttongroup.py",
    "fonts/preview.py",
}

def test_no_bare_print_in_production_code():
    """Production modules must not contain bare print() calls.

    Modules in _ALLOWED_PRINT_MODULES are excluded (legacy debug prints or
    intentional CLI output).  New print() calls in unlisted modules will fail.
    """
    violations = []
    for path in FRESCOBALDI.rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        # Match both "app.py" and "frescobaldi/app.py" style paths
        if any(rel.endswith(m.replace("/", os.sep)) or
               rel.endswith(m) for m in _ALLOWED_PRINT_MODULES):
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    violations.append(f"{rel}:{node.lineno}")
    assert not violations, (
        "Bare print() calls found in production code:\n" + "\n".join(violations)
    )


# Modules where wildcard imports are accepted.
_ALLOWED_WILDCARD_MODULES = {
    # debug.py intentionally does 'from x import *' for interactive use
    "frescobaldi/debug.py",
}

def test_no_wildcard_imports():
    """No module should use 'from x import *' (except known allowlisted modules)."""
    violations = []
    for path in _all_py_files():
        rel = str(path.relative_to(ROOT))
        if rel in _ALLOWED_WILDCARD_MODULES:
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(a.name == "*" for a in node.names):
                    violations.append(f"{rel}:{node.lineno}")
    assert not violations, (
        "'from x import *' found:\n" + "\n".join(violations)
    )


def test_no_return_in_finally_anywhere():
    """No function in the codebase should have 'return' inside a finally block."""
    violations = []
    for path in _all_py_files():
        source = path.read_text()
        rel = str(path.relative_to(ROOT))
        tree = ast.parse(source, filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            for child in ast.walk(func):
                if isinstance(child, ast.Try) and child.finalbody:
                    for stmt in child.finalbody:
                        if isinstance(stmt, ast.Return):
                            violations.append(
                                f"{rel}:{stmt.lineno} in {func.name}()"
                            )
    assert not violations, (
        "'return' inside finally block found:\n" + "\n".join(violations)
    )


def test_no_bare_except():
    """No bare 'except:' clauses (should use 'except Exception:' or narrower)."""
    violations = []
    for path in _all_py_files():
        source = path.read_text()
        rel = str(path.relative_to(ROOT))
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"{rel}:{node.lineno}")
    assert not violations, (
        "Bare 'except:' clauses found (use 'except Exception:' or narrower):\n"
        + "\n".join(violations)
    )


def test_all_init_files_exist_for_subpackages():
    """Every Python sub-package directory must have an __init__.py."""
    missing = []
    for dirpath in FRESCOBALDI.iterdir():
        if dirpath.is_dir() and not dirpath.name.startswith(("_", ".")):
            init = dirpath / "__init__.py"
            if not init.exists():
                missing.append(str(dirpath.relative_to(ROOT)))
    assert not missing, (
        "Sub-packages missing __init__.py:\n" + "\n".join(missing)
    )


def test_no_mutable_default_arguments():
    """No function should use mutable literals (list/dict/set) as defaults."""
    violations = []
    mutable = (ast.List, ast.Dict, ast.Set)
    for path in _all_py_files():
        source = path.read_text()
        rel = str(path.relative_to(ROOT))
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, mutable):
                        violations.append(
                            f"{rel}:{node.lineno} {node.name}()"
                        )
    assert not violations, (
        "Mutable default arguments found:\n" + "\n".join(violations)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    _tests = [
        # Section 1 — syntax
        test_all_py_files_parse,
        # Section 2 — regression: return-in-finally
        test_no_return_in_finally_removerows,
        test_return_true_at_function_level_removerows,
        # Section 3 — regression: svg font
        test_no_bare_century_schoolbook_in_svgs,
        test_all_svgs_are_valid_xml,
        test_oldfontsdialog_platform_aware_default,
        # Section 4 — regression: editor_font singleton
        test_editor_font_returns_copy_not_reference,
        # Section 5 — regression: util.encode variable
        test_encode_uses_enc_not_encoding,
        # Section 6 — regression: job module
        test_job_manager_finished_uses_self_job,
        test_global_job_queue_calls_super_init,
        # Section 7 — regression: importTheme key error
        test_import_theme_uses_style_tag_as_key,
        test_import_export_no_duplicate_xml_import,
        # Section 8 — signals.py
        test_signal_basic_emit,
        test_signal_multiple_listeners,
        test_signal_disconnect,
        test_signal_clear,
        test_signal_no_duplicate_connect,
        test_signal_priority_order,
        test_signal_blocked_context_manager,
        test_signal_method_listener_weakref,
        test_signal_descriptor_per_instance,
        test_signal_accepts_fewer_args,
        test_signal_emit_method,
        test_signal_chain,
        # Section 9 — variables.py
        test_variables_empty_text,
        test_variables_single_line,
        test_variables_multiline_header,
        test_variables_no_variables,
        test_prepare_bool_true,
        test_prepare_bool_false,
        test_prepare_int,
        test_prepare_int_invalid_returns_default,
        test_prepare_string_passthrough,
        # Section 10 — util.py
        test_get_bom_utf8,
        test_get_bom_utf16_le,
        test_get_bom_none,
        test_decode_utf8_bom,
        test_decode_explicit_encoding,
        test_decode_fallback_latin1,
        test_decode_from_coding_variable,
        test_encode_default_utf8,
        test_encode_explicit_encoding,
        test_encode_from_coding_variable,
        test_encode_uses_enc_not_encoding_runtime,
        test_encode_bad_encoding_falls_back_to_default,
        test_universal_newlines_cr,
        test_universal_newlines_crlf,
        test_universal_newlines_lf,
        test_universal_newlines_mixed,
        test_naturalsort_numeric,
        test_naturalsort_strings,
        test_naturalsort_version,
        test_next_file_no_number,
        test_next_file_with_number,
        test_next_file_preserves_extension,
        test_uniq_removes_duplicates,
        test_uniq_preserves_order,
        test_uniq_empty,
        test_group_files_basic,
        test_group_files_negation,
        test_group_files_case_insensitive,
        # Section 11 — snippet/snippets.py
        test_parse_no_variables,
        test_parse_single_variable,
        test_parse_multiple_variables,
        test_parse_boolean_variable,
        test_parse_strips_header_lines,
        test_maketitle_single_line,
        test_maketitle_strips_leading_blank_lines,
        test_maketitle_multiline_elides,
        test_maketitle_replaces_expansions,
        test_expand_plain_text,
        test_expand_dollar_name,
        test_expand_double_dollar,
        test_expand_braced,
        test_expand_escaped_brace,
        test_expand_multiple,
        # Section 12 — job/queue.py data structures
        test_queue_fifo,
        test_queue_empty,
        test_queue_length,
        test_queue_clear,
        test_stack_lifo,
        test_priority_queue_highest_first,
        test_priority_queue_fifo_within_same_priority,
        test_priority_queue_empty,
        test_priority_queue_clear,
        # Section 13 — backup.py
        test_backup_name_tilde_scheme,
        test_backup_name_bak_scheme,
        test_backup_name_preserves_full_path,
        test_backup_name_replaces_file_token,
        # Section 14 — import_export.py
        test_toBool_true,
        test_toBool_false,
        test_toBool_invalid,
        test_indentXml_adds_indentation,
        test_indentXml_nested,
        test_indentXml_empty_element,
        # Section 15 — source analysis
        test_no_bare_print_in_production_code,
        test_no_wildcard_imports,
        test_no_return_in_finally_anywhere,
        test_no_bare_except,
        test_all_init_files_exist_for_subpackages,
        test_no_mutable_default_arguments,
    ]

    passed = failed = 0
    for t in _tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed")
    sys.exit(failed)
