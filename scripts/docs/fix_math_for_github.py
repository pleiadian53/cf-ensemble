"""
Fix LaTeX Math for GitHub
=========================

Rewrite Markdown math so it renders on GitHub as well as in MkDocs (arithmatex).

GitHub runs Markdown *before* MathJax, which breaks math in two ways:

1. A ``$$...$$`` block that is not its own paragraph (no blank line around it,
   or sharing a line with text) is parsed as Markdown: ``_..._`` turns into
   italics and ``\\\\`` collapses to ``\\``.
2. Even in well-formed math, a backslash before ASCII punctuation is treated as
   a Markdown escape and dropped: ``\\{`` -> ``{``, ``\\|`` -> ``|``,
   ``\\,`` -> ``,``, and a mid-line ``\\\\`` -> ``\\``.

The fix keeps plain ``$``/``$$`` LaTeX (portable to MkDocs, VS Code, Obsidian)
and only uses TeX-equivalent spellings that contain no punctuation escapes:

- display blocks get blank lines around them; multi-line blocks get ``$$`` on
  their own lines; a ``\\\\`` row break followed by more content is moved to
  the end of its line
- ``\\{ \\}`` -> ``\\lbrace \\rbrace``, ``\\|`` -> ``\\Vert``,
  ``\\,`` -> ``\\thinspace``, ``\\;`` -> ``\\mskip5mu``, ``\\!`` -> ``\\negthinspace``,
  ``\\%`` -> ``\\unicode{37}``
- GitHub pairs emphasis delimiters even inside math, across separate ``$..$``
  spans. Inline math (and display blocks containing ``& < >``, which emphasis
  double-escapes): a ``_`` after punctuation gets a trailing space
  (``}_{ui}`` -> ``}_ {ui}``) and ``*`` -> ``\\ast``
- inline math only: ``<``/``>`` -> ``\\lt``/``\\gt`` (GitHub double-escapes
  them), and in table rows ``|`` -> ``\\vert`` (avoids splitting the cell)

Anything that cannot be rewritten safely is reported for manual review.
Fenced code blocks and inline code spans are left untouched.

Usage:
    python scripts/docs/fix_math_for_github.py README.md docs/          # fix in place
    python scripts/docs/fix_math_for_github.py --check README.md docs/  # report only
"""

import argparse
import re
import sys
from pathlib import Path

# Punctuation escapes GitHub would strip, mapped to escape-free TeX equivalents
ESCAPE_MAP = {
    "{": r"\lbrace",
    "}": r"\rbrace",
    "|": r"\Vert",
    ",": r"\thinspace",
    ":": r"\mskip4mu",
    ">": r"\mskip4mu",
    ";": r"\mskip5mu",
    "!": r"\negthinspace",
    "%": r"\unicode{37}",  # MathJax-only; `\text{%}` would start a TeX comment
}
# Inline math only: < and > come out double-escaped (&amp;lt;)
INLINE_MAP = {"<": r"\lt", ">": r"\gt"}
# Emphasis delimiters GitHub pairs up even inside (and across) math spans: a `_`
# after punctuation (`}_{ui}`) can open emphasis, and any `*` can. Spacing the `_`
# (`}_ {ui}`) and spelling `*` as `\ast` changes nothing in TeX.
EMPHASIS_OPENER_RE = re.compile(r"(?<=[^\w\s\\])_(?=\S)|(?<!\\)\*")
# Escapes that are harmless on GitHub (not ASCII punctuation, or a TeX control space)
SAFE_ESCAPES = set(" \t\n")

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
INLINE_MATH_RE = re.compile(r"(?<![\\$\w])\$(?=[^\s$])((?:\\.|[^$\\])+?)(?<![\s\\])\$(?![\w$])")
LIST_ITEM_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")


def rewrite_tex(tex, display, in_table=False, nested=False, warnings=None, where=""):
    """Replace GitHub-hostile sequences in a TeX string with equivalent safe ones.

    `nested`: the display block sits inside a list item or other container.
    """
    out = []
    i, n = 0, len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\" and i + 1 < n:
            nxt = tex[i + 1]
            if nxt.isalpha():
                j = i + 1
                while j < n and tex[j].isalpha():
                    j += 1
                out.append(tex[i:j])
                i = j
                continue
            if nxt == "\\":
                j = i + 2
                m = re.match(r"\s*\[[^\]]*\]", tex[j:])  # optional \\[4pt]
                if m:
                    j += m.end()
                    if warnings is not None:
                        warnings.append(f"{where}: '\\\\[...]' row spacing; GitHub drops a backslash")
                rest = tex[j:].split("\n", 1)[0]
                out.append(tex[i:j])
                if rest.strip():
                    if display:
                        out.append("\n")
                        k = j
                        while k < n and tex[k] in " \t":
                            k += 1
                        j = k
                    elif warnings is not None:
                        warnings.append(f"{where}: inline math contains a row break '\\\\'")
                i = j
                continue
            if nxt in ESCAPE_MAP:
                rep = ESCAPE_MAP[nxt]
                if rep[-1].isalpha() and i + 2 < n and tex[i + 2].isalpha():
                    rep += " "
                out.append(rep)
                i += 2
                continue
            if nxt not in SAFE_ESCAPES and warnings is not None:
                warnings.append(f"{where}: unhandled escape '\\{nxt}' (GitHub drops the backslash)")
            out.append(tex[i:i + 2])
            i += 2
            continue
        rep = None
        if not display:
            rep = INLINE_MAP.get(ch)
            if in_table and ch == "|":
                rep = r"\vert"
        if rep is None:
            out.append(ch)
        else:
            if i + 1 < n and tex[i + 1].isalpha():
                rep += " "
            out.append(rep)
        i += 1
    tex = "".join(out)
    # Emphasis can make GitHub drop inline math entirely, leaks across display blocks
    # in separate list items, and double-escapes & < > (breaking `align`/`cases`).
    # Top-level display blocks without & < > are unaffected.
    if not display or nested or re.search(r"[&<>]", tex):
        tex = EMPHASIS_OPENER_RE.sub(_emphasis_proof, tex)
    return tex


def _emphasis_proof(m):
    if m.group(0) == "_":
        return "_ "
    end, s = m.end(), m.string
    return r"\ast " if end < len(s) and s[end].isalpha() else r"\ast"


def mask_code_spans(line):
    """Replace inline code spans with placeholders so their contents are never touched."""
    spans = []

    def _sub(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    return CODE_SPAN_RE.sub(_sub, line), spans


def unmask(line, spans):
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], line)


def fix_inline(line, warnings, where):
    """Rewrite every inline $...$ span on a prose line."""
    masked, spans = mask_code_spans(line)
    in_table = masked.lstrip().startswith("|")

    def _sub(m):
        body = m.group(1)
        if "\x00" in body:  # $`...`$ GitHub inline syntax: content already literal
            return m.group(0)
        if r"\begin{" in body:
            warnings.append(f"{where}: GitHub does not render \\begin{{...}} in inline math")
        tex = rewrite_tex(body, display=False, in_table=in_table, warnings=warnings, where=where)
        # GitHub drops math ending in ")" when ")" follows the closing $, e.g. "($f(x)$)"
        if tex.endswith(")") and m.string[m.end():m.end() + 1] == ")":
            tex += "{}"
        return "$" + tex + "$"

    return unmask(INLINE_MATH_RE.sub(_sub, masked), spans)


def block_indent(before):
    """Indentation for a display block that started after `before` on the same line."""
    m = LIST_ITEM_RE.match(before)
    if m:
        return " " * len(m.group(1))
    return re.match(r"\s*", before).group(0)


def fix_text(text, path="<text>"):
    """Return (fixed_text, warnings)."""
    lines = text.split("\n")
    out = []
    warnings = []
    need_blank = False  # a display block was just emitted
    fence = None

    def emit(line):
        nonlocal need_blank
        if need_blank and line.strip():
            out.append("")
        need_blank = False
        out.append(line)

    def emit_block(indent, body, multiline):
        nonlocal need_blank
        if out and out[-1].strip():
            out.append("")
        need_blank = False
        tex = rewrite_tex(body.strip("\n"), display=True, nested=bool(indent),
                          warnings=warnings, where=where)
        if "\n" in tex.strip():
            multiline = True
        if multiline:
            out.append(indent + "$$")
            for t in tex.strip("\n").split("\n"):
                if t.strip() and not t.startswith(indent):
                    t = indent + t.lstrip()
                out.append(t.rstrip())
            out.append(indent + "$$")
        else:
            out.append(indent + "$$" + tex.strip() + "$$")
        need_blank = True

    i = 0
    pending = None  # line remainder still to be processed
    while i < len(lines) or pending is not None:
        if pending is not None:
            line, pending = pending, None
        else:
            line = lines[i]
            i += 1
        where = f"{path}:{i}"

        m = FENCE_RE.match(line)
        if fence is not None:
            # CommonMark: a closing fence is at least as long and has no info string
            if (m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence)
                    and not line[m.end():].strip()):
                fence = None
            emit(line)
            continue
        if m:
            fence = m.group(1)
            emit(line)
            continue

        masked, spans = mask_code_spans(line)
        p = masked.find("$$")
        if p < 0 or masked.lstrip().startswith("|"):  # no display math, or a table row
            emit(fix_inline(line, warnings, where))
            continue

        before = unmask(masked[:p], spans)
        rest = masked[p + 2:]
        q = rest.find("$$")
        if q >= 0:  # block opens and closes on this line
            body = unmask(rest[:q], spans)
            after = unmask(rest[q + 2:], spans)
            if before.strip() and after.strip() and not LIST_ITEM_RE.fullmatch(before):
                # $$x$$ in the middle of a sentence: make it inline math
                pending = before + "$" + body.strip() + "$" + after
                continue
            if LIST_ITEM_RE.fullmatch(before) and not after.strip():
                # "- $$x$$": a bullet holding only a formula renders fine on GitHub
                emit(before + "$$" + rewrite_tex(body, display=True, nested=True,
                                                 warnings=warnings, where=where) + "$$")
                continue
            indent = block_indent(before)
            if before.strip() and not LIST_ITEM_RE.fullmatch(before):
                emit(fix_inline(before.rstrip(), warnings, where))
            elif LIST_ITEM_RE.fullmatch(before):
                emit(before.rstrip())
            emit_block(indent, body, multiline=False)
            if after.strip():
                pending = indent + after.strip()
            continue

        # Multi-line block: collect until the closing $$
        body_lines = [unmask(rest, spans)]
        start = i
        closed = False
        while i < len(lines):
            nxt = lines[i]
            i += 1
            k = nxt.find("$$")
            if k >= 0:
                body_lines.append(nxt[:k])
                after = nxt[k + 2:]
                closed = True
                break
            body_lines.append(nxt)
        if not closed:
            warnings.append(f"{path}:{start}: unclosed $$ block; left unchanged")
            emit(line)
            out.extend(lines[start:])
            break
        indent = block_indent(before)
        if before.strip() and not LIST_ITEM_RE.fullmatch(before):
            emit(fix_inline(before.rstrip(), warnings, where))
        elif LIST_ITEM_RE.fullmatch(before):
            emit(before.rstrip())
        emit_block(indent, "\n".join(body_lines), multiline=True)
        if after.strip():
            pending = indent + after.strip()

    return "\n".join(out), warnings


def iter_markdown(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(p.rglob("*.md"))
        else:
            yield p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("paths", nargs="+", help="Markdown files or directories")
    parser.add_argument("--check", action="store_true", help="report only; exit 1 if changes are needed")
    args = parser.parse_args(argv)

    changed = 0
    for path in iter_markdown(args.paths):
        text = path.read_text()
        fixed, warnings = fix_text(text, str(path))
        for w in warnings:
            print(f"WARNING {w}")
        if fixed != text:
            changed += 1
            print(f"{'needs fix' if args.check else 'fixed'}: {path}")
            if not args.check:
                path.write_text(fixed)
    print(f"{changed} file(s) {'need fixing' if args.check else 'fixed'}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
