"""
Check Math Rendering: GitHub vs. MkDocs
=======================================

Render Markdown files with both GitHub (``gh api /markdown``, GFM mode) and
Python-Markdown + arithmatex (as configured for MkDocs), then compare the TeX
each renderer hands to MathJax.

arithmatex passes TeX through untouched, so any difference means GitHub
mangled the math (dropped escapes, emphasis inside formulas, unrecognized
blocks). Also reports ``$`` fragments GitHub left outside math regions.

Requires: an authenticated ``gh`` CLI, plus ``markdown`` and ``pymdown-extensions``
(``environment.yml`` / ``requirements-docs.txt``).

Usage:
    python scripts/docs/check_math_rendering.py README.md docs/
    python scripts/docs/check_math_rendering.py --dump-tex out.json docs/   # all TeX, for linting
"""

import argparse
import difflib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown

MKDOCS_EXTENSIONS = [
    "admonition", "attr_list", "def_list", "footnotes", "md_in_html", "tables",
    "pymdownx.arithmatex", "pymdownx.details", "pymdownx.superfences",
    "pymdownx.tabbed", "pymdownx.tasklist", "pymdownx.tilde", "pymdownx.caret",
    "pymdownx.mark",
]
MKDOCS_CONFIG = {"pymdownx.arithmatex": {"generic": True},
                 "pymdownx.tabbed": {"alternate_style": True}}

ARITHMATEX_RE = re.compile(r'<(span|div) class="arithmatex">(.*?)</\1>', re.S)
GITHUB_MATH_RE = re.compile(r'<math-renderer[^>]*>(.*?)</math-renderer>', re.S)


def normalize(tex):
    return re.sub(r"\s+", " ", tex).strip()


def mkdocs_math(text):
    out = markdown.markdown(text, extensions=MKDOCS_EXTENSIONS, extension_configs=MKDOCS_CONFIG)
    spans = []
    for _, body in ARITHMATEX_RE.findall(out):
        body = html.unescape(body).strip()
        body = re.sub(r"^\\[\[(]|\\[\])]$", "", body)
        spans.append(normalize(body))
    return spans


def github_render(text):
    payload = json.dumps({"text": text, "mode": "gfm"})
    res = subprocess.run(["gh", "api", "/markdown", "--input", "-"],
                         input=payload, capture_output=True, text=True, check=True)
    return res.stdout


def github_math(rendered):
    spans = []
    for body in GITHUB_MATH_RE.findall(rendered):
        body = html.unescape(body).strip()
        body = re.sub(r"^\$\$|\$\$$", "", body) if body.startswith("$$") else body.strip("$")
        spans.append(normalize(body))
    return spans


def stray_dollars(rendered):
    rest = GITHUB_MATH_RE.sub("", rendered)
    rest = re.sub(r"<pre.*?</pre>|<code[^>]*>.*?</code>", "", rest, flags=re.S)
    # a $ followed by a digit is currency, not math
    return [m.group(0) for m in re.finditer(r".{0,30}\$(?!\d).{0,30}", html.unescape(rest))]


def iter_markdown(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(p.rglob("*.md"))
        else:
            yield p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("paths", nargs="+", help="Markdown files or directories")
    parser.add_argument("--dump-tex", help="write every GitHub math span to this JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each mismatch")
    args = parser.parse_args(argv)

    bad_files = 0
    dump = []
    for path in iter_markdown(args.paths):
        text = path.read_text()
        if "$" not in text:
            continue
        rendered = github_render(text)
        gh, mk = github_math(rendered), mkdocs_math(text)
        stray = stray_dollars(rendered)
        altered, missing, extra = [], [], []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gh, mk, autojunk=False).get_opcodes():
            if op == "replace":
                altered += list(zip(gh[i1:i2], mk[j1:j2]))
                missing += mk[j1 + (i2 - i1):j2]
                extra += gh[i1 + (j2 - j1):i2]
            elif op == "delete":
                extra += gh[i1:i2]
            elif op == "insert":
                missing += mk[j1:j2]
        # `extra` = math GitHub found that Python-Markdown skipped (e.g. inside raw HTML): informational
        ok = not altered and not missing and not stray
        bad_files += not ok
        status = "ok  " if ok else "FAIL"
        print(f"{status} {path}  spans={len(gh)} altered={len(altered)} "
              f"missing={len(missing)} stray$={len(stray)} github-only={len(extra)}")
        if args.verbose:
            for g, m in altered[:10]:
                print(f"     github : {g[:110]}\n     mkdocs : {m[:110]}")
            for m in missing[:10]:
                print(f"     missing: {m[:110]}")
            for s in stray[:10]:
                print(f"     stray  : {s!r}")
            for g in extra[:10]:
                print(f"     gh-only: {g[:110]}")
        dump += [{"id": f"{path}#{k}", "tex": t, "display": False} for k, t in enumerate(gh)]

    if args.dump_tex:
        Path(args.dump_tex).write_text(json.dumps(dump))
    print(f"{bad_files} file(s) with math GitHub renders differently from MkDocs")
    return 1 if bad_files else 0


if __name__ == "__main__":
    sys.exit(main())
