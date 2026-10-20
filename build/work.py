#!/usr/bin/env python3
"""
Bundle a folder of Luau modules into ONE minified, copy-pasteable file.

    python bundle.py <src_dir> <entry_module> [-o bundle.luau] [options]

    python bundle.py src main
    python bundle.py src main -o out.luau --max-chars 100000 --wrap 200

What it does
  * Follows require() calls from the entry module and bundles only modules
    that are actually reachable (unused files are dropped).
  * Every module is wrapped in its own function (NOT a do-block: a `return`
    inside a do-block would exit the whole script). Each module runs once,
    result is cached, circular requires raise a clear error.
  * Strips comments and all unnecessary whitespace.
  * Renames local variables / parameters to short names (scope-aware).
    Field names, method names, table keys, globals and type annotations are
    never touched. If a module uses syntax the renamer can't handle safely,
    that module just gets whitespace-minified and a warning is printed.

Supported require styles
    require("./utils")  require("../lib/x")  require("@self/x")  require("x")
    require(script.Parent.utils)  require(script.utils)
    require(script.Parent:WaitForChild("utils"))   (or :FindFirstChild)

Safety checks (the build FAILS with file:line errors if any trip)
    * use of `script` anywhere except inside a supported require
    * getfenv / setfenv  (renaming + wrapping would silently break them)
    * unsupported or unresolvable require() forms (dynamic requires, etc.)
    * ambiguous module names, missing entry module
    * identifiers that clash with the bundle's internals (__r, __m, __c, __l)
    * anything you add with --forbid NAME (repeatable), e.g. --forbid loadstring
    * output larger than --max-chars (if given)
"""
import argparse
import itertools
import re
import string
import sys
from pathlib import Path

KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "if", "in", "local", "nil", "not", "or", "repeat", "return", "then",
    "true", "until", "while",
}
INTERNAL = {"__r", "__m", "__c", "__l"}
DEFAULT_FORBID = {"script", "getfenv", "setfenv"}


class BuildError(Exception):
    pass


class Unsupported(Exception):
    pass


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------
class Tok:
    __slots__ = ("k", "t", "line", "parts", "decl")

    def __init__(self, k, t, line=0, parts=None):
        self.k, self.t, self.line, self.parts, self.decl = k, t, line, parts, None


EOF = Tok("eof", "")
SYN = Tok("name", "_")  # synthetic "value just ended" token

NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUM_RE = re.compile(
    r"0[xX][0-9a-fA-F_]+|0[bB][01_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?"
)
LB_RE = re.compile(r"\[(=*)\[")
ATTR_RE = re.compile(r"@[A-Za-z_]\w*")
OPS3 = {"...", "..=", "//="}
OPS2 = {"==", "~=", "<=", ">=", "..", "::", "->", "+=", "-=", "*=", "/=", "%=", "^=", "//"}
OPS1 = set("+-*/%^#&|~<>=(){}[];:,.?")


class Lexer:
    def __init__(self, src, fname):
        self.s, self.i, self.line, self.f = src, 0, 1, fname

    def fail(self, msg):
        raise BuildError(f"{self.f}:{self.line}: {msg}")

    def lex(self, interp=False):
        S, n, out, depth = self.s, len(self.s), [], 0
        while self.i < n:
            i = self.i
            c = S[i]
            if c == "\n":
                self.line += 1
                self.i += 1
                continue
            if c in " \t\r\f\v":
                self.i += 1
                continue
            if c == "-" and S.startswith("--", i):
                m = LB_RE.match(S, i + 2)
                if m:
                    close = "]" + m.group(1) + "]"
                    j = S.find(close, m.end())
                    if j < 0:
                        self.fail("unterminated long comment")
                    j += len(close)
                    self.line += S.count("\n", i, j)
                    self.i = j
                else:
                    j = S.find("\n", i)
                    self.i = n if j < 0 else j
                continue
            m = NAME_RE.match(S, i)
            if m:
                w = m.group(0)
                out.append(Tok("kw" if w in KEYWORDS else "name", w, self.line))
                self.i = m.end()
                continue
            if c.isdigit() or (c == "." and S[i + 1:i + 2].isdigit()):
                m = NUM_RE.match(S, i)
                out.append(Tok("num", m.group(0), self.line))
                self.i = m.end()
                continue
            if c in "\"'":
                j, start = i + 1, self.line
                while True:
                    if j >= n:
                        self.fail("unterminated string")
                    ch = S[j]
                    if ch == "\\":
                        if S[j + 1:j + 2] == "\n":
                            self.line += 1
                        j += 2
                    elif ch == c:
                        break
                    elif ch == "\n":
                        self.fail("unterminated string")
                    else:
                        j += 1
                out.append(Tok("str", S[i:j + 1], start))
                self.i = j + 1
                continue
            if c == "[":
                m = LB_RE.match(S, i)
                if m:
                    close = "]" + m.group(1) + "]"
                    j = S.find(close, m.end())
                    if j < 0:
                        self.fail("unterminated long string")
                    j += len(close)
                    out.append(Tok("str", S[i:j], self.line))
                    self.line += S.count("\n", i, j)
                    self.i = j
                    continue
            if c == "`":
                start = self.line
                self.i += 1
                parts, buf = [], []
                while True:
                    if self.i >= n:
                        self.fail("unterminated interpolated string")
                    ch = S[self.i]
                    if ch == "\\":
                        buf.append(S[self.i:self.i + 2])
                        self.i += 2
                    elif ch == "`":
                        self.i += 1
                        break
                    elif ch == "{":
                        parts.append(("s", "".join(buf)))
                        buf = []
                        self.i += 1
                        parts.append(("e", self.lex(interp=True)))
                    else:
                        if ch == "\n":
                            self.line += 1
                        buf.append(ch)
                        self.i += 1
                parts.append(("s", "".join(buf)))
                out.append(Tok("interp", "", start, parts))
                continue
            if c == "@":
                m = ATTR_RE.match(S, i)
                if not m:
                    self.fail("unsupported attribute syntax")
                out.append(Tok("attr", m.group(0), self.line))
                self.i = m.end()
                continue
            if S[i:i + 3] in OPS3:
                op = S[i:i + 3]
            elif S[i:i + 2] in OPS2:
                op = S[i:i + 2]
            elif c in OPS1:
                op = c
            else:
                self.fail(f"unexpected character {c!r}")
            self.i += len(op)
            if interp:
                if op == "{":
                    depth += 1
                elif op == "}":
                    if depth == 0:
                        return out
                    depth -= 1
            out.append(Tok("op", op, self.line))
        if interp:
            self.fail("unterminated interpolation")
        return out


def lex(src, fname):
    return Lexer(src, fname).lex()


# --------------------------------------------------------------------------
# Token helpers
# --------------------------------------------------------------------------
def G(toks, j):
    return toks[j] if 0 <= j < len(toks) else EOF


def isop(t, s):
    return t is not None and t.k == "op" and t.t == s


def is_value_end(t):
    if t is None:
        return False
    if t.k in ("name", "num", "str", "interp", "ref"):
        return True
    if t.k == "op":
        return t.t in (")", "]", "}", "...")
    return t.k == "kw" and t.t in ("end", "true", "false", "nil")


def match_close(toks, i):
    o = toks[i].t
    c = {"(": ")", "{": "}", "[": "]"}[o]
    d = 0
    for j in range(i, len(toks)):
        t = toks[j]
        if t.k == "op":
            if t.t == o:
                d += 1
            elif t.t == c:
                d -= 1
                if d == 0:
                    return j
    raise Unsupported("unbalanced brackets")


def match_angle(toks, i):
    d = 0
    for j in range(i, len(toks)):
        t = toks[j]
        if t.k == "op":
            if t.t == "<":
                d += 1
            elif t.t == ">":
                d -= 1
                if d == 0:
                    return j
            elif t.t in ("<=", ">="):
                raise Unsupported("ambiguous generic")
    raise Unsupported("unbalanced <>")


def skip_type(toks, i):
    """Return the index just after the type expression starting at i."""
    i = skip_atom(toks, i)
    while G(toks, i).k == "op" and G(toks, i).t in ("|", "&"):
        i = skip_atom(toks, i + 1)
    return i


def skip_atom(toks, i):
    t = G(toks, i)
    while t.k == "op" and t.t in ("|", "&"):
        i += 1
        t = G(toks, i)
    if isop(t, "..."):
        return skip_atom(toks, i + 1)
    if isop(t, "("):
        i = match_close(toks, i) + 1
        if isop(G(toks, i), "->"):
            i = skip_type(toks, i + 1)
    elif isop(t, "{"):
        i = match_close(toks, i) + 1
    elif isop(t, "<"):
        j = match_angle(toks, i) + 1
        if not isop(G(toks, j), "("):
            raise Unsupported("generic type")
        i = match_close(toks, j) + 1
        if not isop(G(toks, i), "->"):
            raise Unsupported("generic type")
        i = skip_type(toks, i + 1)
    elif t.k == "name" and t.t == "typeof":
        raise Unsupported("typeof() in a type")
    elif t.k in ("name", "str") or (t.k == "kw" and t.t in ("nil", "true", "false")):
        i += 1
        while isop(G(toks, i), ".") and G(toks, i + 1).k == "name":
            i += 2
        if isop(G(toks, i), "<"):
            i = match_angle(toks, i) + 1
    else:
        raise Unsupported("unrecognised type syntax")
    while isop(G(toks, i), "?"):
        i += 1
    return i


# --------------------------------------------------------------------------
# Scope-aware local renamer
# --------------------------------------------------------------------------
BLOCKS = {"root", "func", "do", "if", "repeat"}
EXPR_KW = {"return", "and", "or", "not", "in", "until", "while", "elseif"}


class Decl:
    __slots__ = ("name", "visible", "new")

    def __init__(self, name):
        self.name, self.visible, self.new = name, [], None


class Block:
    __slots__ = ("kind", "map", "pending", "exprif", "until_open")

    def __init__(self, kind):
        self.kind, self.map, self.pending = kind, {}, []
        self.exprif, self.until_open = 0, False


def is_type_stmt(toks, i):
    t = toks[i]
    if t.t == "type":
        return G(toks, i + 1).k == "name" and (isop(G(toks, i + 2), "=") or isop(G(toks, i + 2), "<"))
    if t.t == "export":
        return (G(toks, i + 1).k == "name" and G(toks, i + 1).t == "type"
                and G(toks, i + 2).k == "name"
                and (isop(G(toks, i + 3), "=") or isop(G(toks, i + 3), "<")))
    return False


def analyze(toks):
    stack = [Block("root")]
    decls, glob, bound, loops, skip = [], set(INTERNAL), [], [], set()
    prev, else_expr, n, i = None, False, len(toks), 0

    def top():
        return stack[-1]

    def active():
        out = []
        for b in stack:
            if b.kind in BLOCKS:
                out.extend(b.map.values())
        return out

    def activate(block, ds):
        for d in ds:
            d.visible = active()
            decls.append(d)
            block.map[d.name] = d

    def flush(block):
        if block.pending:
            ds, block.pending = block.pending, []
            activate(block, ds)

    def boundary():
        while True:
            b = top()
            flush(b)
            if b.kind == "repeat" and b.until_open:
                stack.pop()
                continue
            break

    def new_decl(tok):
        d = Decl(tok.t)
        tok.decl = d
        bound.append(tok)
        return d

    def use(tok):
        for b in reversed(stack):
            if b.kind in BLOCKS and tok.t in b.map:
                tok.decl = b.map[tok.t]
                bound.append(tok)
                return
        glob.add(tok.t)

    def resolve_interp(tok):
        for kind, val in tok.parts:
            if kind != "e":
                continue
            for idx, x in enumerate(val):
                p = val[idx - 1] if idx else None
                if x.k == "name" and not (isop(p, ".") or isop(p, ":")):
                    use(x)
                elif x.k == "interp":
                    resolve_interp(x)
                elif x.k == "kw" and x.t == "function":
                    raise Unsupported("function inside interpolated string")

    while i < n:
        t = toks[i]
        k, v = t.k, t.t
        flag_else = False
        if k == "name":
            if v in ("type", "export") and is_type_stmt(toks, i):
                boundary()
                j = i + (2 if v == "type" else 3)
                if isop(G(toks, j), "<"):
                    j = match_angle(toks, j) + 1
                if not isop(G(toks, j), "="):
                    raise Unsupported("type statement")
                i = skip_type(toks, j + 1)
                prev = SYN
                continue
            if is_value_end(prev):
                boundary()
            if isop(prev, ".") or isop(prev, ":"):
                pass
            elif (top().kind == "{" and (isop(prev, "{") or isop(prev, ",") or isop(prev, ";"))
                  and isop(G(toks, i + 1), "=")):
                pass
            else:
                use(t)
        elif k == "num" or k == "ref" or k == "attr":
            if is_value_end(prev):
                boundary()
        elif k == "interp":
            resolve_interp(t)
        elif k == "op":
            if v == ";":
                boundary()
            elif v in ("(", "{", "["):
                stack.append(Block(v))
            elif v in (")", "}", "]"):
                if top().kind != {")": "(", "}": "{", "]": "["}[v]:
                    raise Unsupported("bracket mismatch")
                stack.pop()
            elif v == "::":
                i = skip_type(toks, i + 1)
                prev = SYN
                continue
            elif v in (":", "->", "|", "&", "?"):
                if v != ":":
                    raise Unsupported(f"unexpected {v!r}")
        elif k == "kw":
            if v == "local":
                boundary()
                j = i + 1
                if G(toks, j).k == "kw" and G(toks, j).t == "function":
                    nm = G(toks, j + 1)
                    if nm.k != "name":
                        raise Unsupported("local function")
                    d = new_decl(nm)
                    skip.add(j + 1)
                    activate(top(), [d])
                    prev = t
                    i = j
                    continue
                ds = []
                while True:
                    tok = G(toks, j)
                    if tok.k != "name":
                        raise Unsupported("local declaration")
                    if isop(G(toks, j + 1), "<"):
                        raise Unsupported("attribute on local")
                    ds.append(new_decl(tok))
                    j += 1
                    if isop(G(toks, j), ":"):
                        j = skip_type(toks, j + 1)
                    if isop(G(toks, j), ","):
                        j += 1
                        continue
                    break
                if isop(G(toks, j), "="):
                    top().pending = ds
                else:
                    activate(top(), ds)
                prev = SYN
                i = j
                continue
            elif v == "function":
                if is_value_end(prev):
                    boundary()
                j = i + 1
                if G(toks, j).k == "name":
                    if j in skip:
                        j += 1
                    else:
                        use(G(toks, j))
                        j += 1
                        while (isop(G(toks, j), ".") or isop(G(toks, j), ":")) and G(toks, j + 1).k == "name":
                            j += 2
                if isop(G(toks, j), "<"):
                    j = match_angle(toks, j) + 1
                if not isop(G(toks, j), "("):
                    raise Unsupported("function header")
                b = Block("func")
                stack.append(b)
                j += 1
                params = []
                while not isop(G(toks, j), ")"):
                    tok = G(toks, j)
                    if tok.k == "name":
                        params.append(new_decl(tok))
                        j += 1
                        if isop(G(toks, j), ":"):
                            j = skip_type(toks, j + 1)
                    elif isop(tok, "..."):
                        j += 1
                        if isop(G(toks, j), ":"):
                            j = skip_type(toks, j + 1)
                    elif isop(tok, ","):
                        j += 1
                    else:
                        raise Unsupported("parameter list")
                j += 1
                if isop(G(toks, j), ":"):
                    j = skip_type(toks, j + 1)
                activate(b, params)
                prev = SYN
                i = j
                continue
            elif v in ("return", "break"):
                boundary()
            elif v == "if":
                expr = top().exprif > 0 or (
                    prev is not None and (
                        (prev.k == "op" and prev.t not in (")", "]", "}", "...", ";"))
                        or (prev.k == "kw" and prev.t in EXPR_KW)
                        or (prev.k == "kw" and prev.t == "else" and else_expr)
                    )
                )
                if expr:
                    top().exprif += 1
                else:
                    boundary()
                    stack.append(Block("if"))
            elif v == "elseif":
                if top().exprif == 0:
                    boundary()
                    top().map = {}
            elif v == "else":
                if top().exprif > 0:
                    top().exprif -= 1
                    flag_else = True
                else:
                    boundary()
                    top().map = {}
            elif v == "for":
                boundary()
                j, ds = i + 1, []
                while True:
                    tok = G(toks, j)
                    if tok.k != "name" or isop(G(toks, j + 1), ":"):
                        raise Unsupported("for loop variables")
                    ds.append(new_decl(tok))
                    skip.add(j)
                    j += 1
                    if isop(G(toks, j), ","):
                        j += 1
                        continue
                    break
                if not (isop(G(toks, j), "=") or (G(toks, j).k == "kw" and G(toks, j).t == "in")):
                    raise Unsupported("for loop")
                loops.append((len(stack), ds))
                prev = SYN
                i = j
                continue
            elif v == "while":
                boundary()
                loops.append((len(stack), None))
            elif v == "do":
                if loops and loops[-1][0] == len(stack):
                    ds = loops.pop()[1]
                    b = Block("do")
                    stack.append(b)
                    if ds:
                        activate(b, ds)
                else:
                    boundary()
                    stack.append(Block("do"))
            elif v == "repeat":
                boundary()
                stack.append(Block("repeat"))
            elif v == "until":
                b = top()
                if b.kind != "repeat":
                    raise Unsupported("until without repeat")
                flush(b)
                b.until_open = True
            elif v == "end":
                boundary()
                if top().kind not in ("func", "do", "if"):
                    raise Unsupported("unbalanced end")
                stack.pop()
        else_expr = flag_else
        prev = t
        i += 1

    boundary()
    if len(stack) != 1:
        raise Unsupported("unbalanced blocks")
    return decls, glob, bound


def _candidates():
    first = string.ascii_letters + "_"
    rest = first + string.digits
    for c in first:
        yield c
    for length in range(2, 5):
        for combo in itertools.product(first, *([rest] * (length - 1))):
            yield "".join(combo)


def rename_locals(toks):
    """Rename locals in-place. Raises Unsupported if it can't do so safely."""
    try:
        decls, glob, bound = analyze(toks)
    except IndexError:
        raise Unsupported("parse error")
    for d in decls:
        used = {x.new for x in d.visible if x.new}
        for nm in _candidates():
            if nm not in KEYWORDS and nm not in glob and nm not in used:
                d.new = nm
                break
    for tok in bound:
        tok.t = tok.decl.new
    return len(decls)


# --------------------------------------------------------------------------
# Module collection, require resolution, safety checks
# --------------------------------------------------------------------------
class Mod:
    def __init__(self, key, parts, is_init, src, path):
        self.key, self.parts, self.is_init = key, parts, is_init
        self.src, self.path = src, path
        self.dirparts = parts if is_init else parts[:-1]
        self.toks, self.deps = [], []


def collect(src_dir):
    mods = {}
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file() or p.suffix not in (".luau", ".lua"):
            continue
        parts = list(p.relative_to(src_dir).with_suffix("").parts)
        is_init = parts[-1] == "init" and len(parts) > 1
        if is_init:
            parts = parts[:-1]
        key = "/".join(parts)
        if key in mods:
            raise BuildError(f"duplicate module '{key}' ({mods[key].path} and {p})")
        mods[key] = Mod(key, parts, is_init, p.read_text(encoding="utf-8-sig"), p)
    return mods


def _strip_ext(s):
    for e in (".luau", ".lua"):
        if s.endswith(e):
            return s[: -len(e)]
    return s


def resolve(mod, kind, data, mods):
    cands, last = [], None
    if kind == "script":
        base = list(mod.parts)
        for s in data:
            if s == "Parent":
                if not base:
                    return None
                base.pop()
            else:
                base.append(s)
        cands.append("/".join(base))
        last = data[-1] if data and data[-1] != "Parent" else None
    else:
        parts = [p for p in _strip_ext(data).split("/") if p != ""]
        if not parts:
            return None
        last = parts[-1] if parts[-1] not in (".", "..", "@self") else None
        if parts[0] == "@self":
            cands.append("/".join(list(mod.parts) + parts[1:]))
        elif parts[0] in (".", ".."):
            bases = [mod.dirparts] + ([mod.parts[:-1]] if mod.is_init else [])
            for base in bases:
                b = list(base)
                for p in parts:
                    if p == ".":
                        continue
                    if p == "..":
                        if not b:
                            b = None
                            break
                        b.pop()
                    else:
                        b.append(p)
                if b is not None:
                    cands.append("/".join(b))
        else:
            cands.append("/".join(parts))
            cands.append("/".join(mod.dirparts + parts))
    for c in cands:
        if c in mods:
            return c
    if last:
        m = [k for k in mods if k.split("/")[-1] == last]
        if len(m) == 1:
            return m[0]
        if len(m) > 1:
            raise BuildError("ambiguous: " + ", ".join(m))
    return None


def rewrite_requires(mod, mods, errors):
    toks, out, i = mod.toks, [], 0
    while i < len(toks):
        t = toks[i]
        prev = out[-1] if out else None
        if t.k == "name" and t.t == "require" and not (isop(prev, ".") or isop(prev, ":")):
            target, end = None, i
            try:
                if isop(G(toks, i + 1), "("):
                    a = G(toks, i + 2)
                    if a.k == "str" and isop(G(toks, i + 3), ")"):
                        if a.t[0] in "\"'" and "\\" not in a.t:
                            target, end = ("str", a.t[1:-1]), i + 3
                    elif a.k == "name" and a.t == "script":
                        k, segs = i + 3, []
                        while True:
                            x, y = G(toks, k), G(toks, k + 1)
                            if isop(x, ".") and y.k == "name":
                                segs.append(y.t)
                                k += 2
                            elif (isop(x, ":") and y.k == "name"
                                  and y.t in ("WaitForChild", "FindFirstChild")
                                  and isop(G(toks, k + 2), "(") and G(toks, k + 3).k == "str"
                                  and G(toks, k + 3).t[0] in "\"'"):
                                segs.append(G(toks, k + 3).t[1:-1])
                                k = match_close(toks, k + 2) + 1
                            else:
                                break
                        if isop(G(toks, k), ")"):
                            target, end = ("script", segs), k
            except Unsupported:
                target = None
            if target is None:
                errors.append(f"{mod.path}:{t.line}: unsupported require() form "
                              "(only string paths and script.Parent.X chains can be bundled)")
                i += 1
                out.append(t)
                continue
            try:
                key = resolve(mod, target[0], target[1], mods)
            except BuildError as e:
                errors.append(f"{mod.path}:{t.line}: {e}")
                key = None
            if key is None:
                if not errors or not errors[-1].startswith(f"{mod.path}:{t.line}:"):
                    errors.append(f"{mod.path}:{t.line}: cannot resolve require target {target[1]!r}")
            else:
                if key not in mod.deps:
                    mod.deps.append(key)
                out.append(Tok("ref", key, t.line))
            i = end + 1
            continue
        out.append(t)
        i += 1
    mod.toks = out


def safety_scan(mod, toks, forbid, errors):
    for i, t in enumerate(toks):
        if t.k == "interp":
            for kind, val in t.parts:
                if kind == "e":
                    safety_scan(mod, val, forbid, errors)
            continue
        if t.k != "name":
            continue
        p, nx = G(toks, i - 1) if i else None, G(toks, i + 1)
        if isop(p, ".") or isop(p, ":"):
            continue
        if isop(nx, "=") and (isop(p, "{") or isop(p, ",") or isop(p, ";")):
            continue
        if t.t in forbid:
            errors.append(f"{mod.path}:{t.line}: '{t.t}' is not allowed in the bundle")
        elif t.t in INTERNAL:
            errors.append(f"{mod.path}:{t.line}: identifier '{t.t}' is reserved by the bundler")
        elif t.t == "require":
            errors.append(f"{mod.path}:{t.line}: 'require' used in a way that can't be bundled")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def tok_str(t):
    if t.k == "interp":
        out = ["`"]
        for kind, val in t.parts:
            if kind == "s":
                out.append(val)
            else:
                s = render(val)
                out.append("{" + (" " if s.startswith("{") else "") + s + "}")
        out.append("`")
        return "".join(out)
    return t.t


def need_space(a, ak, b):
    ca, cb = a[-1], b[0]
    if (ca.isalnum() or ca == "_") and (cb.isalnum() or cb == "_"):
        return True
    if ca == "-" and cb == "-":
        return True
    if ca == "." and (cb == "." or cb.isdigit()):
        return True
    if ak == "num" and cb == ".":
        return True
    if ca == "[" and cb in "[=":
        return True
    return False


def render(toks, wrap=0):
    out, line_len, last, lastk = [], 0, None, None
    for t in toks:
        s = tok_str(t)
        sep = " " if last is not None and need_space(last, lastk, s) else ""
        if wrap and line_len > 0 and line_len + len(sep) + len(s) > wrap:
            out.append("\n")
            line_len, sep = 0, ""
        out.append(sep + s)
        line_len += len(sep) + len(s)
        last, lastk = s, t.k
    return "".join(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Bundle Luau modules into one minified file.")
    ap.add_argument("src", help="source folder")
    ap.add_argument("entry", help="entry module, e.g. main or lib/app")
    ap.add_argument("-o", "--out", default="bundle.luau")
    ap.add_argument("--no-rename", action="store_true", help="only strip whitespace/comments")
    ap.add_argument("--wrap", type=int, default=0, metavar="N", help="insert line breaks every ~N chars")
    ap.add_argument("--max-chars", type=int, default=0, metavar="N", help="fail if output is larger")
    ap.add_argument("--forbid", action="append", default=[], metavar="NAME",
                    help="extra identifier to forbid (repeatable)")
    args = ap.parse_args()
    forbid = DEFAULT_FORBID | set(args.forbid)

    src_dir = Path(args.src)
    if not src_dir.is_dir():
        sys.exit(f"error: {src_dir} is not a folder")
    try:
        mods = collect(src_dir)
    except BuildError as e:
        sys.exit(f"error: {e}")
    if not mods:
        sys.exit("error: no .luau/.lua files found")

    errors, warnings = [], []
    for m in mods.values():
        try:
            m.toks = lex(m.src, str(m.path))
        except BuildError as e:
            errors.append(str(e))
    if errors:
        sys.exit("\n".join("error: " + e for e in errors))

    entry_mod = Mod("", [], False, "", "")
    try:
        entry = resolve(entry_mod, "str", args.entry, mods)
    except BuildError as e:
        sys.exit(f"error: entry {e}")
    if entry is None:
        sys.exit(f"error: entry module '{args.entry}' not found. Modules: {', '.join(mods)}")

    # Only process modules reachable from the entry.
    order, seen, queue = [], {entry}, [entry]
    while queue:
        key = queue.pop(0)
        order.append(key)
        m = mods[key]
        rewrite_requires(m, mods, errors)
        safety_scan(m, m.toks, forbid, errors)
        for d in m.deps:
            if d not in seen:
                seen.add(d)
                queue.append(d)
    if errors:
        sys.exit("\n".join("error: " + e for e in errors) + "\n\nbuild failed.")

    unused = [k for k in mods if k not in seen]
    if unused:
        warnings.append("not required from entry, left out: " + ", ".join(unused))

    # Static cycle warning
    state = {}

    def dfs(k, path):
        state[k] = 1
        for d in mods[k].deps:
            if state.get(d) == 1:
                cyc = path[path.index(d):] + [d] if d in path else [k, d]
                warnings.append("circular require: " + " -> ".join(cyc)
                                + " (only OK if one side requires lazily inside a function)")
            elif d not in state:
                dfs(d, path + [d])
        state[k] = 2

    dfs(entry, [entry])

    index = {k: i + 1 for i, k in enumerate(order)}
    original = sum(len(mods[k].src) for k in order)

    body = lex("local __r local __m={", "<hdr>")
    renamed = 0
    for key in order:
        m = mods[key]
        if not args.no_rename:
            try:
                renamed += rename_locals(m.toks)
            except Unsupported as e:
                warnings.append(f"{m.path}: variable renaming skipped ({e}); whitespace only")
        body += lex("function()", "<hdr>")
        for t in m.toks:
            if t.k == "ref":
                body += [Tok("name", "__r"), Tok("op", "("), Tok("num", str(index[t.t])), Tok("op", ")")]
            else:
                body.append(t)
        body += lex("end,", "<hdr>")
    body += lex(
        "} local __c,__l={},{} "
        "__r=function(n)local v=__c[n] "
        "if v==__l then error(\"circular require: module #\"..n)end "
        "if v==nil then __c[n]=__l v=__m[n]() if v==nil then v=true end __c[n]=v end "
        f"return v end return __r({index[entry]})",
        "<hdr>",
    )
    result = render(body, args.wrap)

    if args.max_chars and len(result) > args.max_chars:
        sys.exit(f"error: output is {len(result)} chars, over --max-chars {args.max_chars}")

    Path(args.out).write_text(result + "\n", encoding="utf-8")
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    print(f"wrote {args.out}: {len(order)} modules, {original} -> {len(result)} chars "
          f"({100 - 100 * len(result) // max(original, 1)}% smaller), {renamed} variables renamed")


if __name__ == "__main__":
    try:
        main()
    except BuildError as e:
        sys.exit(f"error: {e}")