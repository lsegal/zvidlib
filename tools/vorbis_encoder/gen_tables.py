#!/usr/bin/env python3
"""Generate the Rust static tables of the Vorbis encoder port from libvorbis 1.3.7.

Usage (re-runnable, idempotent):

    python3 tools/vorbis_encoder/gen_tables.py path/to/libvorbis-1.3.7

The argument is an unpacked libvorbis 1.3.7 source release
(https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.gz).
Output goes to crates/zvidlib-vorbis-encoder/src/vorbis_encoder/tables/*.rs (files whose name starts with
`gen_`); every generated file carries a header naming its sources.

What it does:

* tokenizes and parses the C headers/sources that hold the encoder's static
  data (lib/modes/*.h, lib/books/{coupled,uncoupled,floor}/*.h, lib/masking.h,
  the vwin tables of lib/window.c, the FLOOR1_fromdB_LOOKUP table of
  lib/floor1.c and the small templates at the top of lib/vorbisenc.c);
* evaluates every initializer with C semantics (aggregate brace elision,
  zero-fill of missing members, integer arithmetic such as the famous
  `-7  -3` missing-comma in psych_44.h, double->int truncation, correctly
  rounded decimal->float/double conversion);
* walks the object graph starting from vorbisenc.c's `setup_list` (minus the
  5.1 surround template, which needs more than two channels) and emits only the
  objects reachable from it. The `book_aux_managed`/`books_base_managed`
  members of the residue templates are only used by bitrate-managed (ABR/CBR)
  mode, which this port does not implement, so they are not followed;
* writes Rust `static` items using the struct definitions of
  crates/zvidlib-vorbis-encoder/src/vorbis_encoder/tables/types.rs.

Float values are printed as the shortest decimal that round-trips to the same
f32/f64, so the Rust tables are bit-identical to what a C compiler produces.
Needs Python 3.8+ and numpy.
"""
import os
import re
import sys
from fractions import Fraction

import numpy as np

# The repository root: this file is tools/vorbis_encoder/gen_tables.py.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "crates", "zvidlib-vorbis-encoder", "src", "vorbis_encoder", "tables")

# --------------------------------------------------------------------------
# Tokenizer / parser
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"""
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlLuU]*)
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<punct>[{}\[\]();,=&*+\-/.])
  | (?P<ws>\s+)
    """,
    re.VERBOSE,
)


def strip_comments_and_pp(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def tokenize(text):
    toks = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise SyntaxError("cannot tokenize at: %r" % text[pos : pos + 40])
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        toks.append((kind, m.group(kind)))
    return toks


class Num:
    """A numeric constant with C typing: kind is 'int', 'double' or 'float'."""

    def __init__(self, kind, text=None, value=None):
        self.kind = kind
        self.text = text  # original literal (for correctly rounded conversion)
        self.value = value  # int or Fraction

    def __repr__(self):
        return "Num(%s,%s)" % (self.kind, self.text or self.value)


def parse_num(text):
    t = text.rstrip("uUlL")
    if re.fullmatch(r"\d+", t):
        return Num("int", text, int(t))
    if t[-1] in "fF":
        return Num("float", t[:-1], Fraction(t[:-1]))
    return Num("double", t, Fraction(t))


class Ref:
    """Reference to a named object (`foo` or `&foo`)."""

    def __init__(self, name, addr):
        self.name = name
        self.addr = addr

    def __repr__(self):
        return ("&" if self.addr else "") + self.name


NULL = "NULL"


class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self, k=0):
        if self.i + k < len(self.toks):
            return self.toks[self.i + k]
        return (None, None)

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, v):
        t = self.next()
        if t[1] != v:
            raise SyntaxError("expected %r got %r at token %d" % (v, t, self.i))
        return t

    # declarations ---------------------------------------------------------
    def declarations(self):
        decls = []
        while self.i < len(self.toks):
            if self.peek()[1] == "static" or self.peek()[1] == "const":
                d = self.declaration()
                if d is not None:
                    decls.append(d)
            else:
                # skip anything else (function bodies are never in our inputs
                # except window.c/floor1.c/vorbisenc.c, handled via extraction)
                self.next()
        return decls

    def declaration(self):
        quals = []
        while self.peek()[1] in ("static", "const"):
            quals.append(self.next()[1])
        typ = self.next()[1]
        if typ == "struct":
            typ = self.next()[1]
        ptr = 0
        while self.peek()[1] in ("*", "const"):
            if self.next()[1] == "*":
                ptr += 1
        name = self.next()[1]
        dims = []
        while self.peek()[1] == "[":
            self.next()
            if self.peek()[1] == "]":
                dims.append(None)
            else:
                dims.append(self.expr())
            self.expect("]")
        if self.peek()[1] != "=":
            # declaration without initializer (prototype etc.): skip to ';'
            while self.next()[1] != ";":
                pass
            return None
        self.expect("=")
        init = self.initializer()
        self.expect(";")
        return {"type": typ, "ptr": ptr, "name": name, "dims": dims, "init": init}

    def initializer(self):
        if self.peek()[1] == "{":
            self.next()
            items = []
            while self.peek()[1] != "}":
                items.append(self.initializer())
                if self.peek()[1] == ",":
                    self.next()
            self.expect("}")
            return items
        return self.expr()

    # expressions ----------------------------------------------------------
    def expr(self):
        v = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            v = binop(op, v, self.term())
        return v

    def term(self):
        v = self.unary()
        while self.peek()[1] in ("*", "/"):
            op = self.next()[1]
            v = binop(op, v, self.unary())
        return v

    def unary(self):
        t = self.peek()
        if t[1] == "-":
            self.next()
            return negate(self.unary())
        if t[1] == "+":
            self.next()
            return self.unary()
        if t[1] == "&":
            self.next()
            name = self.next()[1]
            return Ref(name, True)
        if t[1] == "(":
            # cast "(type *)expr" or parenthesized expression
            if self.peek(1)[0] == "id" and self.peek(2)[1] in ("*", ")") and self.peek(1)[1] in (
                "char",
                "long",
                "int",
                "float",
                "double",
                "static_codebook",
            ):
                self.next()
                while self.next()[1] != ")":
                    pass
                return self.unary()
            self.next()
            v = self.expr()
            self.expect(")")
            return v
        if t[0] == "num":
            self.next()
            return parse_num(t[1])
        if t[0] == "id":
            self.next()
            if t[1] == "NULL":
                return NULL
            return Ref(t[1], False)
        raise SyntaxError("unexpected token %r" % (t,))


def negate(v):
    if not isinstance(v, Num):
        raise SyntaxError("cannot negate %r" % v)
    text = v.text
    if text is not None:
        text = text[1:] if text.startswith("-") else "-" + text
    return Num(v.kind, text, -v.value)


def binop(op, a, b):
    if not (isinstance(a, Num) and isinstance(b, Num)):
        raise SyntaxError("non numeric binop")
    if a.kind != "int" or b.kind != "int":
        # libvorbis tables only contain integer arithmetic; float arithmetic
        # would need C evaluation-order rounding, so refuse it loudly.
        raise SyntaxError("floating point arithmetic in initializer: %r %s %r" % (a, op, b))
    x, y = a.value, b.value
    if op == "+":
        r = x + y
    elif op == "-":
        r = x - y
    elif op == "*":
        r = x * y
    else:
        r = int(x / y)  # C truncating division (values are small)
    return Num("int", None, r)


def parse_file(path, only_names=None):
    with open(path, encoding="latin-1") as f:
        text = f.read()
    text = strip_comments_and_pp(text)
    if only_names is not None:
        # extract just the given top-level declarations from a .c file
        chunks = []
        for name in only_names:
            m = re.search(r"(static\s+(?:const\s+)?\w+\s*\**\s*%s\s*(\[[^\]]*\])*\s*=\s*\{)" % re.escape(name), text)
            if not m:
                raise KeyError("%s not found in %s" % (name, path))
            start = m.start()
            depth = 0
            j = m.end() - 1
            while True:
                c = text[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            end = text.index(";", j) + 1
            chunks.append(text[start:end])
        text = "\n".join(chunks)
    return Parser(tokenize(text)).declarations()


# --------------------------------------------------------------------------
# C type schema (mirrors libvorbis structs; Rust names in types.rs)
# --------------------------------------------------------------------------

P_BANDS = 17
P_NOISECURVES = 3
NOISE_COMPAND_LEVELS = 40
PACKETBLOBS = 15
VE_BANDS = 7

# field type grammar:
#   'i32' | 'f32' | 'f64' | 'u8'          scalars
#   ('arr', T, N)                          fixed array
#   ('ref', T)                             pointer to a single object (Rust &'static T)
#   ('slice', T)                           pointer to array (Rust &'static [T])
#   ('opt', X)                             nullable pointer (Rust Option<X>)
#   ('skip', X)                            member parsed but not emitted (managed-only)
#   ('struct', name)

STRUCTS = {
    "static_codebook": ("StaticCodebook", [
        ("dim", "i32"), ("entries", "i32"), ("lengthlist", ("slice", "u8")),
        ("maptype", "i32"), ("q_min", "i32"), ("q_delta", "i32"), ("q_quant", "i32"),
        ("q_sequencep", "i32"), ("quantlist", ("opt", ("slice", "i32"))), ("allocedp", ("skip", "i32")),
    ]),
    "vorbis_info_floor1": ("InfoFloor1", [
        ("partitions", "i32"), ("partitionclass", ("arr", "i32", 31)),
        ("class_dim", ("arr", "i32", 16)), ("class_subs", ("arr", "i32", 16)),
        ("class_book", ("arr", "i32", 16)), ("class_subbook", ("arr", ("arr", "i32", 8), 16)),
        ("mult", "i32"), ("postlist", ("arr", "i32", 65)),
        ("maxover", "f32"), ("maxunder", "f32"), ("maxerr", "f32"),
        ("twofitweight", "f32"), ("twofitatten", "f32"), ("n", "i32"),
    ]),
    "vorbis_info_residue0": ("InfoResidue0", [
        ("begin", "i32"), ("end", "i32"), ("grouping", "i32"), ("partitions", "i32"),
        ("partvals", "i32"), ("groupbook", "i32"), ("secondstages", ("arr", "i32", 64)),
        ("booklist", ("arr", "i32", 512)), ("classmetric1", ("arr", "i32", 64)),
        ("classmetric2", ("arr", "i32", 64)),
    ]),
    "vorbis_info_mapping0": ("InfoMapping0", [
        ("submaps", "i32"), ("chmuxlist", ("arr", "i32", 256)), ("floorsubmap", ("arr", "i32", 16)),
        ("residuesubmap", ("arr", "i32", 16)), ("coupling_steps", "i32"),
        ("coupling_mag", ("arr", "i32", 256)), ("coupling_ang", ("arr", "i32", 256)),
    ]),
    "vorbis_info_mode": ("InfoMode", [
        ("blockflag", "i32"), ("windowtype", "i32"), ("transformtype", "i32"), ("mapping", "i32"),
    ]),
    "vorbis_info_psy_global": ("InfoPsyGlobal", [
        ("eighth_octave_lines", "i32"), ("preecho_thresh", ("arr", "f32", VE_BANDS)),
        ("postecho_thresh", ("arr", "f32", VE_BANDS)), ("stretch_penalty", "f32"),
        ("preecho_minenergy", "f32"), ("ampmax_att_per_sec", "f32"),
        ("coupling_pkhz", ("arr", "i32", PACKETBLOBS)),
        ("coupling_pointlimit", ("arr", ("arr", "i32", PACKETBLOBS), 2)),
        ("coupling_prepointamp", ("arr", "i32", PACKETBLOBS)),
        ("coupling_postpointamp", ("arr", "i32", PACKETBLOBS)),
        ("sliding_lowpass", ("arr", ("arr", "i32", PACKETBLOBS), 2)),
    ]),
    "vorbis_info_psy": ("InfoPsy", [
        ("blockflag", "i32"), ("ath_adjatt", "f32"), ("ath_maxatt", "f32"),
        ("tone_masteratt", ("arr", "f32", P_NOISECURVES)), ("tone_centerboost", "f32"),
        ("tone_decay", "f32"), ("tone_abs_limit", "f32"), ("toneatt", ("arr", "f32", P_BANDS)),
        ("noisemaskp", "i32"), ("noisemaxsupp", "f32"), ("noisewindowlo", "f32"),
        ("noisewindowhi", "f32"), ("noisewindowlomin", "i32"), ("noisewindowhimin", "i32"),
        ("noisewindowfixed", "i32"), ("noiseoff", ("arr", ("arr", "f32", P_BANDS), P_NOISECURVES)),
        ("noisecompand", ("arr", "f32", NOISE_COMPAND_LEVELS)), ("max_curve_db", "f32"),
        ("normal_p", "i32"), ("normal_start", "i32"), ("normal_partition", "i32"),
        ("normal_thresh", "f64"),
    ]),
    "att3": ("Att3", [("att", ("arr", "i32", P_NOISECURVES)), ("boost", "f32"), ("decay", "f32")]),
    "adj_stereo": ("AdjStereo", [
        ("pre", ("arr", "i32", PACKETBLOBS)), ("post", ("arr", "i32", PACKETBLOBS)),
        ("khz", ("arr", "f32", PACKETBLOBS)), ("lowpass_khz", ("arr", "f32", PACKETBLOBS)),
    ]),
    "noiseguard": ("NoiseGuard", [("lo", "i32"), ("hi", "i32"), ("fixed", "i32")]),
    "noise3": ("Noise3", [("data", ("arr", ("arr", "i32", 17), P_NOISECURVES))]),
    "vp_adjblock": ("VpAdjBlock", [("block", ("arr", "i32", P_BANDS))]),
    "compandblock": ("CompandBlock", [("data", ("arr", "i32", NOISE_COMPAND_LEVELS))]),
    "static_bookblock": ("StaticBookBlock", [
        ("books", ("arr", ("arr", ("opt", ("ref", ("struct", "static_codebook"))), 4), 12)),
    ]),
    "vorbis_residue_template": ("ResidueTemplate", [
        ("res_type", "i32"), ("limit_type", "i32"), ("grouping", "i32"),
        ("res", ("ref", ("struct", "vorbis_info_residue0"))),
        ("book_aux", ("ref", ("struct", "static_codebook"))),
        ("book_aux_managed", ("skip", None)),
        ("books_base", ("ref", ("struct", "static_bookblock"))),
        ("books_base_managed", ("skip", None)),
    ]),
    "vorbis_mapping_template": ("MappingTemplate", [
        ("map", ("slice", ("struct", "vorbis_info_mapping0"))),
        ("res", ("slice", ("struct", "vorbis_residue_template"))),
    ]),
    "ve_setup_data_template": ("SetupDataTemplate", [
        ("mappings", "i32"),
        ("rate_mapping", ("slice", "f64")),
        ("quality_mapping", ("slice", "f64")),
        ("coupling_restriction", "i32"),
        ("samplerate_min_restriction", "i32"),
        ("samplerate_max_restriction", "i32"),
        ("blocksize_short", ("slice", "i32")),
        ("blocksize_long", ("slice", "i32")),
        ("psy_tone_masteratt", ("slice", ("struct", "att3"))),
        ("psy_tone_0db", ("slice", "i32")),
        ("psy_tone_dbsuppress", ("slice", "i32")),
        ("psy_tone_adj_impulse", ("slice", ("struct", "vp_adjblock"))),
        ("psy_tone_adj_long", ("opt", ("slice", ("struct", "vp_adjblock")))),
        ("psy_tone_adj_other", ("slice", ("struct", "vp_adjblock"))),
        ("psy_noiseguards", ("slice", ("struct", "noiseguard"))),
        ("psy_noise_bias_impulse", ("slice", ("struct", "noise3"))),
        ("psy_noise_bias_padding", ("slice", ("struct", "noise3"))),
        ("psy_noise_bias_trans", ("opt", ("slice", ("struct", "noise3")))),
        ("psy_noise_bias_long", ("opt", ("slice", ("struct", "noise3")))),
        ("psy_noise_dbsuppress", ("slice", "i32")),
        ("psy_noise_compand", ("slice", ("struct", "compandblock"))),
        ("psy_noise_compand_short_mapping", ("slice", "f64")),
        ("psy_noise_compand_long_mapping", ("opt", ("slice", "f64"))),
        ("psy_noise_normal_start", ("arr", ("slice", "i32"), 2)),
        ("psy_noise_normal_partition", ("arr", ("slice", "i32"), 2)),
        ("psy_noise_normal_thresh", ("slice", "f64")),
        ("psy_ath_float", ("slice", "i32")),
        ("psy_ath_abs", ("slice", "i32")),
        ("psy_lowpass", ("slice", "f64")),
        ("global_params", ("slice", ("struct", "vorbis_info_psy_global"))),
        ("global_mapping", ("slice", "f64")),
        ("stereo_modes", ("opt", ("slice", ("struct", "adj_stereo")))),
        ("floor_books", ("slice", ("slice", ("opt", ("ref", ("struct", "static_codebook")))))),
        ("floor_params", ("slice", ("struct", "vorbis_info_floor1"))),
        ("floor_mappings", "i32"),
        ("floor_mapping_list", ("slice", ("slice", "i32"))),
        ("maps", ("slice", ("struct", "vorbis_mapping_template"))),
    ]),
}

SCALAR_C = {"int": "i32", "long": "i32", "char": "u8", "float": "f32", "double": "f64"}

# --------------------------------------------------------------------------
# exact numeric conversion
# --------------------------------------------------------------------------


def f32_from_fraction(fr):
    """Correctly rounded (nearest-even) conversion of an exact rational to f32."""
    if fr == 0:
        return np.float32(0.0)
    approx = np.float32(float(fr))
    cands = {approx, np.nextafter(approx, np.float32(np.inf)), np.nextafter(approx, np.float32(-np.inf))}
    best = None
    for c in cands:
        if not np.isfinite(c):
            continue
        err = abs(Fraction(float(c)) - fr)
        key = (err, int(np.array(c, dtype=np.float32).view(np.uint32)) & 1)
        if best is None or key < best[0]:
            best = (key, c)
    return np.float32(best[1])


def num_to(kind, v):
    """Convert a parsed constant to the C member type `kind` (i32/u8/f32/f64)."""
    if v is NULL:
        raise ValueError("NULL for scalar")
    if not isinstance(v, Num):
        raise ValueError("expected number, got %r" % (v,))
    if kind in ("i32", "u8"):
        if v.kind == "int":
            r = v.value
        else:
            # C double->int conversion truncates toward zero
            q = v.value
            r = int(q) if q >= 0 else -int(-q)
        return r
    if kind in ("f32", "f64") and v.value == 0 and v.text is not None and v.text.startswith("-"):
        return np.float32(-0.0) if kind == "f32" else -0.0  # C keeps the sign of -0.f
    if kind == "f64":
        if v.kind == "int":
            return float(v.value)
        if v.kind == "double":
            return float(v.text)  # Python float() is correctly rounded
        return float(f32_from_fraction(v.value))  # float literal promoted
    if kind == "f32":
        if v.kind == "int":
            return f32_from_fraction(Fraction(v.value))
        if v.kind == "double":
            # literal is a double, then the initializer converts double->float
            return np.float32(float(v.text))
        return f32_from_fraction(v.value)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# initializer evaluation with C aggregate semantics
# --------------------------------------------------------------------------


def is_aggregate(t):
    return (isinstance(t, tuple) and t[0] in ("arr", "struct")) or (isinstance(t, str) and t in STRUCTS)


def members(t):
    """List of member types of an aggregate type."""
    if t[0] == "arr":
        return [t[1]] * t[2]
    return [ft for (_n, ft) in STRUCTS[t[1]][1]]


def zero_of(t):
    if isinstance(t, str):
        return np.float32(0) if t == "f32" else (0.0 if t == "f64" else 0)
    k = t[0]
    if k == "arr":
        return [zero_of(t[1]) for _ in range(t[2])]
    if k == "struct":
        return [zero_of(ft) for ft in members(t)]
    if k in ("opt", "skip"):
        return None
    if k in ("ref", "slice"):
        return None  # must be explicitly initialized (checked at emission)
    raise ValueError(t)


def init_scalar(t, item):
    if isinstance(item, list):
        if not item:
            return zero_of(t)
        item = item[0]
    if isinstance(t, str):
        return num_to(t, item)
    k = t[0]
    if k == "skip":
        return None
    if k == "opt":
        if item is NULL or (isinstance(item, Num) and item.kind == "int" and item.value == 0):
            return None
        return init_scalar(t[1], item)
    if k in ("ref", "slice"):
        if isinstance(item, Ref):
            return item
        if item is NULL or (isinstance(item, Num) and item.value == 0):
            return None
        raise ValueError("bad pointer initializer %r" % (item,))
    raise ValueError(t)


def fill_aggregate(t, items, pos):
    vals = []
    for mt in members(t):
        if pos >= len(items):
            vals.append(zero_of(mt))
            continue
        item = items[pos]
        if is_aggregate(mt):
            if isinstance(item, list):
                vals.append(init_braced(mt, item))
                pos += 1
            else:
                v, pos = fill_aggregate(mt, items, pos)
                vals.append(v)
        else:
            vals.append(init_scalar(mt, item))
            pos += 1
    return vals, pos


def init_braced(t, item):
    if is_aggregate(t):
        if not isinstance(item, list):
            raise ValueError("aggregate needs braces: %r" % (item,))
        v, pos = fill_aggregate(t, item, 0)
        if pos != len(item):
            raise ValueError("excess initializers for %r" % (t,))
        return v
    return init_scalar(t, item)


def count_elements(elem_t, items):
    """Number of array elements an unsized array initializer produces."""
    if not is_aggregate(elem_t):
        return len(items)
    n = 0
    pos = 0
    while pos < len(items):
        if isinstance(items[pos], list):
            pos += 1
        else:
            _v, pos = fill_aggregate(elem_t, items, pos)
        n += 1
    return n


# --------------------------------------------------------------------------
# object database
# --------------------------------------------------------------------------


class Obj:
    def __init__(self, name, elem_t, is_array, value, src):
        self.name = name
        self.elem_t = elem_t  # element type
        self.is_array = is_array
        self.value = value
        self.src = src


MACROS = {"P_BANDS": 17, "EHMER_MAX": 56, "MAX_ATH": 88, "P_NOISECURVES": 3}


def dimval(x):
    if isinstance(x, Ref):
        return MACROS[x.name]
    return x.value


def decl_elem_type(d):
    """Map a C declaration to (element type, is_array)."""
    typ, ptr, dims = d["type"], d["ptr"], d["dims"]
    if typ in SCALAR_C:
        base = SCALAR_C[typ]
    elif typ in STRUCTS:
        base = ("struct", typ)
    else:
        raise ValueError("unknown type %s for %s" % (typ, d["name"]))
    # pointer element types are resolved from context; we store a marker
    if ptr == 1:
        base = ("ptr1", base)
    elif ptr == 2:
        base = ("ptr2", base)
    if len(dims) == 0:
        return base, False
    if len(dims) == 1:
        return base, True
    # multi-dimensional array: array of fixed arrays
    t = base
    for dim in reversed(dims[1:]):
        t = ("arr", t, dimval(dim))
    return t, True


def build_objects(decls, src):
    objs = {}
    for d in decls:
        elem_t, is_array = decl_elem_type(d)
        name = d["name"]
        if isinstance(elem_t, tuple) and elem_t[0] in ("ptr1", "ptr2"):
            # arrays of pointers: keep refs, resolved at emission time
            vals = [None if (x is NULL or (isinstance(x, Num) and x.value == 0)) else x for x in d["init"]]
            objs[name] = Obj(name, elem_t, True, vals, src)
            continue
        if is_array:
            items = d["init"]
            n = dimval(d["dims"][0]) if d["dims"][0] is not None else count_elements(elem_t, items)
            v = init_braced(("arr", elem_t, n), items)
        else:
            v = init_braced(elem_t, d["init"])
        objs[name] = Obj(name, elem_t, is_array, v, src)
    return objs


# --------------------------------------------------------------------------
# Rust emission
# --------------------------------------------------------------------------


def rust_name(cname):
    n = cname.lstrip("_")
    n = re.sub("_+", "_", n).upper()
    if n[0].isdigit():
        n = "CB_" + n
    return n


def fmt_f32(v):
    """Shortest decimal that parses back (correctly rounded) to the same f32."""
    v = np.float32(v)
    if v == 0:
        return "-0.0" if np.signbit(v) else "0.0"
    if 1e-4 <= abs(v) < 1e9:
        s = np.format_float_positional(v, unique=True, trim="0")
    else:
        s = np.format_float_scientific(v, unique=True, trim="0")
        mant, exp = s.split("e")
        s = mant + "e" + str(int(exp))
    assert np.float32(float(s)) == v, (s, v)
    return s


def fmt_f64(v):
    s = repr(float(v))
    if "e" in s:
        mant, exp = s.split("e")
        if "." not in mant:
            mant += ".0"
        s = mant + "e" + str(int(exp))
    elif "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    assert float(s) == v
    return s


class Emitter:
    def __init__(self, objs):
        self.objs = objs
        self.reachable = {}  # cname -> rust type string of the item
        self.order = []

    def obj(self, name):
        if name not in self.objs:
            raise KeyError("unknown object %s" % name)
        return self.objs[name]

    # type of a Rust static holding object `o` when referenced as type `want`
    def rust_type(self, t):
        if isinstance(t, str):
            return t
        k = t[0]
        if k == "arr":
            return "[%s; %d]" % (self.rust_type(t[1]), t[2])
        if k == "struct":
            return STRUCTS[t[1]][0]
        # statics give every reference a 'static lifetime implicitly
        if k == "ref":
            return "&%s" % self.rust_type(t[1])
        if k == "slice":
            return "&[%s]" % self.rust_type(t[1])
        if k == "opt":
            return "Option<%s>" % self.rust_type(t[1])
        raise ValueError(t)

    def mark(self, name, want):
        """Mark object reachable; `want` is the pointer type it is referenced as."""
        o = self.obj(name)
        if name in self.reachable:
            return
        # Determine the Rust element type for this object.
        if isinstance(o.elem_t, tuple) and o.elem_t[0] in ("ptr1", "ptr2"):
            # array of pointers: element type comes from the referencing context
            assert want[0] == "slice", (name, want)
            et = want[1]
            self.reachable[name] = ("slice", et)
            self.order.append(name)
            for x in o.value:
                if x is not None:
                    self.walk_value(et, x)
            return
        if o.is_array:
            self.reachable[name] = ("slice", o.elem_t)
        else:
            self.reachable[name] = ("ref", o.elem_t)
        self.order.append(name)
        if o.is_array:
            for v in o.value:
                self.walk_value(o.elem_t, v)
        else:
            self.walk_value(o.elem_t, o.value)

    def walk_value(self, t, v):
        if v is None:
            return
        if isinstance(t, str):
            return
        k = t[0]
        if k == "arr":
            for x in v:
                self.walk_value(t[1], x)
        elif k == "struct":
            for (fname, ft), x in zip(STRUCTS[t[1]][1], v):
                if isinstance(ft, tuple) and ft[0] == "skip":
                    continue
                self.walk_value(ft, x)
        elif k == "opt":
            self.walk_value(t[1], v)
        elif k in ("ref", "slice"):
            if not isinstance(v, Ref):
                raise ValueError("expected reference, got %r" % (v,))
            self.mark(v.name, t)

    def value_expr(self, t, v, indent=""):
        if isinstance(t, str):
            if t == "f32":
                return fmt_f32(v)
            if t == "f64":
                return fmt_f64(v)
            return str(int(v))
        k = t[0]
        if k == "arr":
            inner = [self.value_expr(t[1], x, indent + "    ") for x in v]
            if all("\n" not in s for s in inner):
                return wrap_list(inner, indent)
            return "[\n" + "".join(indent + "    " + s + ",\n" for s in inner) + indent + "]"
        if k == "struct":
            name, fields = STRUCTS[t[1]]
            parts = []
            for (fname, ft), x in zip(fields, v):
                if isinstance(ft, tuple) and ft[0] == "skip":
                    continue
                parts.append("%s    %s: %s,\n" % (indent, fname, self.value_expr(ft, x, indent + "    ")))
            return "%s {\n%s%s}" % (name, "".join(parts), indent)
        if k == "opt":
            if v is None:
                return "None"
            return "Some(%s)" % self.value_expr(t[1], v, indent)
        if k in ("ref", "slice"):
            if v is None:
                raise ValueError("NULL in non-optional pointer")
            target = self.reachable[v.name]
            rn = rust_name(v.name)
            if k == "ref":
                if target[0] == "slice":
                    # pointer to first element of an array (C decay)
                    return "&%s[0]" % rn
                return "&%s" % rn
            # slice
            if target[0] == "ref":
                raise ValueError("slice to scalar object %s" % v.name)
            return "&%s" % rn
        raise ValueError(t)


def wrap_list(items, indent):
    s = ", ".join(items)
    if len(s) < 90 and "\n" not in s:
        return "[" + s + "]"
    lines = []
    cur = ""
    for it in items:
        piece = it + ", "
        if len(cur) + len(piece) > 96:
            lines.append(cur.rstrip())
            cur = ""
        cur += piece
    if cur:
        lines.append(cur.rstrip())
    return "[\n" + "".join(indent + "    " + ln + "\n" for ln in lines) + indent + "]"


HEADER = """// @generated by tools/vorbis_encoder/gen_tables.py from libvorbis 1.3.7 ({srcs}).
// Do not edit by hand; re-run `python3 tools/vorbis_encoder/gen_tables.py` instead.
// libvorbis is Copyright (c) 2002-2020 Xiph.org Foundation, BSD-3-Clause.

"""


def emit_file(path, em, names, srcs, uses):
    with open(path, "w", newline="\n") as f:
        f.write(HEADER.format(srcs=", ".join(srcs)))
        f.write(uses)
        for cname in names:
            kind, t = em.reachable[cname]
            o = em.obj(cname)
            rn = rust_name(cname)
            if kind == "ref":
                ty = em.rust_type(t)
                val = em.value_expr(t, o.value)
            else:
                vals = o.value
                ty = "[%s; %d]" % (em.rust_type(t), len(vals))
                val = em.value_expr(("arr", t, len(vals)), vals)
            f.write("/// `%s` (%s)\n" % (cname, o.src))
            f.write("pub static %s: %s = %s;\n\n" % (rn, ty, val))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gen_tables.py path/to/libvorbis-1.3.7")
    vorbis = sys.argv[1]
    lib = os.path.join(vorbis, "lib")
    objs = {}

    def add(path, only=None):
        rel = os.path.relpath(path, vorbis).replace(os.sep, "/")
        for k, v in build_objects(parse_file(path, only), rel).items():
            if k in objs:
                raise ValueError("duplicate %s" % k)
            objs[k] = v

    book_files = [
        "books/floor/floor_books.h",
        "books/coupled/res_books_stereo.h",
        "books/uncoupled/res_books_uncoupled.h",
    ]
    for b in book_files:
        add(os.path.join(lib, b))
    for m in sorted(os.listdir(os.path.join(lib, "modes"))):
        if m.endswith(".h") and "p51" not in m:
            add(os.path.join(lib, "modes", m))
    add(os.path.join(lib, "masking.h"))
    add(os.path.join(lib, "vorbisenc.c"), ["_mode_template", "_map_nominal"])
    vwins = ["vwin%d" % (1 << i) for i in range(6, 14)]
    add(os.path.join(lib, "window.c"), vwins)
    add(os.path.join(lib, "floor1.c"), ["FLOOR1_fromdB_LOOKUP"])

    # psy.c carries an identical copy of FLOOR1_fromdB_LOOKUP; make sure.
    psy_copy = build_objects(parse_file(os.path.join(lib, "psy.c"), ["FLOOR1_fromdB_LOOKUP"]), "lib/psy.c")
    assert [float(x) for x in psy_copy["FLOOR1_fromdB_LOOKUP"].value] == \
        [float(x) for x in objs["FLOOR1_fromdB_LOOKUP"].value]

    # setup_list order from vorbisenc.c
    with open(os.path.join(lib, "vorbisenc.c")) as f:
        venc = strip_comments_and_pp(f.read())
    m = re.search(r"setup_list\[\]\s*=\s*\{(.*?)\};", venc, re.S)
    setup_list = [s.strip().lstrip("&") for s in m.group(1).split(",") if s.strip() and s.strip() != "0"]
    setup_list = [s for s in setup_list if s != "ve_setup_44_51"]

    em = Emitter(objs)
    setup_t = ("struct", "ve_setup_data_template")
    for s in setup_list:
        em.mark(s, ("ref", setup_t))
    em.mark("_psy_info_template", ("ref", ("struct", "vorbis_info_psy")))
    em.mark("_mode_template", ("slice", ("struct", "vorbis_info_mode")))
    em.mark("_map_nominal", ("slice", ("struct", "vorbis_info_mapping0")))
    em.mark("ATH", ("slice", "f32"))
    em.mark("tonemasks", ("slice", ("arr", ("arr", "f32", 56), 6)))
    for w in vwins:
        em.mark(w, ("slice", "f32"))
    em.mark("FLOOR1_fromdB_LOOKUP", ("slice", "f32"))

    names = [rust_name(n) for n in em.order]
    assert len(set(names)) == len(names), "rust name collision"

    groups = {
        "gen_books_floor.rs": [],
        "gen_books_stereo.rs": [],
        "gen_books_uncoupled.rs": [],
        "gen_modes.rs": [],
        "gen_misc.rs": [],
    }
    srcs = {k: set() for k in groups}
    for cname in em.order:
        src = objs[cname].src
        if src.startswith("lib/books/floor"):
            g = "gen_books_floor.rs"
        elif src.startswith("lib/books/coupled"):
            g = "gen_books_stereo.rs"
        elif src.startswith("lib/books/uncoupled"):
            g = "gen_books_uncoupled.rs"
        elif src.startswith("lib/modes") or src == "lib/vorbisenc.c":
            g = "gen_modes.rs"
        else:
            g = "gen_misc.rs"
        groups[g].append(cname)
        srcs[g].add(src)

    uses = {
        "gen_books_floor.rs": "use super::types::StaticCodebook;\n\n",
        "gen_books_stereo.rs": "use super::types::StaticCodebook;\n\n",
        "gen_books_uncoupled.rs": "use super::types::StaticCodebook;\n\n",
        "gen_modes.rs": "use super::gen_books_floor::*;\nuse super::gen_books_stereo::*;\n"
        "use super::gen_books_uncoupled::*;\nuse super::types::*;\n\n",
        "gen_misc.rs": "",
    }
    os.makedirs(OUTDIR, exist_ok=True)
    for g, names_g in groups.items():
        # keep a stable, source-order listing
        names_g.sort(key=lambda n: list(objs).index(n))
        emit_file(os.path.join(OUTDIR, g), em, names_g, sorted(srcs[g]), uses[g])

    # setup list
    with open(os.path.join(OUTDIR, "gen_modes.rs"), "a", newline="\n") as f:
        f.write("/// `setup_list` of lib/vorbisenc.c without `ve_setup_44_51` (more than 2 channels).\n")
        f.write("pub static SETUP_LIST: [&SetupDataTemplate; %d] = [\n" % len(setup_list))
        for s in setup_list:
            f.write("    &%s,\n" % rust_name(s))
        f.write("];\n")
    print("generated %d objects into %s" % (len(em.order), OUTDIR))


if __name__ == "__main__":
    main()
