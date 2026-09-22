#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ShaderFuscation :: ShaderBuilder

Generates a GPU compute shader (GLSL 4.30 or HLSL CS 5.0) that carries an
obfuscated copy of an arbitrary payload (`payload.bin`, a Windows PE, or any
raw blob) and reconstructs the original bytes entirely on the GPU.

Every 32-bit word is obfuscated with a chain of *position dependent* and
*individually invertible* operations, so a shader invocation only has to look
at its own index and one slot of the embedded buffer - there is no cross word
dependency and therefore no serialization:

    forward  (python, build time)     inverse  (shader, run time)
    ----------------------------      --------------------------
    y1 = p  ^ k1(i)                   y1 = rotl8(y2, (4 - (k1 & 3)) & 3)
    y2 = rotl8(y1, k1(i) & 3)         y2 = y3 - k2(i)          (mod 2^32)
    y3 = y2 + k2(i)      (mod 2^32)   y3 = byteswap32(buf[perm(i)])
    buf[perm(i)] = byteswap32(y3)     p  = y1 ^ k1(i)

`k1`/`k2` are derived from the word index and a 32-bit build key:

    k1(i) = mix32(i * 0x9E3779B1 + key)
    k2(i) = mix32(k1(i) ^ 0xC2B2AE35)

`perm` is an affine bijection confined to 2^block_bits-word blocks. Because
the multiplier is odd (hence coprime with the power-of-two block size) the map
is a permutation, and because 32-bit wraparound preserves the low bits the
whole thing is computable with wrapping `uint` arithmetic on the GPU - no
permutation table has to be embedded:

    perm(i) = (i & ~mask) | (((i & mask) * a + b) & mask)

The generated shader is self-contained: the obfuscated words are emitted as a
`const uint[]` literal, so nothing but dispatch dimensions has to be provided
by the host. `--verify` (on by default) replays the shader's arithmetic in
python over the embedded buffer and asserts that the payload round-trips.

Examples
--------
    python main.py payload.bin -o build --target both
    python main.py beacon.exe -k 0xC0FFEE11 --meta
    python main.py blob.bin --block-bits 10 -g 128
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

MASK32 = 0xFFFFFFFF

# --- parameters shared verbatim with the generated shaders -------------------
KS_STRIDE = 0x9E3779B1  # k1 = mix32(i * KS_STRIDE + key)
KS_LAYER = 0xC2B2AE35  # k2 = mix32(k1 ^ KS_LAYER)
SM_INC = 0x9E3779B9  # splitmix32 step (build side only)


# ---------------------------------------------------------------------------
# 32-bit integer primitives (must match the shader code exactly)
# ---------------------------------------------------------------------------
def mix32(x: int) -> int:
    """Three round 32-bit finalizer, identical to `mix32` in the shaders."""
    x &= MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK32
    x ^= x >> 16
    return x


def splitmix32(state: int) -> Tuple[int, int]:
    """Returns (new_state, value). Build-side key schedule only."""
    state = (state + SM_INC) & MASK32
    z = state
    z ^= z >> 16
    z = (z * 0x21F0AAAD) & MASK32
    z ^= z >> 15
    z = (z * 0x735A2D97) & MASK32
    return state, (z ^ (z >> 15)) & MASK32


def byteswap32(x: int) -> int:
    x &= MASK32
    return (
        (x >> 24)
        | ((x >> 8) & 0x0000FF00)
        | ((x << 8) & 0x00FF0000)
        | ((x << 24) & 0xFF000000)
    ) & MASK32


def rotl8(x: int, r: int) -> int:
    """Rotate a 32-bit word left by r bytes (r is taken modulo 4)."""
    r &= 3
    return ((x << (8 * r)) | (x >> ((32 - 8 * r) & 31))) & MASK32


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & MASK32
    return h


# ---------------------------------------------------------------------------
# Obfuscation plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    """Everything the shader needs to know about the transformation."""

    key: int
    plain_len: int  # payload size in bytes
    word_count: int  # words that carry payload data
    total_words: int  # words embedded in the shader (block aligned)
    block_bits: int
    perm_a: int
    perm_b: int

    @property
    def block_mask(self) -> int:
        return (1 << self.block_bits) - 1

    @property
    def padding_words(self) -> int:
        return self.total_words - self.word_count


def make_plan(
    plain_len: int, key: int, block_bits: Optional[int] = None
) -> Plan:
    """Choose the permutation block size and derive its affine coefficients."""
    if plain_len <= 0:
        raise ValueError("payload is empty")

    word_count = (plain_len + 3) // 4

    if block_bits is None:
        # Aim for a block of roughly word_count / 8 words, clamped to
        # [16 words, 64 Ki words] (64 bytes .. 256 KiB of padding overhead).
        block_bits = min(16, max(4, (word_count - 1).bit_length() - 3))
    if not 1 <= block_bits <= 31:
        raise ValueError("--block-bits must be in [1, 31]")

    block = 1 << block_bits
    total_words = -(-word_count // block) * block

    state, r_a = splitmix32(key)
    _, r_b = splitmix32(state)

    # An odd multiplier is coprime with 2^block_bits, so x -> a*x + b is a
    # bijection on the block and `perm` is a permutation of the whole buffer.
    perm_a = (r_a | 1) & (block - 1)
    if perm_a <= 1:
        perm_a = 3
    perm_b = r_b & (block - 1)

    return Plan(
        key=key & MASK32,
        plain_len=plain_len,
        word_count=word_count,
        total_words=total_words,
        block_bits=block_bits,
        perm_a=perm_a,
        perm_b=perm_b,
    )


def keypair(i: int, key: int) -> Tuple[int, int]:
    k1 = mix32((i * KS_STRIDE + key) & MASK32)
    k2 = mix32(k1 ^ KS_LAYER)
    return k1, k2


def permute_index(i: int, plan: Plan) -> int:
    m = plan.block_mask
    return (i & ~m & MASK32) | (((i & m) * plan.perm_a + plan.perm_b) & m)


# ---------------------------------------------------------------------------
# Payload <-> buffer
# ---------------------------------------------------------------------------
def to_words(data: bytes) -> List[int]:
    return [
        int.from_bytes(data[i : i + 4].ljust(4, b"\x00"), "little")
        for i in range(0, len(data), 4)
    ]


def obfuscate_word(plain_word: int, i: int, plan: Plan) -> int:
    """Forward transform; its exact inverse lives in the generated shader."""
    k1, k2 = keypair(i, plan.key)
    v = plain_word ^ k1
    v = rotl8(v, k1 & 3)
    v = (v + k2) & MASK32
    return byteswap32(v)


def shader_word(buf: Sequence[int], i: int, plan: Plan) -> int:
    """Python mirror of one shader invocation (used for verification)."""
    k1, k2 = keypair(i, plan.key)
    v = buf[permute_index(i, plan)]
    v = byteswap32(v)
    v = (v - k2) & MASK32
    v = rotl8(v, (4 - (k1 & 3)) & 3)
    return (v ^ k1) & MASK32


def build_buffer(data: bytes, plan: Plan) -> List[int]:
    """Obfuscate the payload into the block-aligned buffer to embed."""
    words = to_words(data)
    buf = [0] * plan.total_words
    for i in range(plan.total_words):
        plain_word = words[i] if i < plan.word_count else 0
        buf[permute_index(i, plan)] = obfuscate_word(plain_word, i, plan)
    return buf


def restore_payload(buf: Sequence[int], plan: Plan) -> bytes:
    """Replay the shader over `buf` exactly as the GPU would."""
    out = bytearray()
    for i in range(plan.word_count):
        out += shader_word(buf, i, plan).to_bytes(4, "little")
    return bytes(out[: plan.plain_len])


# ---------------------------------------------------------------------------
# PE introspection (best effort, never fatal)
# ---------------------------------------------------------------------------
_MACHINES = {
    0x014C: "i386",
    0x01C0: "ARM",
    0x01C4: "ARMv7",
    0x0200: "IA64",
    0x8664: "x86-64",
    0xAA64: "ARM64",
}
_SUBSYSTEMS = {
    1: "native",
    2: "windows-gui",
    3: "windows-cui",
    7: "posix-cui",
    9: "windows-ce",
    10: "efi-app",
    11: "efi-boot",
    14: "xbox",
    16: "windows-boot",
}


def pe_summary(data: bytes) -> List[str]:
    """Return human readable PE facts, or [] if `data` is not a PE image."""
    try:
        if len(data) < 0x40 or data[:2] != b"MZ":
            return []
        (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
        if e_lfanew + 0x18 > len(data) or data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
            return []

        machine, nsections, timestamp = struct.unpack_from("<HHI", data, e_lfanew + 4)
        opt_off = e_lfanew + 0x18
        (magic,) = struct.unpack_from("<H", data, opt_off)
        is_plus = magic == 0x20B
        if magic not in (0x10B, 0x20B):
            return []

        entry, = struct.unpack_from("<I", data, opt_off + 0x10)
        image_base_off = opt_off + (0x18 if is_plus else 0x1C)
        image_base = struct.unpack_from("<Q" if is_plus else "<I", data, image_base_off)[0]
        (subsystem,) = struct.unpack_from("<H", data, opt_off + 0x44)
        size_of_image, = struct.unpack_from("<I", data, opt_off + 0x38)
        size_of_headers, = struct.unpack_from("<I", data, opt_off + 0x3C)
        (opt_size,) = struct.unpack_from("<H", data, e_lfanew + 0x14)

        lines = [
            f"pe format      : {'PE32+' if is_plus else 'PE32'} "
            f"({_MACHINES.get(machine, hex(machine))}) subsystem="
            f"{_SUBSYSTEMS.get(subsystem, subsystem)}",
            f"pe entry       : rva 0x{entry:X} image base 0x{image_base:X} "
            f"image size 0x{size_of_image:X}",
            f"pe headers     : {size_of_headers} bytes, {nsections} sections, "
            f"timestamp {timestamp}",
        ]

        sect_off = opt_off + opt_size
        for n in range(nsections):
            off = sect_off + n * 40
            if off + 40 > len(data):
                break
            name = data[off : off + 8].split(b"\x00")[0].decode("latin-1", "replace")
            vsize, va, raw_size, raw_ptr = struct.unpack_from("<IIII", data, off + 8)
            (chars,) = struct.unpack_from("<I", data, off + 36)
            lines.append(
                f"pe section[{n}]  : {name:<8} va 0x{va:08X} vsize 0x{vsize:08X} "
                f"raw 0x{raw_ptr:08X}+0x{raw_size:X} chars 0x{chars:08X}"
            )
        return lines
    except (struct.error, IndexError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes} B"


def parse_key(raw: str) -> int:
    """Accept `0x...`, decimal, or an arbitrary string (hashed with fnv1a32)."""
    text = raw.strip()
    try:
        value = int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    except ValueError:
        value = fnv1a32(text.encode("utf-8"))
    if not 0 <= value <= MASK32:
        raise argparse.ArgumentTypeError("key must fit in 32 bits")
    return value


def format_literals(words: Sequence[int], per_line: int, indent: str, suffix: str) -> str:
    rows = []
    for off in range(0, len(words), per_line):
        chunk = words[off : off + per_line]
        rows.append(indent + ", ".join(f"0x{w:08X}{suffix}" for w in chunk) + ",")
    return "\n".join(rows)


def comment_lines(lines: Sequence[str], marker: str) -> str:
    return "\n".join(f"{marker} {line}" if line else marker for line in lines)


# ---------------------------------------------------------------------------
# Shader emission
# ---------------------------------------------------------------------------
GLSL_FUNCS = """\
uint mix32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint permute(uint i) {
    return (i & ~g_blockMask) | (((i & g_blockMask) * g_permA + g_permB) & g_blockMask);
}

uint byteswap32(uint x) {
    return (x >> 24) | ((x >> 8) & 0x0000ff00u)
         | ((x << 8) & 0x00ff0000u) | (x << 24);
}

uint rotl8(uint x, uint r) {
    r &= 3u;
    return (x << (r * 8u)) | (x >> ((32u - r * 8u) & 31u));
}

void keypair(uint i, out uint k1, out uint k2) {
    k1 = mix32(i * 0x9e3779b1u + g_key);
    k2 = mix32(k1 ^ 0xc2b2ae35u);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= g_wordCount)
        return;

    uint k1, k2;
    keypair(i, k1, k2);

    uint v = g_obf[permute(i)];
    v = byteswap32(v);
    v -= k2;
    v = rotl8(v, (4u - (k1 & 3u)) & 3u);
    v ^= k1;

    g_out[i] = v;
}
"""

HLSL_FUNCS = """\
uint mix32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint permute(uint i) {
    return (i & ~g_blockMask) | (((i & g_blockMask) * g_permA + g_permB) & g_blockMask);
}

uint byteswap32(uint x) {
    return (x >> 24) | ((x >> 8) & 0x0000ff00u)
         | ((x << 8) & 0x00ff0000u) | (x << 24);
}

uint rotl8(uint x, uint r) {
    r &= 3u;
    return (x << (r * 8u)) | (x >> ((32u - r * 8u) & 31u));
}

void keypair(uint i, out uint k1, out uint k2) {
    k1 = mix32(i * 0x9e3779b1u + g_key);
    k2 = mix32(k1 ^ 0xc2b2ae35u);
}

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= g_wordCount)
        return;

    uint k1, k2;
    keypair(i, k1, k2);

    uint v = g_obf[permute(i)];
    v = byteswap32(v);
    v -= k2;
    v = rotl8(v, (4u - (k1 & 3u)) & 3u);
    v ^= k1;

    g_out[i] = v;
}
"""


def metadata_lines(
    source: str,
    plan: Plan,
    group_size: int,
    groups: int,
    digest: int,
    pe_lines: Sequence[str],
) -> List[str]:
    mask = plan.block_mask
    lines = [
        "ShaderFuscation :: GPU payload restoration",
        "",
        f"source         : {source}",
        f"payload        : {plan.plain_len} bytes ({human(plan.plain_len)})",
        f"words          : {plan.word_count} data + {plan.padding_words} padding "
        f"= {plan.total_words} embedded",
        f"dispatch       : ({groups}, 1, 1) workgroups of {group_size} "
        f"({max(groups * group_size, plan.word_count)} invocations)",
        f"key            : 0x{plan.key:08X}",
        f"permutation    : perm(i) = (i & ~0x{mask:08X}) | "
        f"(((i & 0x{mask:08X}) * 0x{plan.perm_a:08X} + 0x{plan.perm_b:08X}) & 0x{mask:08X})",
        "block bits     : "
        f"{plan.block_bits} ({1 << plan.block_bits} words / {human(4 << plan.block_bits)} blocks)",
        "keystream      : k1 = mix32(i * 0x9E3779B1 + key), k2 = mix32(k1 ^ 0xC2B2AE35)",
        "layers         : xor(k1) -> rotl8(k1 & 3) -> add(k2) -> byteswap -> permute",
        f"fnv1a32        : 0x{digest:08X} (of the original payload)",
    ]
    lines += list(pe_lines)
    lines.append(
        f"generated      : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    lines += [
        "",
        "the embedded words are uint32 little-endian; read the output buffer back,",
        f"truncate to {plan.plain_len} bytes and the original payload is restored.",
    ]
    return lines


def emit_glsl(
    meta: Sequence[str], plan: Plan, buf: Sequence[int], group_size: int
) -> str:
    head = [
        "#version 430",
        "",
        comment_lines(meta, "//"),
        "",
        f"#define GROUP_SIZE {group_size}",
        "",
        "layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;",
        "",
        "layout(std430, binding = 0) buffer Restored {",
        "    uint g_out[];",
        "};",
        "",
        f"const uint g_key       = 0x{plan.key:08X}u;",
        f"const uint g_permA     = 0x{plan.perm_a:08X}u;",
        f"const uint g_permB     = 0x{plan.perm_b:08X}u;",
        f"const uint g_blockMask = 0x{plan.block_mask:08X}u;",
        f"const uint g_wordCount = {plan.word_count}u;",
        "",
        f"const uint g_obf[{plan.total_words}] = uint[{plan.total_words}](",
        format_literals(buf, 12, "    ", "u"),
        ");",
        "",
    ]
    return "\n".join(head) + "\n" + GLSL_FUNCS


def emit_hlsl(
    meta: Sequence[str], plan: Plan, buf: Sequence[int], group_size: int
) -> str:
    head = [
        comment_lines(meta, "//"),
        "",
        f"#define GROUP_SIZE {group_size}",
        "",
        "RWStructuredBuffer<uint> g_out : register(u0);",
        "",
        f"static const uint g_key       = 0x{plan.key:08X}u;",
        f"static const uint g_permA     = 0x{plan.perm_a:08X}u;",
        f"static const uint g_permB     = 0x{plan.perm_b:08X}u;",
        f"static const uint g_blockMask = 0x{plan.block_mask:08X}u;",
        f"static const uint g_wordCount = {plan.word_count}u;",
        "",
        f"static const uint g_obf[{plan.total_words}] = {{",
        format_literals(buf, 12, "    ", "u"),
        "};",
        "",
    ]
    return "\n".join(head) + "\n" + HLSL_FUNCS


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Embed a payload into a compute shader that restores it on the GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[-1],
    )
    p.add_argument("payload", help="payload.bin / PE image to embed")
    p.add_argument("-o", "--outdir", default=".", help="output directory (default: .)")
    p.add_argument("-n", "--name", help="output basename (default: payload stem)")
    p.add_argument(
        "-t",
        "--target",
        choices=("glsl", "hlsl", "both"),
        default="glsl",
        help="shader dialect to emit (default: glsl)",
    )
    p.add_argument(
        "-k",
        "--key",
        default=None,
        help="32-bit key as 0x hex, decimal, or arbitrary string (default: random)",
    )
    p.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=64,
        help="workgroup size baked into the shader (default: 64)",
    )
    p.add_argument(
        "--block-bits",
        type=int,
        default=None,
        help="log2 of the permutation block size in words (default: auto)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the python round-trip check of the generated buffer",
    )
    p.add_argument("--meta", action="store_true", help="also write <name>.meta.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    src = Path(args.payload)
    if not src.is_file():
        print(f"error: no such payload: {src}", file=sys.stderr)
        return 2
    data = src.read_bytes()
    if not data:
        print(f"error: payload is empty: {src}", file=sys.stderr)
        return 2

    if args.group_size < 1 or args.group_size > 1024:
        print("error: --group-size must be in [1, 1024]", file=sys.stderr)
        return 2
    if args.group_size & (args.group_size - 1):
        print(f"warning: --group-size {args.group_size} is not a power of two")

    key = parse_key(args.key) if args.key is not None else int.from_bytes(os.urandom(4), "little")
    try:
        plan = make_plan(len(data), key, args.block_bits)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    buf = build_buffer(data, plan)
    digest = fnv1a32(data)

    if not args.no_verify:
        restored = restore_payload(buf, plan)
        if restored != data:
            bad = next(
                i for i, (a, b) in enumerate(zip(restored, data)) if a != b
            ) if len(restored) == len(data) else 0
            print(
                f"error: self-check failed (length {len(restored)} vs {len(data)}, "
                f"first difference at offset {bad})",
                file=sys.stderr,
            )
            return 1

    groups = max(1, -(-plan.word_count // args.group_size))
    meta = metadata_lines(
        str(src), plan, args.group_size, groups, digest, pe_summary(data)
    )

    name = args.name or src.stem
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    if args.target in ("glsl", "both"):
        path = outdir / f"{name}.comp"
        path.write_text(emit_glsl(meta, plan, buf, args.group_size), encoding="utf-8")
        written.append(path)
    if args.target in ("hlsl", "both"):
        path = outdir / f"{name}.hlsl"
        path.write_text(emit_hlsl(meta, plan, buf, args.group_size), encoding="utf-8")
        written.append(path)

    print(f"payload    : {src} ({human(len(data))})")
    print(f"key        : 0x{key:08X}")
    print(
        f"buffer     : {plan.word_count} words + {plan.padding_words} padding "
        f"= {plan.total_words} words ({human(plan.total_words * 4)})"
    )
    print(f"blocks     : 2^{plan.block_bits} words, a=0x{plan.perm_a:X} b=0x{plan.perm_b:X}")
    print(f"dispatch   : ({groups}, 1, 1) x {args.group_size} threads")
    print(f"reference  : fnv1a32 = 0x{digest:08X}")
    print(f"self-check : {'skipped' if args.no_verify else 'payload round-trips'}")
    if len(buf) > 4_000_000:
        print(
            "warning: very large literal array, some compilers will be slow or refuse it"
        )
    for path in written:
        print(f"wrote      : {path} ({human(path.stat().st_size)})")

    if args.meta:
        meta_path = outdir / f"{name}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "source": str(src),
                    "name": name,
                    "plain_len": plan.plain_len,
                    "word_count": plan.word_count,
                    "total_words": plan.total_words,
                    "padding_words": plan.padding_words,
                    "key": f"0x{plan.key:08X}",
                    "block_bits": plan.block_bits,
                    "perm_a": f"0x{plan.perm_a:08X}",
                    "perm_b": f"0x{plan.perm_b:08X}",
                    "group_size": args.group_size,
                    "dispatch_groups": groups,
                    "fnv1a32": f"0x{digest:08X}",
                    "pe": bool(pe_summary(data)),
                    "shaders": [p.name for p in written],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote      : {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
