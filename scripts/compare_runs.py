#!/usr/bin/env python3
"""Compare wav and debug snapshots across multiple `audio/<dir>/` archives.

Usage:
    scripts/compare_runs.py audio/iPhone_3_manual_F32_gpu_manual1_*  \
                            audio/iPhone_3_manual_*_freshFirst*  \
                            audio/iPhone_3_20260518_*

Each input dir is expected to contain `output_*.wav` and optionally
`debug/<runId>/` subdirs from CMLA_DEBUG_SNAPSHOT=1 runs.

Output: a markdown table comparing F32×cpuAndGPU wav (md5 / rms / peak)
and key debug stage hashes (mel_normalized, encoder_output, postnet_output,
waveform_postdeemph).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
import re
import struct
import sys
from pathlib import Path


def md5_short(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def wav_stats(path: Path):
    """Return (md5_short, rms_int16_or_none, peak_int16_or_none, samples)."""
    with open(path, "rb") as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest()[:10]
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return md5, None, None, 0

    i = 12
    fmt_bits = 16
    while i < len(data) - 8:
        cid = data[i : i + 4]
        clen = struct.unpack("<I", data[i + 4 : i + 8])[0]
        if cid == b"fmt ":
            fmt_bits = struct.unpack("<H", data[i + 22 : i + 24])[0]
        elif cid == b"data":
            samples_bytes = data[i + 8 : i + 8 + clen]
            if fmt_bits == 16:
                n = clen // 2
                samples = struct.unpack("<" + "h" * n, samples_bytes)
            elif fmt_bits == 32:
                n = clen // 4
                samples = struct.unpack("<" + "f" * n, samples_bytes)
            else:
                return md5, None, None, 0
            if not samples:
                return md5, 0, 0, 0
            peak = max(abs(s) for s in samples)
            sum_sq = sum(s * s for s in samples)
            rms = math.sqrt(sum_sq / len(samples))
            return md5, int(rms), int(peak), len(samples)
        i += 8 + clen
        if clen % 2 == 1:
            i += 1
    return md5, None, None, 0


def parse_summary(path: Path) -> dict:
    """Pull sha256 / shape / stats lines out of debug/<run>/summary.txt."""
    info: dict[str, str] = {}
    if not path.exists():
        return info
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        for stage in ("mel_normalized", "encoder_output", "postnet_output",
                      "waveform_predeemph", "waveform_postdeemph"):
            if line.startswith(stage + ":"):
                m = re.search(r"sha256=([0-9a-f]+)", line)
                shape_m = re.search(r"shape=(\[[^\]]+\])", line)
                if m:
                    info[stage + "_sha"] = m.group(1)[:10]
                if shape_m:
                    info[stage + "_shape"] = shape_m.group(1)
        if line.startswith("label="):
            info["label"] = line
        if line.startswith("precision="):
            info["precision_line"] = line
        if line.startswith("runId="):
            info["runId"] = line.split("=", 1)[1]
    return info


def list_debug_runs(audio_dir: Path) -> list[Path]:
    """Find any debug/<runId>/ subdirs under the audio archive."""
    debug = audio_dir / "debug"
    if not debug.is_dir():
        return []
    return sorted([p for p in debug.iterdir() if p.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="audio/<dir> snapshots")
    parser.add_argument(
        "--precision-cu",
        default="Float32_cpuAndGPU",
        help="output_<this>.wav が比較対象 (default: Float32_cpuAndGPU)",
    )
    args = parser.parse_args()

    rows = []
    expanded: list[Path] = []
    for pattern in args.dirs:
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            print(f"# warn: pattern matched nothing: {pattern}", file=sys.stderr)
        expanded.extend(matches)

    for d in expanded:
        if not d.is_dir():
            continue
        wav_path = d / f"output_{args.precision_cu}.wav"
        if not wav_path.exists():
            continue
        md5, rms, peak, n = wav_stats(wav_path)
        debug_runs = list_debug_runs(d)
        if not debug_runs:
            rows.append({
                "src": d.name,
                "wav_md5": md5,
                "wav_rms": rms,
                "wav_peak": peak,
                "label": "(no debug snapshot)",
                "mel": "-",
                "encoder": "-",
                "postnet": "-",
                "wave": "-",
                "runId": "-",
            })
            continue
        for run_dir in debug_runs:
            # Only look at runs matching the target precision_cu in their name
            if args.precision_cu.replace("_", "") not in run_dir.name.replace("_", ""):
                continue
            summary = parse_summary(run_dir / "summary.txt")
            rows.append({
                "src": d.name,
                "wav_md5": md5,
                "wav_rms": rms,
                "wav_peak": peak,
                "label": summary.get("label", "?"),
                "mel": summary.get("mel_normalized_sha", "-"),
                "encoder": summary.get("encoder_output_sha", "-"),
                "postnet": summary.get("postnet_output_sha", "-"),
                "wave": summary.get("waveform_postdeemph_sha", "-"),
                "runId": summary.get("runId", run_dir.name),
            })

    if not rows:
        print(f"no output_{args.precision_cu}.wav found in given dirs")
        return 1

    # Print markdown table
    headers = ["src", "label/runId", "wav_md5", "rms", "peak",
               "mel_sha", "enc_sha", "post_sha", "wave_sha"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        runlabel = r["runId"] if r["runId"] != "-" else r["label"]
        print(
            "| "
            + " | ".join(
                [
                    r["src"],
                    str(runlabel)[:48],
                    r["wav_md5"],
                    str(r["wav_rms"]),
                    str(r["wav_peak"]),
                    r["mel"],
                    r["encoder"],
                    r["postnet"],
                    r["wave"],
                ]
            )
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
