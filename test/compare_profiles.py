#!/usr/bin/env python3
"""compare_profiles.py  —  detect largest time changes between two .prof files.

Usage:
    python compare_profiles.py baseline.prof candidate.prof [options]

Options:
    --top N          Show top N functions per section (default: 20)
    --min-delta SEC  Only show functions with |delta| >= this (default: 0.0001)
    --sort {cum,self,calls}
                     Sort key for the diff table (default: cum)
    --no-color       Disable ANSI color output
"""

import argparse
import io
import pstats
import sys
from typing import Dict

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Stats = Dict[tuple, tuple]  # key -> (cc, nc, tt, ct, callers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _color(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{RESET}" if use_color else text


def _fn(key: tuple) -> str:
    file = key[0].split("/")[-1]
    return f"{key[2]} ({file}:{key[1]})"


def extract(path: str) -> Stats:
    s = pstats.Stats(path, stream=io.StringIO())
    return s.stats


def total_time(stats: Stats) -> float:
    return sum(v[3] for v in stats.values())


def total_calls(stats: Stats) -> int:
    return sum(v[0] for v in stats.values())


def _sign_color(delta: float, use_color: bool) -> str:
    if delta > 0.001:
        return RED
    if delta < -0.001:
        return GREEN
    return ""


def fmt_delta(delta: float, use_color: bool) -> str:
    sign = "+" if delta >= 0 else ""
    text = f"{sign}{delta:+.4f}"
    col = _sign_color(delta, use_color)
    return _color(text, col, use_color) if col else text


def fmt_ratio(ratio: float, use_color: bool) -> str:
    text = f"{ratio:6.2f}x"
    if ratio > 2:
        return _color(text, RED, use_color)
    if ratio < 0.5:
        return _color(text, GREEN, use_color)
    return text


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def build_diff(baseline: Stats, candidate: Stats) -> list:
    """Return list of (key, b_ct, c_ct, d_ct, b_tt, c_tt, d_tt, b_nc, c_nc)."""
    rows = []
    for k in set(baseline) | set(candidate):
        b = baseline.get(k)
        c = candidate.get(k)
        b_ct = b[3] if b else 0.0
        c_ct = c[3] if c else 0.0
        b_tt = b[2] if b else 0.0
        c_tt = c[2] if c else 0.0
        b_nc = b[0] if b else 0
        c_nc = c[0] if c else 0
        rows.append((k, b_ct, c_ct, c_ct - b_ct, b_tt, c_tt, c_tt - b_tt, b_nc, c_nc))
    return rows


def section(title: str, use_color: bool) -> str:
    line = "─" * 100
    return f"\n{_color(BOLD + title + RESET, BOLD, use_color)}\n{line}"


def print_header(cols: list, widths: list, use_color: bool):
    row = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(_color(row, DIM, use_color))
    print("─" * sum(widths + [2 * (len(widths) - 1)]))


def print_diff_table(
    rows: list,
    sort_idx: int,
    top: int,
    min_delta: float,
    use_color: bool,
    reverse: bool = True,
):
    """rows: list of (key, b_ct, c_ct, d_ct, b_tt, c_tt, d_tt, b_nc, c_nc)
    sort_idx: index into the tuple to sort by.
    """
    filtered = [r for r in rows if abs(r[3]) >= min_delta or abs(r[6]) >= min_delta]
    filtered.sort(key=lambda r: r[sort_idx], reverse=reverse)
    shown = filtered[:top]

    cols = [
        "Function",
        "base_cum",
        "new_cum",
        "Δcum",
        "base_tt",
        "new_tt",
        "Δtt",
        "base_nc",
        "new_nc",
    ]
    widths = [55, 9, 9, 10, 9, 9, 10, 8, 8]
    print_header(cols, widths, use_color)

    for k, bc, cc, dc, bt, ct, dt, bn, cn in shown:
        fn_str = _fn(k)[:54]
        flag = ""
        if k not in globals().get("_baseline_keys", set()):
            flag = _color(" ✦NEW", YELLOW, use_color)

        print(
            f"  {fn_str:<54}{flag}"
            f"  {bc:>9.4f}"
            f"  {cc:>9.4f}"
            f"  {fmt_delta(dc, use_color):>10}"
            f"  {bt:>9.4f}"
            f"  {ct:>9.4f}"
            f"  {fmt_delta(dt, use_color):>10}"
            f"  {bn:>8}"
            f"  {cn:>8}"
        )

    if len(filtered) > top:
        print(
            _color(
                f"  … {len(filtered) - top} more rows hidden (increase --top)",
                DIM,
                use_color,
            )
        )


def print_new_only(baseline: Stats, candidate: Stats, top: int, use_color: bool):
    new_only = [(k, v) for k, v in candidate.items() if k not in baseline]
    new_only.sort(key=lambda x: -x[1][3])
    cols = ["Function", "cumtime", "tottime", "ncalls"]
    widths = [60, 9, 9, 8]
    print_header(cols, widths, use_color)
    for k, v in new_only[:top]:
        print(f"  {_fn(k)[:59]:<59}" f"  {v[3]:>9.4f}" f"  {v[2]:>9.4f}" f"  {v[0]:>8}")
    if len(new_only) > top:
        print(_color(f"  … {len(new_only) - top} more", DIM, use_color))


def print_callers(stats: Stats, fn_name: str, use_color: bool):
    """Print callers of a function by name."""
    for k, v in stats.items():
        if k[2] == fn_name:
            callers = v[4]
            if not callers:
                print("  (no callers recorded)")
                return
            for ck, cd in sorted(callers.items(), key=lambda x: -x[1][3]):
                print(f"  {_fn(ck):<60}  ncalls={cd[0]:>6}  cumtime={cd[3]:.4f}")
            return
    print(f"  '{fn_name}' not found in profile.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Compare two .prof files and report the largest time changes."
    )
    ap.add_argument("baseline", help="Reference .prof file")
    ap.add_argument("candidate", help="New branch .prof file")
    ap.add_argument("--top", type=int, default=20, metavar="N")
    ap.add_argument("--min-delta", type=float, default=0.0001, metavar="SEC")
    ap.add_argument("--sort", choices=["cum", "self", "calls"], default="cum")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument(
        "--callers",
        nargs="*",
        metavar="FUNC",
        help="Also print callers for these function names in candidate",
    )
    args = ap.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    # -- load -----------------------------------------------------------
    try:
        baseline = extract(args.baseline)
        candidate = extract(args.candidate)
    except Exception as e:
        sys.exit(f"Error loading profiles: {e}")

    b_total = total_time(baseline)
    c_total = total_time(candidate)
    b_calls = total_calls(baseline)
    c_calls = total_calls(candidate)

    sort_idx = {"cum": 3, "self": 6, "calls": 8}[args.sort]

    diff = build_diff(baseline, candidate)

    # -- summary --------------------------------------------------------
    print(section("SUMMARY", use_color))
    ratio = c_total / b_total if b_total else float("inf")
    r_str = fmt_ratio(ratio, use_color)
    print(f"  {'baseline':<12} {args.baseline}")
    print(f"  {'candidate':<12} {args.candidate}")
    print()
    print(
        f"  {'total time':<14}  baseline={b_total:.4f}s   candidate={c_total:.4f}s   ratio={r_str}"
    )
    print(
        f"  {'total calls':<14}  baseline={b_calls:,}   candidate={c_calls:,}   delta={c_calls - b_calls:+,}"
    )

    # -- regressions (increased cumtime) --------------------------------
    print(
        section(
            f"TOP REGRESSIONS  (largest cumtime increase, showing {args.top})",
            use_color,
        )
    )
    print_diff_table(
        diff,
        sort_idx=sort_idx,
        top=args.top,
        min_delta=args.min_delta,
        use_color=use_color,
        reverse=True,
    )

    # -- improvements (decreased cumtime) -------------------------------
    print(
        section(
            f"TOP IMPROVEMENTS  (largest cumtime decrease, showing {args.top})",
            use_color,
        )
    )
    neg = [
        (k, bc, cc, dc, bt, ct, dt, bn, cn)
        for k, bc, cc, dc, bt, ct, dt, bn, cn in diff
        if dc < -args.min_delta
    ]
    neg.sort(key=lambda r: r[3])  # most negative first
    cols = [
        "Function",
        "base_cum",
        "new_cum",
        "Δcum",
        "base_tt",
        "new_tt",
        "Δtt",
        "base_nc",
        "new_nc",
    ]
    widths = [55, 9, 9, 10, 9, 9, 10, 8, 8]
    print_header(cols, widths, use_color)
    for k, bc, cc, dc, bt, ct, dt, bn, cn in neg[: args.top]:
        print(
            f"  {_fn(k)[:54]:<54}"
            f"  {bc:>9.4f}"
            f"  {cc:>9.4f}"
            f"  {fmt_delta(dc, use_color):>10}"
            f"  {bt:>9.4f}"
            f"  {ct:>9.4f}"
            f"  {fmt_delta(dt, use_color):>10}"
            f"  {bn:>8}"
            f"  {cn:>8}"
        )

    # -- new functions only in candidate --------------------------------
    print(
        section(
            f"NEW FUNCTIONS  (only in candidate, top {args.top} by cumtime)", use_color
        )
    )
    print_new_only(baseline, candidate, args.top, use_color)

    # -- call count changes ---------------------------------------------
    print(section(f"LARGEST CALL COUNT CHANGES  (top {args.top})", use_color))
    by_calls = sorted(diff, key=lambda r: abs(r[8] - r[7]), reverse=True)
    cols = ["Function", "base_nc", "new_nc", "Δcalls", "new_cumtime"]
    widths = [58, 8, 8, 10, 12]
    print_header(cols, widths, use_color)
    for k, bc, cc, dc, bt, ct, dt, bn, cn in by_calls[: args.top]:
        delta_nc = cn - bn
        if abs(delta_nc) < 1:
            continue
        col = RED if delta_nc > 0 else GREEN
        d_str = _color(f"{delta_nc:+,}", col, use_color)
        print(f"  {_fn(k)[:57]:<57}  {bn:>8}  {cn:>8}  {d_str:>10}  {cc:>12.4f}")

    # -- per-call cost changes ------------------------------------------
    print(
        section(
            f"LARGEST PER-CALL COST CHANGES  (top {args.top}, min 5 calls each)",
            use_color,
        )
    )
    per_call = []
    for k, bc, cc, dc, bt, ct, dt, bn, cn in diff:
        if bn >= 5 and cn >= 5:
            b_per = bt / bn
            c_per = ct / cn
            if b_per > 0:
                per_call.append((k, b_per, c_per, c_per / b_per, bn, cn))
    per_call.sort(key=lambda r: -r[3])
    cols = ["Function", "base_tt/call", "new_tt/call", "ratio", "base_nc", "new_nc"]
    widths = [55, 13, 13, 8, 8, 8]
    print_header(cols, widths, use_color)
    for k, bp, cp, ratio, bn, cn in per_call[: args.top]:
        print(
            f"  {_fn(k)[:54]:<54}"
            f"  {bp:>13.6f}"
            f"  {cp:>13.6f}"
            f"  {fmt_ratio(ratio, use_color):>8}"
            f"  {bn:>8}"
            f"  {cn:>8}"
        )

    # -- optional callers -----------------------------------------------
    if args.callers:
        for fn_name in args.callers:
            print(section(f"CALLERS of '{fn_name}' in candidate", use_color))
            print_callers(candidate, fn_name, use_color)

    print()


if __name__ == "__main__":
    main()
