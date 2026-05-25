import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

# ANSI colors
GREEN, RED, CYAN, BOLD, DIM, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[36m",
    "\033[1m",
    "\033[2m",
    "\033[0m",
)

TOTAL_WIDTH = 76


def color(s, color):
    return f"{color}{s}{RESET}"


green, red, cyan, bold, dim = [lambda s, c=c: color(s, c) for c in [GREEN, RED, CYAN, BOLD, DIM]]


def visible_len(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def pad(s, w, align="r"):
    return (
        s + " " * max(0, w - visible_len(s))
        if align == "r"
        else " " * max(0, w - visible_len(s)) + s
    )


def print_panel(lines, title, subtitle):
    inner = TOTAL_WIDTH - 2
    title_str, sub_str = f"─ {bold(title)} ", f" {dim(subtitle)} ─"
    dashes = "─" * max(0, inner - (4 + visible_len(title)) - (3 + visible_len(subtitle)))
    print(f"╭{title_str}{dashes}{sub_str} ╮")
    for line in lines:
        print(f"│{pad(f' {line}', inner)}│")
    print(f"╰{'─' * inner}╯")


def load_data(dir_path):
    """Load all JSONL files into a single Polars DataFrame."""
    records = []
    for f in sorted(Path(dir_path).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec.setdefault("run_id", "unknown")
                rec.setdefault("bench_file", "unknown")
                for row in rec["rows"]:
                    records.append({
                        "bench_file": rec["bench_file"],
                        "bench_title": rec["title"],
                        "field": rec["row_field"],
                        "field_value": row["field_value"],
                        "run_id": rec["run_id"],
                        "mean_us": row["mean_us"],
                        "median_us": row["median_us"],
                        "min_us": row["min_us"],
                        "max_us": row["max_us"],
                    })

    if not records:
        return pl.DataFrame()
    df = pl.DataFrame(records)
    # Add run label
    runs = df.select("run_id").unique().sort("run_id").with_row_index("run_num", offset=1)
    df = df.join(runs, on="run_id").with_columns(
        pl.concat_str([pl.lit("R"), pl.col("run_num")]).alias("run_label")
    )
    return df


def render_table(df, bench_title):
    """Render a single benchmark table."""
    bench_df = df.filter(pl.col("bench_title") == bench_title)
    if bench_df.is_empty():
        return []

    # Get stats for highlighting
    stats = bench_df.select([
        pl.col("mean_us").min().alias("mean_min"),
        pl.col("mean_us").max().alias("mean_max"),
        pl.col("median_us").min().alias("median_min"),
        pl.col("median_us").max().alias("median_max"),
        pl.col("min_us").min().alias("min_min"),
        pl.col("max_us").max().alias("max_max"),
    ]).row(0)

    has_variation = bench_df["mean_us"].n_unique() > 1

    # Collect data and format in Python (simpler!)
    data = bench_df.select([
        "run_label",
        "field_value",
        "mean_us",
        "median_us",
        "min_us",
        "max_us",
    ]).rows()

    # Helper for formatting values
    def fmt_val(val, min_val, max_val):
        if not has_variation:
            return f"{val:.2f}"
        if val == min_val:
            return green(f"{val:.2f}")
        if val == max_val:
            return red(f"{val:.2f}")
        return f"{val:.2f}"

    field_label = bench_df["field"][0].replace("_", " ").capitalize()

    # Build table
    header = " ".join([
        cyan(pad("Run", 5)),
        cyan(pad(field_label, 8)),
        cyan(pad("Mean (µs)", 12, "l")),
        cyan(pad("Median (µs)", 12, "l")),
        cyan(pad("Min (µs)", 10, "l")),
        cyan(pad("Max (µs)", 10, "l")),
    ])

    lines = [header]
    for run, field_val, mean, median, min_val, max_val in data:
        lines.append(
            " ".join([
                dim(pad(run, 5)),
                dim(pad(str(field_val), 8)),
                pad(fmt_val(mean, stats[0], stats[1]), 12, "l"),
                pad(fmt_val(median, stats[2], stats[3]), 12, "l"),
                pad(green(f"{min_val:.2f}") if min_val == stats[4] else f"{min_val:.2f}", 10, "l"),
                pad(red(f"{max_val:.2f}") if max_val == stats[5] else f"{max_val:.2f}", 10, "l"),
            ])
        )

    return lines


def delete_benchmark(dir_path, bench_title):
    """Delete all records matching a benchmark title from JSONL files."""
    deleted = 0
    for jsonl_file in sorted(Path(dir_path).glob("*.jsonl")):
        lines = jsonl_file.read_text().splitlines()
        filtered = []
        removed = 0
        for line in lines:
            if not line.strip():
                filtered.append(line)
                continue
            rec = json.loads(line)
            if bench_title == "__ALL__" or rec.get("title") == bench_title:
                removed += 1
            else:
                filtered.append(line)
        if removed > 0:
            with open(jsonl_file, "w") as f:
                f.write("\n".join(filtered))
            deleted += removed

    return deleted


def clear_screen():
    """Clear the terminal screen cross-platform."""
    cmd = "cls" if sys.platform == "win32" else "clear"
    subprocess.run([cmd], shell=True, check=False)


def refresh_state(df, args):
    df = load_data(args.dir)
    groups = df.select("bench_file").unique().sort("bench_file")["bench_file"].to_list()
    if args.group:
        groups = [g for g in args.group if g in groups]
        df = df.filter(pl.col("bench_file").is_in(groups))
    benchmarks = df.select("bench_title").unique().sort("bench_title")["bench_title"].to_list()
    return df, groups, benchmarks


def render_all_groups(df, groups):
    for group in groups:
        group_df = df.filter(pl.col("bench_file") == group)
        group_benches = (
            group_df.select("bench_title").unique().sort("bench_title")["bench_title"].to_list()
        )
        print()
        print(bold(f"── {group} ──────────────────────────────────────"))
        for bench in group_benches:
            lines = render_table(group_df, bench)
            if lines:
                field = group_df.filter(pl.col("bench_title") == bench)["field"][0]
                print_panel(lines, bench, f"{field} · {len(lines) - 1} rows")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default="training/benchmarks")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output", default="benchmarks_results.csv")
    parser.add_argument(
        "--group", nargs="+", help="Filter by benchmark groups (e.g., mixing models)"
    )
    args = parser.parse_args()

    df = load_data(args.dir)
    if df.is_empty():
        print(red("No benchmark results found."))
        return

    all_groups = df.select("bench_file").unique().sort("bench_file")["bench_file"].to_list()
    groups = [g for g in args.group] if args.group else all_groups
    df = df.filter(pl.col("bench_file").is_in(groups))

    if df.is_empty():
        requested = ", ".join(args.group) if args.group else "(none)"
        print(red(f"No benchmark results found for group(s): {requested}"))
        return

    benchmarks = df.select("bench_title").unique().sort("bench_title")["bench_title"].to_list()

    if args.once:
        print(bold("Benchmark Results Viewer"))
        print(
            dim(
                f"Source: {args.dir} · Groups: {', '.join(groups)} · Runs: {df.select('run_label').n_unique()}"
            )
        )
        render_all_groups(df, groups)
        return

    # TUI mode
    clear_screen()
    print(bold("Benchmark Results Viewer"))
    print(
        dim(
            f"Source: {args.dir} · Groups: {', '.join(groups)} · Runs: {df.select('run_label').n_unique()}"
        )
    )

    while True:
        render_all_groups(df, groups)

        print()
        answer = input("Export (e) / Delete (d) / Quit (q): ").lower()
        if answer == "e":
            df.write_csv(args.output)
            print(green(f"Exported to {args.output}"))
            break
        if answer == "d":
            print()
            print("  0. Delete all benchmarks")
            for i, bench in enumerate(benchmarks, 1):
                print(f"  {i}. {bench}")
            print()
            choice = input("Enter benchmark number or name to delete: ").strip()
            if choice == "0" or choice.lower() == "all":
                deleted = delete_benchmark(args.dir, "__ALL__")
                print(green(f"Deleted all {deleted} records"))
                df, groups, benchmarks = refresh_state(df, args)
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(benchmarks):
                        bench_title = benchmarks[idx]
                    else:
                        print(red(f"Invalid number: {choice}"))
                        continue
                except ValueError:
                    bench_title = choice
                deleted = delete_benchmark(args.dir, bench_title)
                if deleted > 0:
                    print(green(f"Deleted {deleted} records for '{bench_title}'"))
                    df, groups, benchmarks = refresh_state(df, args)
                else:
                    print(red(f"No benchmark found matching '{bench_title}'"))
        if answer == "q":
            break
        clear_screen()
        print(bold("Benchmark Results Viewer"))
        print(
            dim(
                f"Source: {args.dir} · Groups: {', '.join(groups)} · Runs: {df.select('run_label').n_unique()}"
            )
        )


if __name__ == "__main__":
    main()
