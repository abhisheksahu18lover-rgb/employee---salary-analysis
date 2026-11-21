#!/usr/bin/env python3
"""Salary data analyzer for the Data Science Salaries 2024 dataset.

Reads cleaned_DataScience_salaries_2024.csv by default, falling back to
DataScience_salaries_2024.csv when the cleaned input is missing or invalid.
Writes charts and CSV/text reports into output/ (or --output-dir).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED = {"salary_in_usd", "job_title", "experience_level", "employment_type",
            "remote_ratio", "company_size", "company_location", "work_year"}
OPTIONAL = {"salary", "salary_currency", "employee_residence"}
EXPERIENCE_ORDER = ["EN", "MI", "SE", "EX"]
EXPERIENCE_NAMES = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior-level", "EX": "Executive-level"}
REMOTE_NAMES = {0: "On-site (0%)", 50: "Hybrid (50%)", 100: "Remote (100%)"}


def load_data(cleaned: Path, raw: Path) -> tuple[pd.DataFrame, Path, str]:
    errors = []
    for path, label in ((cleaned, "cleaned"), (raw, "raw fallback")):
        if not path.is_file():
            errors.append(f"{path}: file not found")
            continue
        try:
            df = pd.read_csv(path)
            missing = REQUIRED - set(df.columns)
            if missing:
                errors.append(f"{path}: missing required columns: {', '.join(sorted(missing))}")
                continue
            df["salary_in_usd"] = pd.to_numeric(df["salary_in_usd"], errors="coerce")
            df = df.dropna(subset=["salary_in_usd"])
            df = df[df["salary_in_usd"] > 0].copy()
            if df.empty:
                errors.append(f"{path}: no valid positive salary_in_usd values")
                continue
            return df, path, label
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    raise ValueError("Could not load a valid input dataset. " + " | ".join(errors))


def grouped(df: pd.DataFrame, column: str, min_count: int = 1) -> pd.DataFrame:
    out = df.groupby(column, dropna=False)["salary_in_usd"].agg(
        records="count", mean_usd="mean", median_usd="median", min_usd="min", max_usd="max"
    ).reset_index()
    return out[out.records >= min_count].sort_values("median_usd", ascending=False)


def money(v: float) -> str:
    return f"${v:,.0f}"


def chart_bar(frame: pd.DataFrame, label_col: str, value_col: str, title: str,
              path: Path, *, horizontal: bool = True, color: str = "#2878B5") -> None:
    if frame.empty:
        return
    labels = frame[label_col].astype(str)
    fig, ax = plt.subplots(figsize=(10, max(4, .42 * len(frame))))
    if horizontal:
        ax.barh(labels[::-1], frame[value_col].to_numpy()[::-1], color=color)
        ax.set_xlabel("Annual salary (USD)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    else:
        ax.bar(labels, frame[value_col], color=color)
        ax.set_ylabel("Annual salary (USD)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y/1000:.0f}k"))
        ax.tick_params(axis="x", rotation=25)
    ax.set_title(title, loc="left", weight="bold", pad=14)
    ax.grid(axis="x" if horizontal else "y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def analyze(df: pd.DataFrame, source: Path, source_label: str, outdir: Path, top_n: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Normalize category whitespace, preserve original values in the input file.
    for col in REQUIRED | OPTIONAL:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA
    df["salary_in_usd"] = pd.to_numeric(df.salary_in_usd, errors="coerce")
    df = df.dropna(subset=["salary_in_usd"])
    n = len(df)
    salary = df.salary_in_usd
    stats = {"records_analyzed": n, "mean_usd": salary.mean(), "median_usd": salary.median(),
             "std_dev_usd": salary.std(), "minimum_usd": salary.min(), "q1_usd": salary.quantile(.25),
             "q3_usd": salary.quantile(.75), "maximum_usd": salary.max()}
    pd.DataFrame([stats]).to_csv(outdir / "overall_salary_statistics.csv", index=False)

    # Detailed comparison tables; titles with at least 10 records are used for the chart.
    tables = [("job_title", "job_title_salary_analysis.csv", "Median USD salary by job title"),
              ("experience_level", "experience_level_salary_analysis.csv", "Salary by experience level"),
              ("employment_type", "employment_type_salary_analysis.csv", "Salary by employment type"),
              ("remote_ratio", "remote_work_salary_analysis.csv", "Salary by remote-work ratio"),
              ("company_size", "company_size_salary_analysis.csv", "Salary by company size"),
              ("company_location", "company_location_salary_analysis.csv", "Salary by company location")]
    results = {}
    for col, filename, title in tables:
        frame = grouped(df.dropna(subset=[col]), col, min_count=1)
        if col == "experience_level":
            frame["experience_name"] = frame[col].map(EXPERIENCE_NAMES).fillna(frame[col].astype(str))
            frame["_order"] = frame[col].map({v:i for i,v in enumerate(EXPERIENCE_ORDER)}).fillna(99)
            frame = frame.sort_values("_order").drop(columns="_order")
            chart_bar(frame, "experience_name", "median_usd", title, outdir / "experience_level_salary.png", horizontal=False)
        elif col == "remote_ratio":
            frame["remote_category"] = pd.to_numeric(frame[col], errors="coerce").map(REMOTE_NAMES).fillna(frame[col].astype(str))
            frame = frame.sort_values(col)
            chart_bar(frame, "remote_category", "median_usd", title, outdir / "remote_work_salary.png", horizontal=False, color="#31A354")
        else:
            chart_frame = frame.head(top_n)
            if col == "job_title":
                chart_frame = frame[frame.records >= 10].head(top_n)
            chart_bar(chart_frame, col, "median_usd", title,
                      outdir / {"job_title":"job_title_salary.png", "employment_type":"employment_type_salary.png",
                                "company_size":"company_size_salary.png", "company_location":"company_location_salary.png"}[col],
                      horizontal=col in ("job_title", "company_location"), color="#756BB1")
        frame.to_csv(outdir / filename, index=False)
        results[col] = frame

    # Salary distribution and annual trend.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(salary, bins=35, color="#2878B5", edgecolor="white")
    ax.axvline(salary.median(), color="#E6550D", linestyle="--", label=f"Median {money(salary.median())}")
    ax.set(title="Annual salary distribution", xlabel="Annual salary (USD)", ylabel="Number of records")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k")); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(outdir / "salary_distribution.png", dpi=170); plt.close(fig)
    yearly = df.groupby("work_year").salary_in_usd.agg(records="count", mean_usd="mean", median_usd="median").reset_index().sort_values("work_year")
    yearly.to_csv(outdir / "salary_by_work_year.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(yearly.work_year, yearly.median_usd, marker="o", linewidth=2.5, color="#2878B5")
    ax.set(title="Median salary by work year", xlabel="Work year", ylabel="Median annual salary (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y/1000:.0f}k")); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(outdir / "salary_by_work_year.png", dpi=170); plt.close(fig)

    # Report notes keep currency interpretation explicit and useful for a student project.
    top_jobs = results["job_title"].query("records >= 10").head(5)
    exp = results["experience_level"].copy()
    exp["name"] = exp.experience_level.map(EXPERIENCE_NAMES).fillna(exp.experience_level.astype(str))
    remote = results["remote_ratio"].copy()
    remote["name"] = pd.to_numeric(remote.remote_ratio, errors="coerce").map(REMOTE_NAMES).fillna(remote.remote_ratio.astype(str))
    by_size = results["company_size"].head(1)
    by_loc = results["company_location"].query("records >= 10").head(5)
    lines = ["DATA SCIENCE SALARY ANALYSIS", "=" * 31, "",
             f"Input used: {source.name} ({source_label})", f"Rows analyzed: {n:,}",
             f"Columns: {', '.join(df.columns)}", "",
             "SALARY SUMMARY (salary_in_usd)", "-" * 31,
             f"Mean: {money(stats['mean_usd'])}", f"Median: {money(stats['median_usd'])}",
             f"Standard deviation: {money(stats['std_dev_usd'])}", f"Range: {money(stats['minimum_usd'])} to {money(stats['maximum_usd'])}",
             f"Middle 50%: {money(stats['q1_usd'])} to {money(stats['q3_usd'])}", "",
             "All comparisons use salary_in_usd (annual USD equivalent). The raw salary column may contain different currencies and is not directly comparable.", "",
             "EXPERIENCE LEVEL (median USD)", "-" * 31]
    for _, r in exp.iterrows(): lines.append(f"{r['name']}: {money(r.median_usd)} (n={int(r.records):,})")
    lines += ["", "REMOTE WORK (median USD)", "-" * 31]
    for _, r in remote.iterrows(): lines.append(f"{r['name']}: {money(r.median_usd)} (n={int(r.records):,})")
    lines += ["", f"TOP JOB TITLES (at least 10 records; top {top_n} shown)", "-" * 31]
    lines += [f"{r.job_title}: {money(r.median_usd)} median (n={int(r.records):,})" for _,r in top_jobs.iterrows()] or ["No job titles met the minimum count."]
    lines += ["", "COMPANY SIZE (highest median)", "-" * 31]
    lines += [f"{r.company_size}: {money(r.median_usd)} median (n={int(r.records):,})" for _,r in by_size.iterrows()]
    lines += ["", "COMPANY LOCATIONS (at least 10 records; highest medians)", "-" * 31]
    lines += [f"{r.company_location}: {money(r.median_usd)} median (n={int(r.records):,})" for _,r in by_loc.iterrows()] or ["No locations met the minimum count."]
    lines += ["", "INTERPRETATION NOTES", "-" * 31,
              "Group medians describe the records in this dataset; they do not establish that a category causes higher pay.",
              "Job titles and locations with fewer than 10 records are retained in the CSV tables but omitted from selected charts and highlights.",
              "Employment type, remote ratio, company size, and country labels are interpreted as supplied by the dataset."]
    (outdir / "analysis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {n:,} records from {source} ({source_label}).")
    print(f"Generated report, CSV summaries, and charts in {outdir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=Path("cleaned_DataScience_salaries_2024.csv"), help="Preferred cleaned CSV input")
    parser.add_argument("--raw", type=Path, default=Path("DataScience_salaries_2024.csv"), help="Raw CSV fallback")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for report, tables, and charts")
    parser.add_argument("--top-n", type=int, default=15, help="Maximum categories shown in category charts/highlights")
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    try:
        df, source, label = load_data(args.cleaned, args.raw)
        analyze(df, source, label, args.output_dir, args.top_n)
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
