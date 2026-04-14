"""
End-to-end social media usage analysis.
Outputs: cleaned CSV, figures/, optional Plotly HTML dashboard.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_IN = ROOT.parent / "social_media_usage.csv"
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = [
    "Daily_Minutes_Spent",
    "Posts_Per_Day",
    "Likes_Per_Day",
    "Follows_Per_Day",
]
PLATFORM_ORDER = [
    "Facebook",
    "Instagram",
    "Twitter",
    "Snapchat",
    "TikTok",
    "LinkedIn",
    "Pinterest",
]


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def cap_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["_outlier_flag_any"] = False
    for c in cols:
        lo, hi = iqr_bounds(df[c].astype(float))
        mask = (df[c] < lo) | (df[c] > hi)
        df.loc[mask, f"{c}_capped"] = True
        df.loc[mask, c] = df.loc[mask, c].clip(lower=lo, upper=hi)
        df["_outlier_flag_any"] = df["_outlier_flag_any"] | mask
    cap_cols = [f"{c}_capped" for c in cols]
    df["_outlier_flag_any"] = df[cap_cols].any(axis=1)
    df.drop(columns=cap_cols, inplace=True)
    return df


def main() -> None:
    if not DATA_IN.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA_IN}")

    raw = pd.read_csv(DATA_IN)
    report_lines: list[str] = []

    # --- Data understanding ---
    report_lines.append("=== DATA UNDERSTANDING ===\n")
    report_lines.append(f"Rows: {len(raw)}, Columns: {list(raw.columns)}\n")
    report_lines.append(f"Missing per column:\n{raw.isna().sum().to_string()}\n")
    report_lines.append(f"Dtypes:\n{raw.dtypes.to_string()}\n")

    df = raw.copy()
    df["App"] = df["App"].astype(str).str.strip()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_missing = df[NUM_COLS].isna().any(axis=1).sum()
    if n_missing:
        df = df.dropna(subset=NUM_COLS)
        report_lines.append(f"Dropped {n_missing} rows with missing numerics.\n")
    else:
        report_lines.append("No missing values in numeric features.\n")

    df_clean = cap_outliers(df, NUM_COLS)
    report_lines.append(
        f"Capped numeric outliers (Tukey IQR 1.5). Rows flagged: {df_clean['_outlier_flag_any'].sum()}.\n"
    )

    # Activity index for ranking & segmentation (equal weight, scaled)
    z = df_clean[NUM_COLS].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    df_clean["Activity_Score"] = z.mean(axis=1)
    q33, q66 = df_clean["Activity_Score"].quantile([1 / 3, 2 / 3])

    def segment(score: float) -> str:
        if score <= q33:
            return "Low"
        if score <= q66:
            return "Medium"
        return "High"

    df_clean["Activity_Segment"] = df_clean["Activity_Score"].map(segment)
    df_clean = df_clean.drop(columns=["_outlier_flag_any"], errors="ignore")
    df_clean.to_csv(OUT_DIR / "social_media_usage_cleaned.csv", index=False)

    # --- EDA aggregates ---
    plat_time = (
        df_clean.groupby("App")["Daily_Minutes_Spent"]
        .mean()
        .reindex([p for p in PLATFORM_ORDER if p in df_clean["App"].unique()])
        .sort_values(ascending=False)
    )
    top_time_platform = plat_time.index[0]
    report_lines.append(
        f"\n=== EDA ===\nHighest avg daily minutes: {top_time_platform} ({plat_time.iloc[0]:.1f} min).\n"
    )

    eng_cols = ["Likes_Per_Day", "Follows_Per_Day"]
    eng = df_clean.groupby("App")[eng_cols].mean()
    plat_eng_order = [p for p in PLATFORM_ORDER if p in eng.index]
    eng = eng.reindex(plat_eng_order)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5))
    plat_time.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
    ax.set_title("Average Daily Time Spent by Platform")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Minutes / day")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_platform_avg_time.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(eng))
    w = 0.35
    ax.bar(x - w / 2, eng["Likes_Per_Day"], w, label="Avg likes/day", color="#55A868")
    ax.bar(x + w / 2, eng["Follows_Per_Day"], w, label="Avg follows/day", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(eng.index.tolist(), rotation=25, ha="right")
    ax.set_title("Engagement by Platform (Daily Averages)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_platform_engagement_bars.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_clean["Posts_Per_Day"], bins=20, kde=True, ax=ax, color="#8172B2")
    ax.set_title("Distribution of Posts per Day")
    ax.set_xlabel("Posts per day")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_posts_per_day_hist.png", dpi=150)
    plt.close()

    # Segment distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    seg_counts = df_clean["Activity_Segment"].value_counts().reindex(["Low", "Medium", "High"])
    seg_counts.plot(kind="bar", ax=ax, color=["#8DA0CB", "#E78AC3", "#A6D854"], edgecolor="black")
    ax.set_title("User Activity Segments")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Users")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_activity_segments.png", dpi=150)
    plt.close()

    # Correlation heatmap
    corr = df_clean[NUM_COLS].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df_clean,
        x="Daily_Minutes_Spent",
        y="Likes_Per_Day",
        hue="App",
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("Time Spent vs Likes (by Platform)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_scatter_time_vs_likes.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df_clean,
        x="Posts_Per_Day",
        y="Follows_Per_Day",
        hue="App",
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("Posts per Day vs Followers Gained (by Platform)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_scatter_posts_vs_follows.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top users
    top10 = df_clean.nlargest(10, "Activity_Score")[
        ["User_ID", "App", "Daily_Minutes_Spent", "Posts_Per_Day", "Likes_Per_Day", "Follows_Per_Day", "Activity_Score"]
    ]
    top10.to_csv(OUT_DIR / "top_10_active_users.csv", index=False)

    r_time_likes = df_clean["Daily_Minutes_Spent"].corr(df_clean["Likes_Per_Day"])
    r_posts_follows = df_clean["Posts_Per_Day"].corr(df_clean["Follows_Per_Day"])
    report_lines.append(
        f"\n=== CORRELATIONS ===\n"
        f"Time spent vs Likes (Pearson): {r_time_likes:.3f}\n"
        f"Posts vs Follows gained: {r_posts_follows:.3f}\n"
    )

    report_text = "\n".join(report_lines)
    (OUT_DIR / "analysis_summary.txt").write_text(report_text, encoding="utf-8")

    # Optional Plotly HTML
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plat_sorted = (
            df_clean.groupby("App")["Daily_Minutes_Spent"].mean().sort_values(ascending=True)
        )
        fig_dash = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Avg daily minutes by platform",
                "Avg likes & follows by platform",
                "Posts per day (histogram)",
                "Time vs likes (all users)",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )
        fig_dash.add_trace(
            go.Bar(x=plat_sorted.values, y=plat_sorted.index, orientation="h", name="Minutes"),
            row=1,
            col=1,
        )
        apps_list = plat_time.index.tolist()
        fig_dash.add_trace(
            go.Bar(name="Likes/day", x=apps_list, y=df_clean.groupby("App")["Likes_Per_Day"].mean().reindex(apps_list).values),
            row=1,
            col=2,
        )
        fig_dash.add_trace(
            go.Bar(name="Follows/day", x=apps_list, y=df_clean.groupby("App")["Follows_Per_Day"].mean().reindex(apps_list).values),
            row=1,
            col=2,
        )
        fig_dash.add_trace(
            go.Histogram(x=df_clean["Posts_Per_Day"], nbinsx=20, name="Posts"),
            row=2,
            col=1,
        )
        fig_dash.add_trace(
            go.Scatter(
                x=df_clean["Daily_Minutes_Spent"],
                y=df_clean["Likes_Per_Day"],
                mode="markers",
                marker=dict(color=df_clean["App"].astype("category").cat.codes, opacity=0.5),
                text=df_clean["App"],
                hovertemplate="App=%{text}<br>Minutes=%{x}<br>Likes=%{y}<extra></extra>",
                name="Users",
            ),
            row=2,
            col=2,
        )
        fig_dash.update_layout(
            height=900,
            width=1100,
            title_text="Social Media Usage — Interactive Summary",
            showlegend=True,
        )
        fig_dash.write_html(str(OUT_DIR / "dashboard_plotly.html"), include_plotlyjs="cdn")
    except Exception as e:
        (OUT_DIR / "dashboard_note.txt").write_text(
            f"Plotly dashboard skipped: {e}", encoding="utf-8"
        )

    print(report_text)
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
