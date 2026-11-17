"""Figures for the QALMA variational MF paper.

Reads benchmark_results/variational_mf_paper_results.json and produces:

  Figure 1 — Free energy vs beta for all models (Family 1 validation)
             Panel a: F(mixed), F(SC), F(var) vs beta at fixed L
             Panel b: T-score for SC and variational vs beta at fixed L
                      (shows improvement of variational over SC)

  Figure 2 — T-score vs model at fixed (L, beta)
             One panel per model: T-score(SC) and T-score(var) vs L,
             showing the systematic improvement of the variational method.

  Figure 3 — Free energy F vs numfields for the J1-J2 chain (Family 2)
             Panel a: F curves for each J2/J1 at fixed L and beta
             Panel b: magnetization pattern <Sz_i> for selected numfields

  Figure 4 — T-score vs numfields for the J1-J2 chain
             One curve per J2/J1; shows convergence of the T-score
             diagnostic as more fields are included.

  Figure 5 — Phase diagram proxy: Delta_F = F(nf=1) - F(nf=max) vs J2/J1
             Shows where extra fields matter most (frustration detector)

Usage:
    python plot_variational_mf_paper.py                     # reads default JSON
    python plot_variational_mf_paper.py path/to/results.json
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style — PRL/PRB compatible
# ---------------------------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,  # set True if LaTeX is available
    }
)

# Single-column width (PRL): 3.375 in
# Double-column width (PRL): 6.75 in
COL1 = 3.375
COL2 = 6.75

# Color palette — colorblind friendly
COLORS = {
    "mixed": "#AAAAAA",
    "sc": "#E87833",
    "var": "#1F77B4",
}

MARKERS = {
    "mixed": "s",
    "sc": "^",
    "var": "o",
}

# J2/J1 color gradient for Figures 3–5
J2_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 7))

# Short model labels for legends
SHORT_LABELS = {
    "Ising transverse (Gamma=0.5J)": r"Ising $(\Gamma=0.5J)$",
    "Ising transverse critical (Gamma=J)": r"Ising critical $(\Gamma=J)$",
    "XX chain": "XX",
    "XXX Heisenberg AFM": "XXX AFM",
    "XXX Heisenberg FM": "XXX FM",
    "XYZ anisotropic (Jz=1, Jxy=0.5)": "XYZ",
}


# ---------------------------------------------------------------------------
# Data loading and indexing
# ---------------------------------------------------------------------------


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def index_exact(data: list) -> dict:
    """Index exact_validation results as {(label, L, beta): row}."""
    idx = {}
    for row in data:
        key = (row["label"], row["L"], row["beta"])
        idx[key] = row
    return idx


def index_nf(data: list) -> dict:
    """Index numfields_convergence results as {(J2_ratio, L, beta, nf): row}."""
    idx = {}
    for row in data:
        key = (row["J2_over_J1"], row["L"], row["beta"], row["numfields"])
        idx[key] = row
    return idx


def _safe_tscore(row: dict, key: str):
    """Return T-score value or NaN when absent/None."""
    val = row.get(key)
    return float(val) if val is not None else float("nan")


# ---------------------------------------------------------------------------
# Figure 1 — Free energy + T-score vs beta (representative model, fixed L)
# ---------------------------------------------------------------------------


def plot_figure1(exact_data: list, out_dir: Path):
    """Two panels for a representative model (Ising transverse, L=8):

    (a) Free energy F for mixed, SC, and variational states vs beta.
    (b) T-score for SC and variational vs beta — quantifies improvement.
    """
    idx = index_exact(exact_data)

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.44))

    model_label = "Ising transverse (Gamma=0.5J)"
    L_fixed = 8
    betas = sorted({row["beta"] for row in exact_data if row["label"] == model_label})

    f_mixed_vals, f_sc_vals, f_var_vals = [], [], []
    t_sc_vals, t_var_vals = [], []
    valid_betas = []

    for beta in betas:
        key = (model_label, L_fixed, beta)
        if key not in idx:
            continue
        row = idx[key]
        f_mixed_vals.append(row["F_mixed"])
        f_sc_vals.append(row["F_sc"])
        f_var_vals.append(row["F_variational"])
        t_sc_vals.append(_safe_tscore(row, "T_score_sc"))
        t_var_vals.append(_safe_tscore(row, "T_score_variational"))
        valid_betas.append(beta)

    # ---- Panel (a): Free energy vs beta ----------------------------------
    ax = axes[0]
    if valid_betas:
        ax.plot(
            valid_betas,
            f_mixed_vals,
            color=COLORS["mixed"],
            marker=MARKERS["mixed"],
            label="Mixed state",
            linestyle="--",
        )
        ax.plot(
            valid_betas,
            f_sc_vals,
            color=COLORS["sc"],
            marker=MARKERS["sc"],
            label="Self-consistent MF",
        )
        ax.plot(
            valid_betas,
            f_var_vals,
            color=COLORS["var"],
            marker=MARKERS["var"],
            label="Variational MF",
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$F[\sigma]$")
    ax.set_title(f"Ising transverse ($\\Gamma=0.5J$), $L={L_fixed}$")
    ax.legend()
    ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    # ---- Panel (b): T-score vs beta --------------------------------------
    ax = axes[1]
    if valid_betas:
        ax.plot(
            valid_betas,
            t_sc_vals,
            color=COLORS["sc"],
            marker=MARKERS["sc"],
            label="Self-consistent MF",
        )
        ax.plot(
            valid_betas,
            t_var_vals,
            color=COLORS["var"],
            marker=MARKERS["var"],
            label="Variational MF",
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$T_{\rm score}$")
    ax.set_title(f"T-score, Ising transverse ($\\Gamma=0.5J$), $L={L_fixed}$")
    ax.set_yscale("log")
    ax.legend()
    ax.text(-0.18, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "fig1_freeenergy_and_tscore_vs_beta.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — T-score: SC vs variational across all models and system sizes
# ---------------------------------------------------------------------------


def plot_figure2(exact_data: list, out_dir: Path):
    """T-score(SC) and T-score(var) vs L for each model at fixed beta.

    Panels arranged as a grid: one row per beta value, one column per model.
    Shows systematically how much the variational method reduces the T-score
    relative to the plain self-consistent solution.
    """
    idx = index_exact(exact_data)

    models_to_show = [
        "Ising transverse (Gamma=0.5J)",
        "XX chain",
        "XXX Heisenberg AFM",
        "XXX Heisenberg FM",
        "XYZ anisotropic (Jz=1, Jxy=0.5)",
    ]
    betas_to_show = [1.0, 2.0, 5.0]

    Ls_all = sorted({row["L"] for row in exact_data})

    n_models = len(models_to_show)
    n_betas = len(betas_to_show)

    fig, axes = plt.subplots(
        n_betas,
        n_models,
        figsize=(COL2, COL2 * 0.55 * n_betas / 2),
        sharex=True,
        sharey="row",
    )

    panel_labels = iter("abcdefghijklmnopqrstuvwxyz")

    for row_i, beta in enumerate(betas_to_show):
        for col_j, model_label in enumerate(models_to_show):
            ax = axes[row_i][col_j]
            label = next(panel_labels)

            Ls, t_sc_vals, t_var_vals = [], [], []
            for L in Ls_all:
                key = (model_label, L, beta)
                if key not in idx:
                    continue
                row = idx[key]
                ts = _safe_tscore(row, "T_score_sc")
                tv = _safe_tscore(row, "T_score_variational")
                if not (np.isnan(ts) and np.isnan(tv)):
                    Ls.append(L)
                    t_sc_vals.append(ts)
                    t_var_vals.append(tv)

            if Ls:
                ax.plot(
                    Ls,
                    t_sc_vals,
                    color=COLORS["sc"],
                    marker=MARKERS["sc"],
                    label="SC",
                )
                ax.plot(
                    Ls,
                    t_var_vals,
                    color=COLORS["var"],
                    marker=MARKERS["var"],
                    label="Variational",
                )
                # Improvement ratio as shaded area
                t_sc_arr = np.array(t_sc_vals, dtype=float)
                t_var_arr = np.array(t_var_vals, dtype=float)
                valid = (~np.isnan(t_sc_arr)) & (~np.isnan(t_var_arr))
                if valid.any():
                    ax.fill_between(
                        np.array(Ls)[valid],
                        t_var_arr[valid],
                        t_sc_arr[valid],
                        alpha=0.12,
                        color=COLORS["var"],
                        label="_nolegend_",
                    )

            # Column headers (model names) on top row only
            if row_i == 0:
                ax.set_title(SHORT_LABELS.get(model_label, model_label), fontsize=8)

            # Row labels (beta) on leftmost column only
            if col_j == 0:
                ax.set_ylabel(f"$T_{{\\rm score}}$\n($\\beta={beta}$)", fontsize=8)

            # x-axis label on bottom row only
            if row_i == n_betas - 1:
                ax.set_xlabel("$L$", fontsize=8)

            ax.set_yscale("log")
            ax.text(
                -0.18,
                1.02,
                f"({label})",
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=7,
            )

            # Legend only in top-left panel
            if row_i == 0 and col_j == 0:
                ax.legend(fontsize=7)

    fig.suptitle(
        "T-score: self-consistent vs variational MF",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "fig2_tscore_sc_vs_var_all_models.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Free energy F vs numfields (J1-J2 chain)
# ---------------------------------------------------------------------------


def plot_figure3(
    nf_data: list, out_dir: Path, L_plot: int = 12, beta_plot: float = 5.0
):
    """Two panels:
    (a) F vs numfields for each J2/J1 at fixed L and beta.
    (b) Magnetization pattern <Sz_i> for max frustration (J2/J1=0.5)
        comparing nf=1, 4, 10.
    """
    idx = index_nf(nf_data)

    j2_ratios = sorted({r["J2_over_J1"] for r in nf_data})
    nf_vals = sorted({r["numfields"] for r in nf_data})

    fig = plt.figure(figsize=(COL2, COL2 * 0.44))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ---- Panel (a): F vs numfields ----------------------------------------
    for i, j2r in enumerate(j2_ratios):
        nfs, fs = [], []
        for nf in nf_vals:
            key = (j2r, L_plot, beta_plot, nf)
            if key in idx:
                nfs.append(nf)
                fs.append(idx[key]["f"])
        if nfs:
            ax_left.plot(
                nfs,
                fs,
                color=J2_COLORS[i % len(J2_COLORS)],
                marker="o",
                label=f"$J_2/J_1={j2r:.1f}$",
            )

    ax_left.set_xlabel("Number of fields $m$")
    ax_left.set_ylabel(r"$F[\sigma_m]$")
    ax_left.set_title(rf"J1-J2 chain, $L={L_plot}$, $\beta={beta_plot}$")
    ax_left.legend(fontsize=7, ncol=2)
    ax_left.text(-0.18, 1.02, "(a)", transform=ax_left.transAxes, fontweight="bold")

    # ---- Panel (b): magnetization pattern ---------------------------------
    j2_frustrated = 0.6
    sites_plotted = False
    for nf, ls, lw in [(1, "--", 1.0), (4, "-.", 1.2), (10, "-", 1.5)]:
        key = (j2_frustrated, L_plot, beta_plot, nf)
        if key not in idx:
            continue
        mag = idx[key]["magnetization"]
        sites = list(range(1, len(mag) + 1))
        ax_right.plot(
            sites,
            mag,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=4,
            label=f"$m={nf}$",
            color=plt.cm.Blues(0.4 + 0.2 * (nf == 10)),
        )
        sites_plotted = True

    if sites_plotted:
        ax_right.axhline(0, color="0.7", linewidth=0.6)
        ax_right.set_xlabel("Site $i$")
        ax_right.set_ylabel(r"$\langle S^z_i \rangle$")
        ax_right.set_title(
            f"Magnetization, $J_2/J_1={j2_frustrated}$, "
            f"$L={L_plot}$, $\\beta={beta_plot}$ (spiral phase)"
        )
        ax_right.legend()
    else:
        ax_right.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax_right.transAxes,
            color="0.5",
        )

    ax_right.text(-0.18, 1.02, "(b)", transform=ax_right.transAxes, fontweight="bold")

    fig.tight_layout()
    suffix = f"L{L_plot}_b{int(beta_plot)}"
    out = out_dir / f"fig3_freeenergy_vs_numfields_{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Variance ratio vs numfields (J1-J2 chain)
# ---------------------------------------------------------------------------


def plot_figure4(
    nf_data: list, out_dir: Path, L_plot: int = 12, beta_plot: float = 5.0
):
    """Variance ratio R_m = Var[sigma_m] / Var[sigma_SC] vs numfields.

    For large systems where F_exact is unavailable, R_m is a natural
    convergence diagnostic:

    * R_m = 1  at nf = 0 (the SC baseline, by construction).
    * R_m < 1  means the variational state sigma_m has smaller fluctuations
      of hat{F} than the SC state — a direct measure of improvement.
    * R_m -> 0 would indicate convergence to the exact Gibbs state in
      distribution.

    Two panels:
    (a) R_m vs numfields — one curve per J2/J1 ratio.
    (b) Absolute Var[sigma_m] vs numfields on a log scale, showing the
        raw scale of residual fluctuations across frustration regimes.
    """
    idx = index_nf(nf_data)

    j2_ratios = sorted({r["J2_over_J1"] for r in nf_data})
    nf_vals = sorted({r["numfields"] for r in nf_data})

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.44))

    # ---- Panel (a): variance ratio R_m vs numfields -----------------------
    ax = axes[0]

    for i, j2r in enumerate(j2_ratios):
        nfs, ratios = [], []
        for nf in nf_vals:
            key = (j2r, L_plot, beta_plot, nf)
            if key not in idx:
                continue
            val = idx[key].get("var_f_ratio")
            if val is not None:
                nfs.append(nf)
                ratios.append(val)
        if len(nfs) < 2:
            continue
        ax.plot(
            nfs,
            ratios,
            color=J2_COLORS[i % len(J2_COLORS)],
            marker="o",
            label=f"$J_2/J_1={j2r:.1f}$",
        )

    ax.axhline(1.0, color="0.55", linewidth=0.8, linestyle="--", label="SC baseline")
    ax.set_xlabel("Number of fields $m$")
    ax.set_ylabel(
        r"$R_m = \mathrm{Var}[\sigma_m]\,/\,\mathrm{Var}[\sigma_{\mathrm{SC}}]$"
    )
    ax.set_title(rf"Variance ratio, $L={L_plot}$, $\beta={beta_plot}$")
    ax.legend(fontsize=7, ncol=2)
    ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    # ---- Panel (b): absolute Var[sigma_m] on log scale --------------------
    ax = axes[1]

    for i, j2r in enumerate(j2_ratios):
        nfs, var_fs = [], []
        for nf in nf_vals:
            key = (j2r, L_plot, beta_plot, nf)
            if key not in idx:
                continue
            val = idx[key].get("var_f")
            if val is not None:
                nfs.append(nf)
                var_fs.append(val)
        if len(nfs) < 2:
            continue
        ax.plot(
            nfs,
            var_fs,
            color=J2_COLORS[i % len(J2_COLORS)],
            marker="o",
            label=f"$J_2/J_1={j2r:.1f}$",
        )

    ax.set_xlabel("Number of fields $m$")
    ax.set_ylabel(r"$\mathrm{Var}_{\sigma_m}[\hat{F}]$")
    ax.set_title(rf"Absolute variance, $L={L_plot}$, $\beta={beta_plot}$")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.text(-0.18, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    suffix = rf"L{L_plot}_b{int(beta_plot)}"
    out = out_dir / f"fig4_variance_ratio_vs_numfields_{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Frustration detector: Delta F vs J2/J1
# ---------------------------------------------------------------------------


def plot_figure5(nf_data: list, out_dir: Path):
    """Delta F = F(nf=1) - F(nf=max) as a function of J2/J1.

    Large Delta_F means many variational fields are needed to converge
    the free energy — a signature of frustrated regions.
    One curve per (L, beta) combination.
    """
    idx = index_nf(nf_data)
    nf_min = 1
    nf_max = max(r["numfields"] for r in nf_data)

    j2_ratios = sorted({r["J2_over_J1"] for r in nf_data})
    Ls = sorted({r["L"] for r in nf_data})
    betas = sorted({r["beta"] for r in nf_data})

    fig, ax = plt.subplots(figsize=(COL1 * 1.3, COL1))

    ls_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D"]
    color_cycle = plt.cm.tab10(np.linspace(0, 0.5, len(Ls) * len(betas)))

    idx_color = 0
    for L in Ls:
        for beta in betas:
            deltas = []
            valid_j2 = []
            for j2r in j2_ratios:
                k1 = (j2r, L, beta, nf_min)
                kmax = (j2r, L, beta, nf_max)
                if k1 in idx and kmax in idx:
                    deltas.append(idx[k1]["f"] - idx[kmax]["f"])
                    valid_j2.append(j2r)
            if valid_j2:
                i = idx_color % len(ls_cycle)
                ax.plot(
                    valid_j2,
                    deltas,
                    color=color_cycle[idx_color],
                    linestyle=ls_cycle[i],
                    marker=marker_cycle[i],
                    label=f"$L={L}$, $\\beta={beta}$",
                )
            idx_color += 1

    ax.axvline(
        0.5,
        color="0.5",
        linewidth=0.8,
        linestyle="--",
        label="$J_2/J_1=0.5$ (critical)",
    )
    ax.set_xlabel(r"$J_2 / J_1$")
    ax.set_ylabel(rf"$\Delta F = F[\sigma_{{m=1}}] - F[\sigma_{{m={nf_max}}}]$")
    ax.set_title("Frustration detector")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out = out_dir / "fig5_frustration_detector.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary tables for the paper appendix
# ---------------------------------------------------------------------------


def print_summary_table(exact_data: list, nf_data: list):
    """Print LaTeX-ready summary tables for the paper appendix."""

    # --- Table 1: Free energy and T-score for exact validation cases ------
    print("\n% --- Table 1: Free energy and T-score, exact validation ---")
    print(r"\begin{tabular}{llccccccc}")
    print(r"\hline")
    print(
        r"Model & $L$ & $\beta$ & "
        r"$F^{\rm mixed}$ & $F^{\rm SC}$ & $F^{\rm var}$ & "
        r"$T_{\rm score}^{\rm SC}$ & $T_{\rm score}^{\rm var}$ & "
        r"Improvement \\ \hline"
    )

    idx = index_exact(exact_data)
    cases_table = [
        ("XX chain", [4, 8], [1.0, 2.0]),
        ("XXX Heisenberg AFM", [4, 8], [1.0, 2.0]),
        ("Ising transverse (Gamma=0.5J)", [4, 8], [1.0, 2.0]),
    ]
    for label, L_list, beta_list in cases_table:
        for L in L_list:
            for beta in beta_list:
                key = (label, L, beta)
                if key not in idx:
                    continue
                row = idx[key]
                fm = row["F_mixed"]
                fsc = row["F_sc"]
                fv = row["F_variational"]
                tsc = row.get("T_score_sc")
                tv = row.get("T_score_variational")
                imp = (fm - fv) / fm * 100 if fm > 0 else 0
                short = SHORT_LABELS.get(label, label.split("(")[0].strip())
                tsc_s = f"{tsc:.4f}" if tsc is not None else "--"
                tv_s = f"{tv:.4f}" if tv is not None else "--"
                print(
                    f"{short} & {L} & {beta:.1f} & "
                    f"{fm:.4f} & {fsc:.4f} & {fv:.4f} & "
                    f"{tsc_s} & {tv_s} & "
                    f"{imp:.1f}\\% \\\\"
                )
    print(r"\hline")
    print(r"\end{tabular}")

    # --- Table 2: F vs numfields for J1-J2 (L=12, beta=5) ----------------
    print("\n% --- Table 2: F vs numfields, J1-J2 (L=12, beta=5) ---")
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(r"$J_2/J_1$ & $F(m=1)$ & $F(m=4)$ & $F(m=10)$ & " r"$\Delta F$ \\ \hline")
    j2_ratios = sorted({r["J2_over_J1"] for r in nf_data})
    idx_nf_map = index_nf(nf_data)
    for j2r in j2_ratios:
        s1 = idx_nf_map.get((j2r, 12, 5.0, 1), {}).get("f", float("nan"))
        s4 = idx_nf_map.get((j2r, 12, 5.0, 4), {}).get("f", float("nan"))
        s10 = idx_nf_map.get((j2r, 12, 5.0, 10), {}).get("f", float("nan"))
        delta = s1 - s10 if not (np.isnan(s1) or np.isnan(s10)) else float("nan")
        print(f"{j2r:.2f} & {s1:.4f} & {s4:.4f} & {s10:.4f} & {delta:.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmark_results/variational_mf_paper_results.json"
    )

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run test_variational_mf_paper.py first to generate it.")
        sys.exit(1)

    out_dir = results_path.parent / "figures"
    out_dir.mkdir(exist_ok=True)

    data = load_results(results_path)
    exact_data = data.get("exact_validation", [])
    nf_data = data.get("numfields_convergence", [])

    print(f"Loaded {len(exact_data)} exact validation records")
    print(f"Loaded {len(nf_data)} numfields convergence records")

    if exact_data:
        # Fig 1: F + T-score vs beta (representative model)
        plot_figure1(exact_data, out_dir)
        # Fig 2: T-score SC vs variational, all models × L × beta
        plot_figure2(exact_data, out_dir)

    if nf_data:
        # Fig 3: F vs numfields + magnetization (J1-J2 chain)
        for L_plot in [12, 16]:
            plot_figure3(nf_data, out_dir, L_plot=L_plot, beta_plot=5.0)
        # Fig 4: F convergence vs numfields (absolute and relative)
        for L_plot in [12, 16]:
            plot_figure4(nf_data, out_dir, L_plot=L_plot, beta_plot=5.0)
        # Fig 5: Frustration detector
        plot_figure5(nf_data, out_dir)

    if exact_data or nf_data:
        print_summary_table(exact_data, nf_data)

    print(f"\nAll figures saved to {out_dir}/")
