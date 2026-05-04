"""Figures for the QALMA variational MF paper.

Reads benchmark_results/variational_mf_paper_results.json and produces:

  Figure 1 — F vs beta for all models (Family 1 validation)
             Panel a: F(mixed), F(SC), F(var) vs beta at fixed L
             Panel b: F(var)/F(mixed) vs L at fixed beta (quality ratio)

  Figure 2 — F vs numfields for the J1-J2 chain (Family 2)
             Panel a: curves for each J2/J1 at fixed L and beta
             Panel b: magnetization pattern <Sz_i> for selected numfields

  Figure 3 — Phase diagram proxy: F(nf=1) - F(nf=10) vs J2/J1
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

# J2/J1 color gradient for Figure 2
J2_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 7))


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
    """Index numfields_convergence results as {(J2_ratio, L, beta, nf):
    row}.
    """
    idx = {}
    for row in data:
        key = (row["J2_over_J1"], row["L"], row["beta"], row["numfields"])
        idx[key] = row
    return idx


# ---------------------------------------------------------------------------
# Figure 1 — Validation against exact diagonalization
# ---------------------------------------------------------------------------


def plot_figure1(exact_data: list, out_dir: Path):
    """Two panels:
    (a) F vs beta for a representative model (XXX AFM) at L=8
    (b) F(var) / F(mixed) vs L at beta=2 for all models.
    """
    idx = index_exact(exact_data)

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.42))

    # ---- Panel (a): F vs beta, L=8, Ising transverse -----------------
    ax = axes[0]
    model_label = "Ising transverse (Gamma=0.5J)"
    L_fixed = 8
    betas = sorted({row["beta"] for row in exact_data if row["label"] == model_label})

    s_mixed_vals, s_sc_vals, s_var_vals = [], [], []
    valid_betas = []
    for beta in betas:
        key = (model_label, L_fixed, beta)
        if key not in idx:
            continue
        row = idx[key]
        s_mixed_vals.append(row["F_mixed"])
        s_sc_vals.append(row["F_sc"])
        s_var_vals.append(row["F_variational"])
        valid_betas.append(beta)

    if valid_betas:
        ax.plot(
            valid_betas,
            s_mixed_vals,
            color=COLORS["mixed"],
            marker=MARKERS["mixed"],
            label="Mixed state",
            linestyle="--",
        )
        ax.plot(
            valid_betas,
            s_sc_vals,
            color=COLORS["sc"],
            marker=MARKERS["sc"],
            label="Self-consistent MF",
        )
        ax.plot(
            valid_betas,
            s_var_vals,
            color=COLORS["var"],
            marker=MARKERS["var"],
            label="Variational MF",
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$S_{\rm rel}(\sigma \| e^{-\beta H})$")
    ax.set_title(f"Ising transverse ($\\Gamma=0.5J$), $L={L_fixed}$")
    ax.legend()
    ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    # ---- Panel (b): quality ratio vs L, beta=5 ----------------------------
    ax = axes[1]
    beta_fixed = 5.0
    models_to_show = [
        "XX chain",
        "XXX Heisenberg AFM",
        "XXX Heisenberg FM",
        "Ising transverse (Gamma=0.5J)",
        "XYZ anisotropic (Jz=1, Jxy=0.5)",
    ]
    ls_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    marker_cycle = ["o", "s", "^", "D", "v"]

    for i, label in enumerate(models_to_show):
        Ls, ratios = [], []
        for row in exact_data:
            if row["label"] != label or row["beta"] != beta_fixed:
                continue
            if row["F_mixed"] == 0:
                continue
            Ls.append(row["L"])
            ratios.append(row["F_variational"] / row["F_mixed"])
        if Ls:
            # sort by L
            pairs = sorted(zip(Ls, ratios))
            Ls, ratios = zip(*pairs)
            short = label.split("(")[0].strip()
            ax.plot(
                Ls,
                ratios,
                linestyle=ls_cycle[i % len(ls_cycle)],
                marker=marker_cycle[i % len(marker_cycle)],
                label=short,
            )

    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$S_{\rm rel}^{\rm var} / S_{\rm rel}^{\rm mixed}$")
    ax.set_title(r"Quality ratio, $\beta=5$")
    ax.legend(fontsize=7)
    ax.text(-0.18, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "fig1_validation.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_figure1_Tscore(exact_data: list, out_dir: Path):
    """Two panels:
    (a) F vs beta for a representative model (XXX AFM) at L=8
    (b) F(var) / F(mixed) vs L at beta=2 for all models.
    """
    idx = index_exact(exact_data)

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.42))

    # ---- Panel (a): F vs beta, L=8, Ising transverse -----------------
    ax = axes[0]
    model_label = "Ising transverse (Gamma=0.5J)"
    L_fixed = 8
    betas = sorted({row["beta"] for row in exact_data if row["label"] == model_label})

    s_mixed_vals, s_sc_vals, s_var_vals = [], [], []
    valid_betas = []
    for beta in betas:
        key = (model_label, L_fixed, beta)
        if key not in idx:
            continue
        row = idx[key]
        s_mixed_vals.append(row["T_score_mixed"])
        s_sc_vals.append(row["T_score_sc"])
        s_var_vals.append(row["T_score_variational"])
        valid_betas.append(beta)

    if valid_betas:
        ax.plot(
            valid_betas,
            s_mixed_vals,
            color=COLORS["mixed"],
            marker=MARKERS["mixed"],
            label="Mixed state",
            linestyle="--",
        )
        ax.plot(
            valid_betas,
            s_sc_vals,
            color=COLORS["sc"],
            marker=MARKERS["sc"],
            label="Self-consistent MF",
        )
        ax.plot(
            valid_betas,
            s_var_vals,
            color=COLORS["var"],
            marker=MARKERS["var"],
            label="Variational MF",
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$T_{\rm score}(\sigma \| e^{-\beta H})$")
    ax.set_title(f"Ising transverse ($\\Gamma=0.5J$), $L={L_fixed}$")
    ax.legend()
    ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    # ---- Panel (b): quality ratio vs L, beta=5 ----------------------------
    ax = axes[1]
    beta_fixed = 5.0
    models_to_show = [
        "XX chain",
        "XXX Heisenberg AFM",
        "XXX Heisenberg FM",
        "Ising transverse (Gamma=0.5J)",
        "XYZ anisotropic (Jz=1, Jxy=0.5)",
    ]
    ls_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    marker_cycle = ["o", "s", "^", "D", "v"]

    for i, label in enumerate(models_to_show):
        Ls, ratios = [], []
        for row in exact_data:
            if row["label"] != label or row["beta"] != beta_fixed:
                continue
            if row["F_mixed"] == 0:
                continue
            Ls.append(row["L"])
            # ratios.append(row["T_score_variational"] / row["T_score_mixed"])


    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$S_{\rm rel}^{\rm var} / S_{\rm rel}^{\rm mixed}$")
    ax.set_title(r"Quality ratio, $\beta=5$")
    ax.legend(fontsize=7)
    ax.text(-0.18, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "fig1_tscore_validation.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 — F vs numfields (J1-J2 chain)
# ---------------------------------------------------------------------------


def plot_figure2(
    nf_data: list, out_dir: Path, L_plot: int = 12, beta_plot: float = 5.0
):
    """Two panels:
    (a) F vs numfields for each J2/J1 at fixed L and beta
    (b) Magnetization pattern <Sz_i> for max frustration (J2/J1=0.5)
        at nf=1 vs nf=10.
    """
    idx = index_nf(nf_data)

    j2_ratios = sorted({r["J2_over_J1"] for r in nf_data})
    nf_vals = sorted({r["numfields"] for r in nf_data})

    fig = plt.figure(figsize=(COL2, COL2 * 0.44))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ---- Panel (a): F vs numfields ------------------------------------
    for i, j2r in enumerate(j2_ratios):
        nfs, srels = [], []
        for nf in nf_vals:
            key = (j2r, L_plot, beta_plot, nf)
            if key in idx:
                nfs.append(nf)
                srels.append(idx[key]["f"])
        if nfs:
            ax_left.plot(
                nfs,
                srels,
                color=J2_COLORS[i % len(J2_COLORS)],
                marker="o",
                label=f"$J_2/J_1={j2r:.1f}$",
            )

    ax_left.set_xlabel("Number of fields $m$")
    ax_left.set_ylabel(r"$S_{\rm rel}(\sigma_m \| e^{-\beta H})$")
    ax_left.set_title(f"J1-J2 chain, $L={L_plot}$, $\\beta={beta_plot}$")
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
    out = out_dir / "fig2_numfields_convergence.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Frustration detector: Delta F vs J2/J1
# ---------------------------------------------------------------------------


def plot_figure3(nf_data: list, out_dir: Path):
    """Delta F = F(nf=1) - F(nf=nf_max) as a function of J2/J1.
    Large Delta means many fields are needed → frustrated region.
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
    ax.set_ylabel(
        rf"$\Delta S_{{\rm rel}} = S_{{\rm rel}}^{{m=1}} - S_{{\rm rel}}^{{m={nf_max}}}$"
    )
    ax.set_title("Frustration detector")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out = out_dir / "fig3_frustration_detector.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bonus: summary table printed to stdout
# ---------------------------------------------------------------------------


def print_summary_table(exact_data: list, nf_data: list):
    """Print a LaTeX-ready summary table for the paper appendix."""
    print("\n% --- Table: F for exact validation cases ---")
    print(r"\begin{tabular}{llccccc}")
    print(r"\hline")
    print(
        r"Model & $L$ & $\beta$ & "
        r"$S_{\rm rel}^{\rm mixed}$ & "
        r"$S_{\rm rel}^{\rm SC}$ & "
        r"$S_{\rm rel}^{\rm var}$ & "
        r"Improvement \\ \hline"
    )

    idx = index_exact(exact_data)
    for label, parms, L_list, beta_list in [
        ("XX chain", {}, [4, 8], [1.0, 2.0]),
        ("XXX Heisenberg AFM", {}, [4, 8], [1.0, 2.0]),
        ("Ising transverse (Gamma=0.5J)", {}, [4, 8], [1.0, 2.0]),
    ]:
        for L in L_list:
            for beta in beta_list:
                key = (label, L, beta)
                if key not in idx:
                    continue
                row = idx[key]
                sm = row["F_mixed"]
                ssc = row["F_sc"]
                sv = row["F_variational"]
                imp = (sm - sv) / sm * 100 if sm > 0 else 0
                short = label.split("(")[0].strip()
                print(
                    f"{short} & {L} & {beta:.1f} & "
                    f"{sm:.4f} & {ssc:.4f} & {sv:.4f} & "
                    f"{imp:.1f}\\% \\\\"
                )
    print(r"\hline")
    print(r"\end{tabular}")

    print("\n% --- Table: F vs numfields for J1-J2 (L=12, beta=5) ---")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"$J_2/J_1$ & $m=1$ & $m=4$ & $m=10$ & " r"$\Delta S_{\rm rel}$ \\ \hline")
    j2_ratios = sorted({r["J2_over_J1"] for r in nf_data})
    idx_nf = index_nf(nf_data)
    for j2r in j2_ratios:
        s1 = idx_nf.get((j2r, 12, 5.0, 1), {}).get("f", float("nan"))
        s4 = idx_nf.get((j2r, 12, 5.0, 4), {}).get("f", float("nan"))
        s10 = idx_nf.get((j2r, 12, 5.0, 10), {}).get("f", float("nan"))
        delta = s1 - s10 if not (np.isnan(s1) or np.isnan(s10)) else float("nan")
        print(f"{j2r:.2f} & {s1:.4f} & {s4:.4f} & {s10:.4f} & " f"{delta:.4f} \\\\")
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
        plot_figure1(exact_data, out_dir)
        plot_figure1_Tscore(exact_data, out_dir)

    if nf_data:
        plot_figure2(nf_data, out_dir, L_plot=12, beta_plot=5.0)
        plot_figure2(nf_data, out_dir, L_plot=16, beta_plot=5.0)
        plot_figure3(nf_data, out_dir)

    if exact_data or nf_data:
        print_summary_table(exact_data, nf_data)

    print(f"\nAll figures saved to {out_dir}/")
