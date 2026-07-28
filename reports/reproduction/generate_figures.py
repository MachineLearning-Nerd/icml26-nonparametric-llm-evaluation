"""Generate the four evidence figures used by the reproduction report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / ".openresearch/artifacts"
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f5a94"
ORANGE = "#e07a2f"
GREEN = "#2a8c6a"
RED = "#b94b5f"
GRAY = "#7b8794"


def load(path: str) -> dict:
    return json.loads((ART / path).read_text(encoding="utf-8"))


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def headline() -> None:
    data = load("claim6/raw/final_with_width_vectors.json")
    labels = ["Borda", "Bradley–Terry", "Rank Centrality"]
    keys = ["borda", "bt", "rank_centrality"]
    observed = [
        100 * data["gars"][key]["plugin_to_debiased_median_width_ratio"]
        for key in keys
    ]
    controls = [
        100 * data["gars"][key]["negative_control_disabled_eif_width_ratio"]
        for key in keys
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x - 0.19, observed, 0.38, color=BLUE, label="Observed plugin / EIF")
    ax.bar(x + 0.19, controls, 0.38, color=GRAY, label="EIF-disabled control")
    ax.axhline(50, color=RED, linestyle="--", linewidth=1.4, label="Contract limit")
    ax.set_yscale("log")
    ax.set_ylim(1, 145)
    ax.set_ylabel("Median interval-width ratio (%) · log scale")
    ax.set_xticks(x, labels)
    ax.set_title("Full Chatbot Arena scale: plugin intervals collapse for all three GARS")
    for index, value in enumerate(observed):
        ax.text(index - 0.19, value * 1.18, f"{value:.2f}%", ha="center", fontsize=9)
    for index, value in enumerate(controls):
        ax.text(index + 0.19, value * 1.06, "100%", ha="center", fontsize=9)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", alpha=0.18)
    finish(fig, "claim6_headline.png")


def theory_certificates() -> None:
    data = load("claim1/raw/exact_theory.json")
    checks = [
        ("BT recovery", data["claim_1"]["checks"]["bt_recovers_centered_strengths_max_abs"], 2e-9),
        ("BT independent", data["claim_1"]["checks"]["independent_bt_max_abs"], 2e-9),
        ("RC stationarity", data["claim_1"]["checks"]["rc_stationarity_max_abs"], 2e-9),
        ("EIF pathwise", data["claim_2"]["checks"]["max_pathwise_identity_abs"], 1e-12),
        ("Budget binding", data["claim_4"]["checks"]["budget_abs_error"], 1e-10),
        ("SLSQP agreement", data["claim_4"]["checks"]["independent_slsqp_max_abs"], 1e-6),
        ("KKT ratios", data["claim_4"]["checks"]["interior_kkt_ratio_range"], 1e-10),
    ]
    labels = [item[0] for item in checks]
    normalized = [max(item[1] / item[2], 1e-12) for item in checks]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    colors = [BLUE, BLUE, BLUE, ORANGE, GREEN, GREEN, GREEN]
    ax.bar(np.arange(len(labels)), normalized, color=colors)
    ax.axhline(1, color=RED, linestyle="--", label="Acceptance threshold")
    ax.set_yscale("log")
    ax.set_ylabel("Observed residual / tolerance · log scale")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_title("Independent algebraic certificates remain below their tolerances")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    finish(fig, "theory_certificates.png")


def table1_routes() -> None:
    p2 = load("claim3/routes/route1_p2_raw.json")
    p5 = load("claim3/routes/route2_p5_raw.json")
    paper_error = [p2["paper"]["plugin_error"]["mean"], p2["paper"]["debiased_error"]["mean"]]
    paper_cov = [p2["paper"]["plugin_coverage"]["mean"], p2["paper"]["debiased_coverage"]["mean"]]
    rows = [
        ("Paper", paper_error, paper_cov),
        (
            "Appendix p=2",
            [p2["observed"]["plugin_error"]["mean"], p2["observed"]["debiased_error"]["mean"]],
            [p2["observed"]["plugin_coverage"]["mean"], p2["observed"]["debiased_coverage"]["mean"]],
        ),
        (
            "Main text p=5",
            [p5["observed"]["plugin_error"]["mean"], p5["observed"]["debiased_error"]["mean"]],
            [p5["observed"]["plugin_coverage"]["mean"], p5["observed"]["debiased_coverage"]["mean"]],
        ),
    ]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, index, title, ylabel in [
        (axes[0], 1, "Borda vector error", "Mean error"),
        (axes[1], 2, "Joint 95% coverage", "Coverage"),
    ]:
        plugin = [row[index][0] for row in rows]
        debiased = [row[index][1] for row in rows]
        ax.bar(x - 0.19, plugin, 0.38, color=ORANGE, label="Plugin")
        ax.bar(x + 0.19, debiased, 0.38, color=BLUE, label="Debiased")
        ax.set_xticks(x, [row[0] for row in rows], rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    axes[1].axhline(0.95, color=GREEN, linestyle="--", linewidth=1.4, label="Nominal 0.95")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Table 1: coverage pattern returns, reported error scale does not")
    finish(fig, "claim3_table1_routes.png")


def table2_routes() -> None:
    routes = [load(f"claim5/routes/route{route}_raw.json") for route in range(1, 4)]
    keys = ["borda", "bt", "rank_centrality"]
    labels = ["Borda", "Bradley–Terry", "Rank Centrality"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    widths = 0.22
    for offset, route in enumerate(routes):
        ratios = [
            route["observed_mse_x100"][key]["aopt"]["mean"]
            / route["observed_mse_x100"][key]["random"]["mean"]
            for key in keys
        ]
        ax.bar(
            x + (offset - 1) * widths,
            ratios,
            widths,
            label=f"Route {offset + 1}",
            color=[BLUE, ORANGE, GREEN][offset],
        )
    ax.axhline(1, color=RED, linestyle="--", label="Equal MSE")
    ax.set_xticks(x, labels)
    ax.set_ylabel("A-optimal MSE / random MSE")
    ax.set_title("Table 2 clean-room routes: direction depends on nuisance interpretation")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    finish(fig, "claim5_table2_routes.png")


if __name__ == "__main__":
    headline()
    theory_certificates()
    table1_routes()
    table2_routes()
