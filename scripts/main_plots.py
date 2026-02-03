from __future__ import annotations

import argparse

import pandas as pd

from utils import *

set_seaborn_paper_context()

MODELS = {
    "LinearDistScore": "Lin–Score",
    "PoissonDistScore": "Pois–Score",
    "MechProbLinearDist": "Lin–Mech",
    "MechProbPoissonDist": "Pois–Mech",
    "LogitProbLinearDist_0.1": "Lin–Logit(0.1)",
    "LogitProbLinearDist_1.0": "Lin–Logit(1)",
    "LogitProbPoissonDist_0.1": "Pois–Logit(0.1)",
    "LogitProbPoissonDist_1.0": "Pois–Logit(1)",
}

SCENARIOS = {
    "baseline": "Baseline",
    "surveillance_moderate": "Surveillance (moderate)",
    "surveillance_severe": "Surveillance (severe)",

    # Low evolutionary signal scenarios
    "low_clock_signal": "Low clock signal",
    "low_k_inc": "Low incubation shape",
    "low_scale_inc": "Low incubation scale",
    # High evolutionary signal scenarios
    "high_clock_signal": "High clock signal",
    "high_k_inc": "High incubation shape",
    "high_scale_inc": "High incubation scale",

    "relaxed_clock": "Relaxed clock",
    "adversarial": "Adversarial",
}

def plot_pr_auc_grid(df, out_path, title, col_wrap=3, scenarios=None):
    d = df.loc[:, ["Scenario", "Model", "PR_AUC"]].copy()

    d["DistType"] = d["Model"].apply(
        lambda m: "Stochastic" if "Poisson" in m else "Deterministic"
    )

    d["ModelLabel"] = d["Model"].map(MODELS)
    model_order = list(MODELS.values())

    d["ScenarioLabel"] = d["Scenario"].map(SCENARIOS)
    scenario_order = list(SCENARIOS.values())

    if scenarios is not None:
        scenario_order = [SCENARIOS[s] for s in scenarios]
        d = d[d["Scenario"].isin(scenarios)]

    g = sns.catplot(
        data=d,
        kind="bar",
        x="ModelLabel",
        y="PR_AUC",
        hue="DistType",
        order=model_order,
        col="ScenarioLabel",
        col_order=scenario_order,
        col_wrap=col_wrap,
        height=2.5,
        aspect=1.15,
        sharey=True,
        dodge=False,
    )

    g.set_axis_labels("", "PR-AUC")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1.05))
    g.legend.set_title("Divergence")

    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=90)

    g.figure.suptitle(title, y=1.03)

    g.figure.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    g.figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(g.figure)

def plot_heatmaps_best_f1_and_opt_gamma(df, out_path, title, scenarios=None):
    d = df.loc[:, ["Scenario", "Weight_Column", "gamma", "BCubed_F1_Score"]].copy()
    d["ModelLabel"] = d["Weight_Column"].map(MODELS)
    d["ScenarioLabel"] = d["Scenario"].map(SCENARIOS)

    model_order = [MODELS[m] for m in d["Weight_Column"].unique()]
    scenario_order = list(SCENARIOS.values())

    if scenarios is not None:
        scenario_order = [SCENARIOS[s] for s in scenarios]
        d = d[d["Scenario"].isin(scenarios)]

    # ---------- Best F1 ----------
    best = (
        d.groupby(["ScenarioLabel", "ModelLabel"], as_index=False)["BCubed_F1_Score"]
         .max()
         .rename(columns={"BCubed_F1_Score": "Best_F1"})
    )

    best = best.sort_values(by="Best_F1", ascending=False)

    mat_f1 = (
        best.pivot(index="ScenarioLabel", columns="ModelLabel", values="Best_F1")
            .reindex(index=scenario_order, columns=model_order)
    )

    # ---------- Optimal gamma ----------
    idx = d.groupby(["ScenarioLabel", "ModelLabel"])["BCubed_F1_Score"].idxmax()
    opt = (
        d.loc[idx, ["ScenarioLabel", "ModelLabel", "gamma"]]
         .rename(columns={"gamma": "Opt_gamma"})
    )

    mat_gamma = (
        opt.pivot(index="ScenarioLabel", columns="ModelLabel", values="Opt_gamma")
           .reindex(index=scenario_order, columns=model_order)
    )

    # ---------- Plot ----------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(10, 5),
        sharey=True,
        sharex=True,
    )

    sns.heatmap(
        mat_f1,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "F1-score"},
        ax=axes[0]
    )
    axes[0].set_title("Best F1-score")
    axes[0].set_ylabel("Scenario")

    sns.heatmap(
        mat_gamma,
        annot=True,
        fmt=".2g",
        cbar_kws={"label": r"Resolution ($\gamma$)"},
        ax=axes[1]
    )
    axes[1].set_title(r"Optimal Resolution ($\gamma$)")
    axes[1].set_ylabel("")

    for ax in axes.flatten():
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlabel("")

    fig.suptitle(title)

    plt.tight_layout()

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_gamma_curves_grid(df, out_path, metric="BCubed_F1_Score", col_wrap=3):
    d = df.loc[:, ["Scenario", "Weight_Column", "gamma", metric]].copy()

    d["ModelLabel"] = d["Weight_Column"].map(MODELS)
    model_order = [MODELS[m] for m in d["Weight_Column"].unique()]

    d["ScenarioLabel"] = d["Scenario"].map(SCENARIOS)
    scenario_order = list(SCENARIOS.values())

    g = sns.relplot(
        data=d,
        x="gamma",
        y=metric,
        hue="ModelLabel",
        hue_order=model_order,
        col="ScenarioLabel",
        col_order=scenario_order,
        col_wrap=col_wrap,
        kind="line",
        marker="o",
        height=3,
        aspect=1.15
    )

    g.set_titles("{col_name}")
    g.set_axis_labels(r"Resolution ($\gamma$)", "-".join(metric.split("_")[1:]))
    g.set(ylim=(0, 1.05))

    g.figure.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    g.figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.figure)

def plot_gamma_stability_curves_grid(df, out_path, metric="BCubed_F1_Score", col_wrap=3):
    d = df.loc[:, ["Scenario", "Weight_Column", "gamma1", metric]].copy()

    d["ModelLabel"] = d["Weight_Column"].map(MODELS).fillna(d["Weight_Column"])
    model_order = [MODELS[m] for m in d["Weight_Column"].unique()]

    d["ScenarioLabel"] = d["Scenario"].map(SCENARIOS)
    scenario_order = list(SCENARIOS.values())

    g = sns.relplot(
        data=d,
        x="gamma1",
        y=metric,
        hue="ModelLabel",
        hue_order=model_order,
        col="ScenarioLabel",
        col_order=scenario_order,
        col_wrap=col_wrap,
        kind="line",
        marker="o",
        height=3,
        aspect=1.15
    )

    g.set_titles("{col_name}")
    g.set_axis_labels(r"Resolution ($\gamma$)", "-".join(metric.split("_")[1:]))
    g.set(ylim=(0, 1.05))

    g.figure.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    g.figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.figure)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "main"], "../figures/main"))
    sup_figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    sup_figs_dir = sup_figs_dir / "clustering"
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    ensure_dirs(figs_dir, sup_figs_dir, tabs_dir)

    fig1_data = pd.read_csv(tabs_dir / "edge_eval_metrics.csv")
    fig2a_data = pd.read_csv(tabs_dir / "clustering_metrics.csv")
    fig2b_data = pd.read_csv(tabs_dir / "clustering_stability.csv")

    scenarios = [
        "baseline",
        "surveillance_moderate",
        "low_clock_signal",
        "high_clock_signal",
        "relaxed_clock",
        "adversarial"
    ]

    plot_pr_auc_grid(
        df=fig1_data,
        out_path=figs_dir / "fig1_pr_auc_grid",
        title="",
        scenarios=scenarios
    )

    plot_pr_auc_grid(
        df=fig1_data,
        out_path=sup_figs_dir / "fig1_pr_auc_grid",
        title="PR-AUC across scenarios and models"
    )

    plot_heatmaps_best_f1_and_opt_gamma(
        df=fig2a_data,
        out_path=figs_dir / "fig2_heatmap_best_f1",
        title="",
        scenarios=scenarios
    )

    plot_heatmaps_best_f1_and_opt_gamma(
        df=fig2a_data,
        out_path=sup_figs_dir / "fig2_heatmap_best_f1",
        title=r"Best F1 score and $\gamma$ across scenarios and models"
    )

    plot_gamma_curves_grid(
        df=fig2a_data,
        out_path=sup_figs_dir / "gamma_curves_grid_F1_Score",
        metric="BCubed_F1_Score",
        col_wrap=3
    )

    plot_gamma_curves_grid(
        df=fig2a_data,
        out_path=sup_figs_dir / "gamma_curves_grid_Precision",
        metric="BCubed_Precision",
        col_wrap=3
    )

    plot_gamma_curves_grid(
        df=fig2a_data,
        out_path=sup_figs_dir / "gamma_curves_grid_Recall",
        metric="BCubed_Recall",
        col_wrap=3
    )

    plot_gamma_stability_curves_grid(
        df=fig2b_data,
        out_path=sup_figs_dir / "gamma_stability_curves_grid_F1_Score",
        col_wrap=3
    )


if __name__ == "__main__":
    main()