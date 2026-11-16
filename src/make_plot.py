import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_PLOTS = Path("outputs") / "plots"
BASE_PLOTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


def acc_vs_calls_sc():
    with open("outputs/sc_scaling_curve.json", "r") as f:
        data = json.load(f)
    
    with open("outputs/results_summary.json") as f:
        prm_points = json.load(f)
    
    df = pd.DataFrame(data)
    df["N"] *= 50  # total calls for SC = N generations * total problems
    
    df = df.rename(columns={"Accuracy (%)": "Self-Consistency"})
    
    ax = df.plot(
        x="N",
        # y="Accuracy (%)",
        y="Self-Consistency",
        marker="o",
        linewidth=2.5,
        markersize=6,
        figsize=(8, 5),
    )
    
    for strat, color in zip(
        ["min", "product", "step_penalty"],
        ["red", "green", "blue"]
    ):
        ax.axhline(y=[p["Accuracy (%)"] for p in prm_points if f"prm_{strat}" == p["Method"]][0],
                   color=color,
                   linestyle="--", linewidth=2, label=f"PRM ({strat})")
    ax.legend()

    plt.title("Model Accuracy by Total LLM Calls", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Calls", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    sns.despine()
    
    plt.savefig(BASE_PLOTS / "scaling_acc_vs_calls.pdf")

def acc_vs_tokens():
    with open("outputs/sc_scaling_curve.json", "r") as f:
            data = json.load(f)
        
    df = pd.DataFrame(data)
    df = df.rename(columns={"Accuracy (%)": "Self-Consistency"})
    
    df.plot(
        x="Avg Gen Tokens / Problem",
        # y="Accuracy (%)",
        y="Self-Consistency",
        marker="o",
        linewidth=2.5,
        markersize=6,
        figsize=(8, 5),
    )

    plt.title("Model Accuracy by Tokens", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Total Tokens", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    sns.despine()
    
    plt.savefig(BASE_PLOTS / "scaling_acc_vs_tokens.pdf")


def judge_vs_human():
    with open("outputs/manual_audit.json", "r") as f:
        data =  json.load(f)
    df = pd.DataFrame(data)
    
    conf = pd.crosstab(df["judge_verdict"], df["human_verdict"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(BASE_PLOTS / "judge_vs_human.pdf")


if __name__ == "__main__":
    acc_vs_calls_sc()
    acc_vs_tokens()
    judge_vs_human()