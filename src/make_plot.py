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
    
    df = pd.DataFrame(data)
    
    df.plot(
        x="N",
        y="Accuracy (%)",
        marker="o",
        linewidth=2.5,
        markersize=6,
        figsize=(8, 5),
    )

    plt.title("Model Accuracy by N", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("N", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    sns.despine()
    
    plt.savefig(BASE_PLOTS / "scaling_acc_vs_calls.pdf")

def acc_vs_tokens():
    with open("outputs/sc_scaling_curve.json", "r") as f:
            data = json.load(f)
        
    df = pd.DataFrame(data)
    
    df.plot(
        x="Avg Gen Tokens / Problem",
        y="Accuracy (%)",
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