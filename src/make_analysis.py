import json
import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_resampled_problems(n_resamples=1_000, seed=5326):
    with open("outputs/per_problem.jsonl", "r") as f:
        lines = [json.loads(line) for line in f]
    
    rng = random.Random(seed)
    all_samples = []
    for _ in range(n_resamples):
        sampled = rng.choices(lines, k=len(lines))
        all_samples.append(sampled)
    
    return lines, all_samples

def paired_win_loss_counts(lines, use_matched=False):
    counts = {"prm": 0, "other": 0, "both": 0}
    for line in lines:
        GT = line["ground_truth_answer"]
        PRM = line["prm"]["min"]
        SC = line["matched_self_consistency"] if use_matched else line["self_consistency"]
        
        PRM = PRM["answer"]
        SC = SC["answer"]
        
        if PRM == SC:
            counts["both"] += 1
        
        if PRM == GT:
            if SC == GT:
                pass  # counted under both
            else:
                counts["prm"] += 1
        else:
            if SC == GT:
                counts["other"] += 1
            else:
                pass  # both did not select the right one

    return counts

def prm_vs_sc():
    lines, all_samples = load_resampled_problems()
    
    orig_sc_counts = paired_win_loss_counts(lines, use_matched=False)
    orig_msc_counts = paired_win_loss_counts(lines, use_matched=True)

    print("=== Paired Win / Loss Counts ===")
    print(f"PRM-BoN (min) vs. Self-Consistency (N=5):")
    print(
        f"  - PRM won: {orig_sc_counts['prm']}, SC won: {orig_sc_counts['other']}, Both agreed: {orig_sc_counts['both']}"
    )

    print(f"\nPRM-BoN (min) vs. Compute-Matched SC (N=15):")
    print(
        f"  - PRM won: {orig_msc_counts['prm']}, SC won: {orig_msc_counts['other']}, Both agreed: {orig_msc_counts['both']}"
    )

    def compute_accuracies(sample):
        accs = {
            "prm-min": 0.0,
            "prm-product": 0.0,
            "prm-step_penalty": 0.0,
            "self_consistency": 0.0,
            "matched_self_consistency": 0.0,
            "greedy": 0.0,
        }
        n = len(sample)
        for strat in accs:
            if "prm" in strat:
                prm_type = strat.split("-")[-1]
                accs[strat] = 100 * sum(1 for line in sample if line["prm"][prm_type]["answer"] == line["ground_truth_answer"]) / n
            else:
                accs[strat] = 100 * sum(1 for line in sample if line[strat]["answer"] == line["ground_truth_answer"]) / n
        
        return accs

    orig_accs = compute_accuracies(lines)

    accs = {k: [] for k in orig_accs}
    for sample in all_samples:
        acc = compute_accuracies(sample)
        for k in acc:
            accs[k].append(acc[k])


    def get_ci(accs):
        return np.percentile(accs, [2.5, 97.5])


    print("\n=== 95% CI (Accuracy) ===")
    for strat in accs:
        strat_ci = get_ci(accs[strat])
        print(
            f"{strat}: {orig_accs[strat]:.2f}% ([{strat_ci[0]:.2f}%, {strat_ci[1]:.2f}%])"
        )

def judge_failure_cases():
    lines, _ = load_resampled_problems()
    for line in lines:
        pid = line["id"]
        q = line["question"]
        
        candidates = line["prm"]["min"]["candidates"]
        shared_flaw = all([len(c["steps"]) > 0 and c["step_scores"][0] == -1.0 for c in candidates])
        show_this = random.random() < 0.5
        
        if shared_flaw and show_this:
            print(pid, q)
        
        for candidate in candidates:
            steps = candidate["steps"]
            scores = candidate["step_scores"]
            rationales = candidate["rationales"]
            
            if shared_flaw and show_this:
                print("First Step for candidate", candidates.index(candidate)+1)
                print("Ground Truth: ", line["ground_truth_answer"], "\nParsed: ", candidate["answer"])
                print("Verdict: ", "INVALID" if scores[0]==-1.0 else "VALID")
                print("Step: ", steps[0])
                print("Rationale: ", rationales[0])
                print()
            
            # randomly show this one
            # if all([s == +1 for s in scores]) and random.random() < 0.01:
            # if all([s == -1 for s in scores]) and random.random() < 0.1:
            # if any([s == -1 for s in scores]) and random.random() < 0.01:
            # if shared_flaw:
            #     print("Ground Truth: ", line["ground_truth_answer"], "\nParsed: ", candidate["answer"])
            #     print(pid, q)
            #     for i in range(len(steps)):
            #         print(f'==== Step {i+1}: Got {scores[i]}')
            #         print(f'\n\t {steps[i]}')
            #         print(f'\n\t {rationales[i]}')
                
                # show only one
                # raise StopIteration
        if shared_flaw and show_this:
            raise StopIteration

def get_steps_for_manual_audit():
    TOTAL = 30
    MIN_INVALID = 10
    MAX_VALID = TOTAL - MIN_INVALID
    
    lines, _ = load_resampled_problems()
    chosen = []
    n_invalid = 0
    
    rng = random.Random(5326)
    rng.shuffle(lines)
    for line in lines:
        q = line["question"]
        ground_truth = line["ground_truth_answer"]
        
        candidates = line["prm"]["min"]["candidates"]
        for candidate in rng.choices(candidates, k=3):
            og_steps = candidate["steps"]
            steps = [str(s) for s in og_steps]
            scores = candidate["step_scores"]
            
            rng.shuffle(steps)
            steps = list(filter(lambda x: len(re.findall(r"[0-9]", x)) != 0, steps))  # pick steps with numbers
            for i in range(min(len(steps), 1)):
                if scores[i] == 1.0 and len(chosen) - n_invalid >= MAX_VALID: # remaining chosen steps should be only INVALID steps
                    continue
                    
                if scores[i] == -1.0:
                    n_invalid += 1
                
                preceding_steps = og_steps[:og_steps.index(steps[i])]
                chosen.append({
                    "problem": q.split("\n"),
                    "ground_truth": ground_truth,
                    "preceding_steps": [s.split("\n") for s in preceding_steps],
                    "step": steps[i].split("\n"),
                    "judge_verdict": "INVALID" if scores[i] == -1.0 else "VALID",
                    "human_verdict": None
                })
                
                if len(chosen) >= TOTAL and n_invalid >= MIN_INVALID:
                    return chosen
    return chosen

def judge_precision_recall():
    with open("outputs/manual_audit.json", "r") as f:
            data =  json.load(f)
    df = pd.DataFrame(data)
    
    print(pd.crosstab(df["judge_verdict"], df["human_verdict"]))
    
    # INVALID is positive class
    df["judge_verdict"] = df["judge_verdict"].map({"VALID": 0, "INVALID": 1})
    df["human_verdict"] = df["human_verdict"].map({"VALID": 0, "INVALID": 1})
    
    conf = pd.crosstab(df["judge_verdict"], df["human_verdict"])
    
    conf = conf.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    TP = conf.loc[1, 1]
    FP = conf.loc[0, 1]
    FN = conf.loc[1, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f'Precision: {precision:.2f}')
    print(f'Recall:    {recall:.2f}')

if __name__ == "__main__":
    prm_vs_sc()
    print("\n")
    
    # judge_failure_cases()
    # print("\n")
    
    if not os.path.exists("outputs/manual_audit.json"):
        steps = get_steps_for_manual_audit()
        
        assert len(list(filter(lambda x: x["judge_verdict"] == "VALID", steps))) >= 30-10
        assert len(list(filter(lambda x: x["judge_verdict"] == "INVALID", steps))) >= 10
        
        with open("outputs/manual_audit.json", "w") as f:
            json.dump(steps, f, indent=4)

    judge_precision_recall()
    print("\n")