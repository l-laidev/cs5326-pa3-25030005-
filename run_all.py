import json
from pathlib import Path

from src.generation import compute_majority_vote, run_greedy, run_self_consistency
from src.search import run_compute_matched_self_consistency, run_prm_best_of_n
from src.utils import load_models, save_aggregation_ablation, save_results_summary, save_sc_scaling_curve, set_seed_for_problem


if __name__ == "__main__":
    gsm8k_path = Path("data") / "gsm8k_test_50.jsonl"
    
    with open(gsm8k_path, "r") as f:
        gsm8k = [json.loads(line) for line in f.readlines()]
    
    gen_model, gen_tokenizer, judge_model, judge_tokenizer = load_models()
    
    samples_per_problem = []
    for idx, problem in enumerate(gsm8k):
        set_seed_for_problem(idx)
        
        greedy = run_greedy(gen_model, gen_tokenizer, problem["question"])
        self_consistency = run_self_consistency(gen_model, gen_tokenizer, problem["question"], n=5)
        prm = {}
        for prm_strategy in ["min", "product", "step_penalty"]:
            cur_prm = run_prm_best_of_n(gen_model, judge_model, gen_tokenizer, judge_tokenizer,
                              problem["question"], n=5, aggregation_strategy=prm_strategy,
                              candidates=self_consistency["candidates"])
            
            cur_prm["generation_calls"] += self_consistency["generation_calls"]
            cur_prm["generated_tokens"] += self_consistency["generated_tokens"]
            cur_prm["time_seconds"] += self_consistency["time_seconds"]
            
            prm[prm_strategy] = cur_prm
        
        matched_self_consistency = run_compute_matched_self_consistency(
            gen_model, gen_tokenizer, problem["question"], target_calls=15,
            existing_candidates=self_consistency["candidates"])
        
        matched_self_consistency["generation_calls"] += self_consistency["generation_calls"]
        matched_self_consistency["generated_tokens"] += self_consistency["generated_tokens"]
        matched_self_consistency["time_seconds"] += self_consistency["time_seconds"]
        
        samples_per_problem.append(
            {
                **problem,
                "greedy": greedy,
                "self_consistency": self_consistency,
                "prm": prm,
                "matched_self_consistency": matched_self_consistency,
            }
        )
    methods = ["greedy", "self_consistency", "prm_min", "prm_product", "prm_step_penalty", "matched_self_consistency"]
    results = []
    for method in methods:
        this_method = [
            {
                **(p[method] if "prm" not in method else p["prm"][method.replace("prm_", "")]),
                "ground_truth_answer": p["ground_truth_answer"],
            }
            for p in samples_per_problem
        ]

        result = {
            "Method": method,
            "Accuracy (%)": 100 * sum(p["answer"] == p["ground_truth_answer"] for p in this_method) / len(this_method),
            "Generation Calls": sum(p["generation_calls"] for p in this_method),
            "Judge Calls": sum(p["judge_calls"] for p in this_method if "judge_calls" in p),
            "Avg Gen Tokens / Problem": sum(p["generated_tokens"] for p in this_method) / len(this_method),
            "Avg Judge Tokens / Problem": sum(p["judge_tokens"] for p in this_method if "judge_tokens" in p) / len(this_method),
            "Total Time (s)": sum(p["time_seconds"] for p in this_method),
        }
        result["Total LLM Calls"] = result["Generation Calls"] + result["Judge Calls"]
        
        results.append(result)
    
    
    ablations = []
    for prm_method in filter(lambda x: "prm" in x, methods):
        this_strategy = prm_method.replace("prm_", "")
        this_method = [
            {
                **(p["prm"][this_strategy]),
                "question": p["question"],
                "ground_truth_answer": p["ground_truth_answer"],
                "self_consistency_candidates": p["self_consistency"]["candidates"]
            }
            for p in samples_per_problem
        ]
        if "step_penalty" in this_strategy:
            for lam in [0.0, 0.5, 2.0]:
                acc, tie = 0.0, 0.0
                for problem in this_method:
                    cur_prm = run_prm_best_of_n(gen_model, judge_model, gen_tokenizer, judge_tokenizer,
                                                problem["question"], n=5,
                                                aggregation_strategy=this_strategy,
                                                candidates=problem["self_consistency_candidates"],
                                                penalty_lambda=lam)
                    acc += int(cur_prm["answer"] == problem["ground_truth_answer"])
                    tie += int(cur_prm["selected"]["num_tied"] > 1)
                
                ablation = {
                    "Strategy": prm_method,
                    "Lambda": lam,
                    "Accuracy (%)": 100 * acc / len(this_method),
                    "Tie Rate (%)": 100 * tie / len(this_method),
                }
                ablations.append(ablation)
        else:
            ablation = {
                "Strategy": prm_method,
                "Lambda": None,
                "Accuracy (%)": 100 * sum(p["answer"] == p["ground_truth_answer"] for p in this_method) / len(this_method),
                "Tie Rate (%)": 100 * sum(p["selected"]["num_tied"] > 1 for p in this_method) / len(this_method),
            }
            ablations.append(ablation)
    
    curve = []
    for N in range(1, 15 +1):
        acc, gen_tokens = 0.0, 0
        for problem in samples_per_problem:
            upto_N = problem["matched_self_consistency"]["candidates"][:N]
            answer = compute_majority_vote([candidate["answer"] for candidate in upto_N])
            
            acc += int(answer == problem["ground_truth_answer"])
            gen_tokens += sum(candidate["num_tokens"] for candidate in upto_N)
        
        this_one = {
            "N": N,
            "Accuracy (%)": 100 * acc / len(samples_per_problem),
            "Avg Gen Tokens / Problem": gen_tokens / len(samples_per_problem),
        }
        curve.append(this_one)
    
    save_results_summary(results)
    save_aggregation_ablation(ablations)
    save_sc_scaling_curve(curve)
