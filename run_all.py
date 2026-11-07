import json
from pathlib import Path

import tqdm

from src.generation import compute_majority_vote, run_greedy, run_self_consistency
from src.search import run_compute_matched_self_consistency, run_prm_best_of_n
from src.utils import load_models, save_aggregation_ablation, save_results_summary, save_sc_scaling_curve, set_seed_for_problem


def append_jsonl(path, record):
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(path, "a") as f:
        f.write(json.dumps(record))

def read_jsonl(path):
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]

if __name__ == "__main__":
    gsm8k_path = Path("data") / "gsm8k_test_50.jsonl"
    per_prob_path = Path("outputs") / "per_problem.jsonl"
    
    with open(gsm8k_path, "r") as f:
        gsm8k = [json.loads(line) for line in f.readlines()]
    
    gen_model, gen_tokenizer, judge_model, judge_tokenizer = load_models()
    print(f"Generator on {gen_model.device} and Judge on {judge_model.device}")
    
    samples_per_problem = []
    already_done = set()
    
    if per_prob_path.exists():
        samples_per_problem = read_jsonl(per_prob_path)
        already_done = set(record["idx"] for record in samples_per_problem)
    
    for idx, problem in (pbar := tqdm.tqdm(enumerate(gsm8k), total=len(gsm8k))):
        pbar.set_description_str(f"Processing Problem #{idx+1}")
        if idx in already_done:
            continue
        
        set_seed_for_problem(idx)
        
        pbar.set_description_str(f"Processing Problem #{idx+1} [Greedy]")
        greedy = run_greedy(gen_model, gen_tokenizer, problem["question"])
        pbar.set_description_str(f"Processing Problem #{idx+1} [SC@5]")
        self_consistency = run_self_consistency(gen_model, gen_tokenizer, problem["question"], n=5)
        prm = {}
        for prm_strategy in ["min", "product", "step_penalty"]:
            pbar.set_description_str(f"Processing Problem #{idx+1} [PRM, {prm_strategy}]")
            cur_prm = run_prm_best_of_n(gen_model, judge_model, gen_tokenizer, judge_tokenizer,
                              problem["question"], n=5, aggregation_strategy=prm_strategy,
                              candidates=self_consistency["candidates"])
            
            cur_prm["generation_calls"] += self_consistency["generation_calls"]
            cur_prm["generated_tokens"] += self_consistency["generated_tokens"]
            cur_prm["time_seconds"] += self_consistency["time_seconds"]
            
            prm[prm_strategy] = cur_prm
        
        pbar.set_description_str(f"Processing Problem #{idx+1} [Matched SC@15]")
        matched_self_consistency = run_compute_matched_self_consistency(
            gen_model, gen_tokenizer, problem["question"], target_calls=15,
            existing_candidates=self_consistency["candidates"])
        
        matched_self_consistency["generation_calls"] += self_consistency["generation_calls"]
        matched_self_consistency["generated_tokens"] += self_consistency["generated_tokens"]
        matched_self_consistency["time_seconds"] += self_consistency["time_seconds"]
        
        record = {
            **problem,
            "idx": idx,
            "greedy": greedy,
            "self_consistency": self_consistency,
            "prm": prm,
            "matched_self_consistency": matched_self_consistency,
        }
        
        append_jsonl(per_prob_path, record)
        samples_per_problem.append(record)
    
    compliance_rate = sum(("final answer:" in p["greedy"]["completion"].lower()) and (p["greedy"]["answer"] is not None)
                       for p in samples_per_problem) / len(samples_per_problem)
    print(f"Greedy Delimiter-Compliance Rate: {compliance_rate}")
    
    methods = ["greedy", "self_consistency", "prm_min", "prm_product", "prm_step_penalty", "matched_self_consistency"]
    results = []
    for method in (pbar := tqdm.tqdm(methods)):
        pbar.set_description_str(f"Processing: {method}")
        
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
    for prm_method in (pbar := tqdm.tqdm(filter(lambda x: "prm" in x, methods))):
        this_strategy = prm_method.replace("prm_", "")
        pbar.set_description_str(f"Processing: {this_strategy}")
        
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
                for problem in tqdm.tqdm(this_method, desc=f"Lambda={lam}", total=len(this_method), leave=False):
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
    for N in tqdm.trange(1, 15 +1, desc="Self-Consistency Curve"):
        acc, gen_tokens = 0.0, 0
        for problem in tqdm.tqdm(samples_per_problem, desc=f"N={N}", total=len(samples_per_problem), leave=False):
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
