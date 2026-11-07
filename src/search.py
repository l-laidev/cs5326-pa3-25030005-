"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import time
from typing import Any, Callable

from src.extraction import extract_answer, parse_steps
from src.generation import compute_majority_vote, format_step_prompt, run_self_consistency
from src.prm_judge import score_path
from src.utils import SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    if len(step_scores) == 0:
        return -1.0

    if strategy == "min":
        return min(step_scores)
    elif strategy == "product":
        p = [max(min((r_t+1)/2, 0.99), 0.01)
             for r_t in step_scores]
        
        prod = 1.0
        for p_t in p:
            prod *= p_t
        
        return prod
    elif strategy == "step_penalty":
        mean = sum(step_scores) / len(step_scores)
        penalty = sum((r_t == -1.0) for r_t in step_scores)
        return mean - penalty_lambda * penalty
    else:
        raise ValueError(f"Strategy must be 'min', 'product', or 'step_penalty'. Got: {strategy} ")


def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    any_non_none = any(path["answer"] is not None for path in candidate_paths)
    eligible_candidates = list(filter(lambda x: x["answer"] is not None, candidate_paths)) if any_non_none else candidate_paths
    
    path_scores = []
    for path in eligible_candidates:
        path_score = aggregate_path_score(path["step_scores"], aggregation_strategy, penalty_lambda)
        path_scores.append(path_score)
    
    max_score = max(path_scores)
    tied_set = set()
    for i, score in enumerate(path_scores):
        if abs(max_score - score) < 1e-9:
            tied_set.add(i)
    
    if len(tied_set) == 1:
        chosen = tied_set[0]
        return {
            **eligible_candidates[chosen],
            "path_score": path_scores[chosen],
            "num_tied": len(tied_set),  # TODO: is this equal to size of tied set, or size-1
        }
    
    tied_answers = [eligible_candidates[idx]["answer"] for idx in tied_set]
    ans = compute_majority_vote(tied_answers)
    for idx in sorted(tied_set):
        if ans is None or ans == eligible_candidates[idx]["answer"]:
            return {
                **eligible_candidates[idx],
                "path_score": path_scores[idx],
                "num_tied": len(tied_set),
            }


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    aggregation = {
        "generation_calls": 0,
        "generated_tokens": 0,
        "time_seconds": 0,
    }
    if candidates is None:
        sampled = run_self_consistency(model, gen_tokenizer, question, n, generate_fn=generate_fn)
        candidates = sampled["candidates"]
        
        aggregation["generated_tokens"] = sampled["generated_tokens"]
        aggregation["generation_calls"] = sampled["generation_calls"]
        aggregation["time_seconds"] = sampled["time_seconds"]
    
    start = time.perf_counter()
    
    aggregation["candidates"] = []
    aggregation["judge_calls"] = 0
    aggregation["judge_tokens"] = 0
    for candidate in candidates:
        scoring = score_path(judge_model, judge_tokenizer, question, candidate, judge_fn)
        aggregation["judge_calls"] += scoring["judge_calls"]
        aggregation["judge_tokens"] += scoring["judge_tokens"]
        aggregation["candidates"].append(
            {
                **candidate,
                "step_scores": scoring["step_scores"],
                "rationales": scoring["rationales"],
            }
        )
    # total judging time
    aggregation["time_seconds"] += (time.perf_counter() - start)
    
    start = time.perf_counter()
    chosen = select_best_path(aggregation["candidates"], aggregation_strategy, penalty_lambda)
    # selection time
    aggregation["time_seconds"] += (time.perf_counter() - start)
    
    aggregation["selected"] = chosen
    aggregation["answer"] = chosen["answer"]
    
    return aggregation


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    num_to_generate = target_calls
    if existing_candidates is not None:
        num_to_generate = target_calls - len(existing_candidates)
    
    prompt = format_step_prompt(question)
    
    if num_to_generate > 0:
        if generate_fn is None:
            generate_fn = make_hf_generate_fn(model, tokenizer)
        
        start = time.perf_counter()
        responses = generate_fn(prompt=prompt, num_samples=num_to_generate, gen_kwargs={
            **SAMPLING_GEN_KWARGS,
            "temperature": temperature,
        })
        elapsed = time.perf_counter() - start
    else:
        elapsed = 0
        responses = []
    
    candidates = []
    for res in responses:
        candidates.append(
            {
                "completion": res["text"],
                "steps": parse_steps(res["text"]),
                "answer": extract_answer(res["text"]),
                "num_tokens": res["num_tokens"],
            }
        )
    
    if existing_candidates is not None:
        candidates = existing_candidates + candidates
    
    answer = compute_majority_vote([c["answer"] for c in candidates])
    
    return {
        "candidates": candidates,
        "answer": answer,
        "generation_calls": num_to_generate,
        "total_samples": len(candidates),
        "generated_tokens": sum(res["num_tokens"] for res in responses),
        "time_seconds": elapsed,
    }