"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
from collections import Counter
import time
from typing import Any, Callable

from src.extraction import extract_answer, parse_steps
from src.utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the 
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.
    
    Returns a raw instruction string (no chat template applied).
    """
    return f"""
You must strictly adhere to the following format:
Step 1: <First deduction or arithmetic operation>
Step 2: <Second deduction conditioning on Step 1>
...
Step N: <Final calculation>
Final Answer: <Integer>

Edge Cases:
- Never use markdown.
- Always separate each step by a newline.
- The final answer must follow this exact pattern: "Final Answer: "
- The final answer must always be an integer.

Solve this problem following the format guidelines:
{question}
""".strip()


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    answers = list(filter(lambda x: x is not None, answers))
    counts = Counter(answers)
    
    ans_ordering = dict()
    for i, ans in enumerate(answers):
        if ans not in ans_ordering:
            ans_ordering[ans] = i
    
    majority_ans = None
    majority_vote = None
    for (ans, vote) in counts.most_common():
        if majority_ans is None:
            majority_ans = ans
            majority_vote = vote
        else:
            if vote > majority_vote:
                majority_ans = ans
                majority_vote = vote
            elif vote == majority_vote and ans_ordering[ans] < ans_ordering[majority_ans]:
                majority_ans = ans
                majority_vote = vote
    
    return answers


def run_greedy(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.
    
    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    prompt = format_step_prompt(question)
    
    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)
    
    start = time.perf_counter()
    response = generate_fn(prompt=prompt, num_samples=1, gen_kwargs=GREEDY_GEN_KWARGS)
    elapsed = time.perf_counter() - start
    
    response = response[0]
    
    return {
        "prompt": prompt,
        "completion": response["text"],
        "steps": parse_steps(response["text"]),
        "answer": extract_answer(response["text"]),
        "generation_calls": 1,
        "generated_tokens": response["num_tokens"],
        "time_seconds": elapsed,
    }


def run_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    temperature: float = 0.7, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.
    
    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer, 
      generation_calls, generated_tokens, time_seconds
    """
    prompt = format_step_prompt(question)
    
    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)
    
    start = time.perf_counter()
    responses = generate_fn(prompt=prompt, num_samples=n, gen_kwargs={
        **SAMPLING_GEN_KWARGS,
        "temperature": temperature,
    })
    elapsed = time.perf_counter() - start
    
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
    
    answer = compute_majority_vote([c["answer"] for c in candidates])
    
    return {
        "candidates": candidates,
        "answer": answer,
        "generation_calls": n,
        "generated_tokens": sum(res["num_tokens"] for res in responses),
        "time_seconds": elapsed,
    }
