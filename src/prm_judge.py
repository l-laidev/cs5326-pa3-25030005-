"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import re
from typing import Any, Callable

from src.utils import make_hf_judge_fn

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    formatted_preceding_steps = [f"Step {i+1}: {step}"
                                 for i, step in enumerate(preceding_steps)]
    
    if len(formatted_preceding_steps) > 0:
        formatted_preceding_steps = "\n".join(formatted_preceding_steps)
    else:
        formatted_preceding_steps = "(no preceding steps)"
    
    return f"""
PROBLEM: {question}
Preceding Steps:
{formatted_preceding_steps}
Current Step: {current_step}

Assume all preceding steps are correct.
You must judge whether the deduction in "Current Step" is VALID or INVALID.
A step is VALID if its arithmetic is sound and it follows logically from the problem statement plus preceding steps as written. An error in an earlier step does NOT make subsequent steps invalid by itself.

You must reply following this structure:
Verdict: VALID OR Verdict: INVALID
Rationale: <your reasnoning or rationale for why the Current Step is deemed VALID or INVALID>
""".strip()


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    score = -1.0
    verdict_found = False

    # Rule 1: Check line by line for explicit verdict pattern
    for line in judge_output.splitlines():
        m = re.search(r'verdict\s*:\s*\**\s*(VALID|INVALID)\b', line, re.IGNORECASE)
        if m:
            v = m.group(1).upper()
            if v == 'VALID':
                score = 1.0
            else:
                score = -1.0
            verdict_found = True
            break
            
    # Rules 2, 3, 4: Fallback checks across the entire output if no verdict line found
    if not verdict_found:
        if re.search(r'\bINVALID\b|\bNOT\s+VALID\b', judge_output, re.IGNORECASE):
            score = -1.0
        elif re.search(r'\bVALID\b', judge_output, re.IGNORECASE):
            score = 1.0
        else:
            score = -1.0
            
    # Parse rationale
    rationale = ""
    match = re.search(r'rationale\s*:\s*', judge_output, re.IGNORECASE)
    if match:
        start_idx = match.end()
        remaining = judge_output[start_idx:]
        # Find first newline or sentence end (. ! ? followed by whitespace or end of string)
        end_match = re.search(r'[\r\n]|[.!?](?=\s|$)', remaining)
        if end_match:
            end_idx = end_match.start()
            rationale = remaining[:end_idx+1].strip()
        else:
            rationale = remaining.strip()
            
    return score, rationale


def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    
    if judge_fn is None:
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)
    
    steps = candidate["steps"]
    scores = []
    rationales = []
    judge_stats = {"judge_calls": 0, "judge_tokens": 0}
    for i in range(len(steps)):
        preceding_steps = steps[:i]
        current_step = steps[i]
        
        prompt = build_prm_judge_prompt(question, preceding_steps, current_step)
        
        decision = judge_fn(prompt)
        judge_stats["judge_calls"] += 1
        judge_stats["judge_tokens"] += decision["num_tokens"]
        
        (score, rationale) = parse_judge_decision(decision["text"])
        
        scores.append(score)
        rationales.append(rationale)
    
    return {
        "step_scores": scores,
        "rationales": rationales,
        "judge_calls": judge_stats["judge_calls"],
        "judge_tokens": judge_stats["judge_tokens"],
    }