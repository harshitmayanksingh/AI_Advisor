# advisor_common.py
import os, re, textwrap, random, math
from typing import Optional, List, Dict

# ========= Variants (shared across steps) =========
VARIANTS = [
    dict(name='V1', rf=1.00, r=1.10, sigma=0.25),
    dict(name='V2', rf=1.05, r=1.10, sigma=0.25),
    dict(name='V3', rf=1.00, r=1.20, sigma=0.40),
    dict(name='V4', rf=1.05, r=1.20, sigma=0.40),
]

# ========= Client types (can be overridden via session config) =========
CLIENTS = [
    dict(code='C', label='Conservative', p=2.5),
    dict(code='B', label='Balanced',     p=1.5),
    dict(code='A', label='Aggressive',   p=0.9),
]

def clip(x, lo, hi): return max(lo, min(hi, x))

def k_star_for(client_p: float, variant: dict, budget: float = 10.0) -> float:
    """
    Returns k* in DOLLARS for a given budget.
    Uses k*_frac = mu / (2 p sigma^2), clipped to [0,1], then scales by budget.
    """
    mu = float(variant['r']) - 1.0
    sigma = float(variant['sigma'])
    denom = 2.0 * float(client_p) * (sigma ** 2)
    if denom <= 1e-12:
        frac = 0.0 if mu <= 0 else 1.0
    else:
        frac = max(0.0, min(1.0, mu / denom))
    return budget * frac


# ========= AI config (env defaults; session.config takes precedence) =========
ENV_AI_MODE   = os.getenv("AI_MODE", "deterministic")       # 'deterministic' | 'model'
ENV_AI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-5-mini")
ENV_AI_TEMP   = float(os.getenv("OPENAI_TEMP", "0.0"))      # GPT-5 ignores temperature; kept for compatibility
ENV_AI_SEED   = int(os.getenv("OPENAI_SEED", "7"))          # seed may not be supported; handled defensively
ENV_AI_LIMIT  = int(os.getenv("AI_MAX_MSGS_PER_ROUND", "0"))# 0 => unlimited (not enforced here)
ENV_API_KEY   = os.getenv("OPENAI_API_KEY") or ""

# ========= System prompt (guardrails relaxed; stay on-task) =========
CONTROL_PROMPT = textwrap.dedent("""
You are a finance assistant inside a controlled lab experiment.

Your goals:
- Stay strictly relevant to the participant’s current task (this screen).
- Briefly ask about their goal (e.g., protect principal, steady growth, or pursue growth) if unclear.
- Explain concepts clearly: risk-free rate (rf), risky average (expected value), standard deviation σ (dispersion),
  and percentiles (P10/P50/P90), diversification, and trade-offs.
- Help them make an informed decision in plain language.
- Keep answers concise (aim < 120 words), neutral, and easy to read.
""").strip()

# Optional per-stage briefs layered on top of CONTROL_PROMPT
PROMPT_REGISTRY: Dict[str, str] = {
    "S1": (
        "Context: Participant is choosing their own safe vs risky mix. "
        "Start by briefly asking their goal (protect principal, steady growth, pursue growth). "
        "Map goals to risk–return trade-offs and σ (dispersion). Qualitative guidance is fine."
    ),
    "S2": (
        "Context: Participant is advising a specific client (Conservative/Balanced/Aggressive). "
        "Confirm client type and ask what the client values most (stability vs growth). "
        "Explain trade-offs and σ in that client’s terms. Qualitative tilts are fine."
    ),
    "S3": (
        "Context: Participant revisits their own choice. Reinforce earlier concepts and prompt brief reflection."
    ),
}

# ========= Lightweight intent & canned replies (used only in deterministic mode) =========
INTENTS = ["rf_vs_avg", "sigma_spread", "percentiles", "diversification", "payment_rules"]

def classify_intent(user_text: str) -> str:
    t = (user_text or "").lower()
    if any(k in t for k in ["rf", "risk-free", "safe", "avg", "average", "expected", "compare"]):
        return "rf_vs_avg"
    if any(k in t for k in ["sigma", "std", "stdev", "vol", "volatility", "spread", "range", "p10", "p90"]):
        return "sigma_spread"
    if any(k in t for k in ["p10", "p50", "p90", "percentile", "downside", "upside"]):
        return "percentiles"
    if any(k in t for k in ["diversify", "diversification", "mix"]):
        return "diversification"
    if any(k in t for k in ["payment", "bonus", "rim", "random"]):
        return "payment_rules"
    return "rf_vs_avg"

def canonical_answer(intent: str, v: Optional[dict], stage: str) -> str:
    # Generic if no variant passed (kept simple)
    if v is None:
        if intent == "rf_vs_avg":
            return "Compare the risky average (expected value) to rf; choose based on your goal and risk tolerance."
        if intent == "sigma_spread":
            return "σ is the dispersion of outcomes around the average; higher σ means bumpier results."
        if intent == "percentiles":
            return "P10/P50/P90 summarize downside, typical, and upside outcomes; σ indicates spread."
        if intent == "diversification":
            return "Mixing safe and risky balances certainty and growth potential."
        if intent == "payment_rules":
            return "Bonuses follow the experiment’s payment rules."
        return "Use rf vs. risky average and σ to weigh stability vs. growth."
    # Variant-aware (still short & neutral)
    rf = v['rf']; r = v['r']; s = v['sigma']
    if intent == "rf_vs_avg":
        return (f"Per $1, rf={rf:g}, risky average={r:g}. Align with your goal: more stability vs. more growth.")
    if intent == "sigma_spread":
        return (f"σ={s:g} widens the spread of outcomes around the same average. Higher σ → more variability.")
    if intent == "percentiles":
        return "Percentiles mark downside/typical/upside outcomes; use with σ to gauge variability."
    if intent == "diversification":
        return "A mix of rf and risky can smooth outcomes while keeping some growth potential."
    if intent == "payment_rules":
        return "This step pays according to the experiment’s rules."
    return "Weigh rf vs. risky average and consider σ (variability) given your goal."

# ========= Session-aware config helpers =========
def _get_ai_cfg(session_config: dict) -> dict:
    return {
        "mode":    session_config.get("ai_mode", ENV_AI_MODE),
        "model":   session_config.get("openai_model", ENV_AI_MODEL),
        "limit":   int(session_config.get("ai_max_msgs_per_round", ENV_AI_LIMIT)),
        "temp":    float(session_config.get("openai_temp", ENV_AI_TEMP)),
        "seed":    int(session_config.get("openai_seed", ENV_AI_SEED)),
        "api_key": ENV_API_KEY,
    }

def _variant_context(variant: Optional[dict]) -> str:
    if not variant:
        return ""
    return (
        f"Stage variant context — name: {variant.get('name')}, "
        f"rf: {variant.get('rf')}, risky average r: {variant.get('r')}, σ: {variant.get('sigma')}."
    )

def _build_messages(
    stage: str,
    user_text: str,
    *,
    step_prompt: Optional[str] = None,
    variant: Optional[dict] = None,
    history: Optional[List[Dict]] = None
):
    """
    Build Responses API 'input' with system + optional history + current user block.
    history: list of {'role': 'user'|'assistant', 'content': '...'}
    """
    sys = CONTROL_PROMPT if not step_prompt else f"{CONTROL_PROMPT}\n\nStep brief: {step_prompt}"
    ctx = _variant_context(variant)

    # Compose conversation input
    input_blocks: List[Dict[str, str]] = [{"role": "system", "content": sys}]
    if history:
        # Push prior turns in order
        for msg in history:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if content:
                input_blocks.append({"role": role, "content": content})

    # Current user turn with context reminder
    user_block = f"{ctx}\nParticipant asked: \"{user_text}\"".strip()
    input_blocks.append({"role": "user", "content": user_block})

    return input_blocks

# --- Free/guarded toggle (env + session-config aware) ---
ENV_FREE_MODE = (os.getenv("AI_FREE_MODE", "0").strip() in {"1","true","True","yes","YES"})

def _is_free_mode(session_config: dict | None) -> bool:
    # session config wins; falls back to env; default False
    if session_config and "ai_free_mode" in session_config:
        return bool(session_config["ai_free_mode"])
    return ENV_FREE_MODE

def post_filter(text: str, *, session_config: dict | None = None) -> str:
    """
    If free mode is ON, we don't block allocations—just trim and return.
    If free mode is OFF, we can lightly nudge the model back on-topic (no hard blocks).
    """
    out = (text or "").strip()
    if _is_free_mode(session_config):
        return out

    # very light nudge if it drifts way off-topic (no hard censorship)
    lower = out.lower()
    off_topic_hits = any(k in lower for k in ["politics", "celebrity", "weather forecast"])
    on_task = any(k in lower for k in ["rf", "risky", "standard deviation", "sigma", "percentile", "diversification", "goal"])
    if off_topic_hits and not on_task:
        out += "\n\n(Quick nudge: keep it focused on rf vs risky average, σ/percentiles, diversification, and the participant’s goal.)"
    return out


def call_llm_safely(
    session,
    user_text: str,
    stage: str,
    *,
    step_prompt: Optional[str] = None,
    variant: Optional[dict] = None,
    history: Optional[List[Dict]] = None,
    previous_response_id: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """
    Unified entry with optional history + previous_response_id for GPT-5 continuity.
    Returns: (answer_text, new_previous_response_id_or_None)
    """
    cfg   = _get_ai_cfg(session.config)
    mode  = cfg["mode"]
    model = cfg["model"]
    api_key = cfg["api_key"]

    print(f"[AI] mode={mode} model={model} api_key={'set' if bool(api_key) else 'missing'}")

    # Deterministic fallback
    if (mode != "model") or not api_key:
        intent = classify_intent(user_text)
        txt = canonical_answer(intent, variant, stage)
        return post_filter(txt, session_config=session.config), None

    # Model path (Responses API for GPT-5 family)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = _build_messages(stage, user_text, step_prompt=step_prompt, variant=variant, history=history)

        kwargs = dict(
            model=model,
            input=messages,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
            max_output_tokens=350,
        )

        # Pass previous_response_id if available (helps continuity across turns)
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        resp = client.responses.create(**kwargs)

        # Extract text
        out = (getattr(resp, "output_text", None) or "").strip()
        if not out:
            try:
                out = resp.output[0].content[0].text.strip()
            except Exception:
                out = ""

        if not out:
            out = "Based on your goal, weigh the stable rf against the risky average and its σ (variability)."

        # Try to grab the new previous_response_id for next turn continuity
        new_prev_id = None
        try:
            new_prev_id = getattr(resp, "id", None)
        except Exception:
            pass

        return post_filter(out, session_config=session.config), new_prev_id

    except Exception as e:
        print(f"[AI ERROR] {e}")
        return post_filter(
            "Consider your goal first (stability vs. growth). Compare rf (certain) to the risky average and its σ (variability)."
        ), None
