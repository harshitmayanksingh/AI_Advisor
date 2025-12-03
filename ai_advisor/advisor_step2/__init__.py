# advisor_step2/__init__.py
from otree.api import *
import random
from typing import List, Dict

# Single source of truth
from advisor_common import (
    VARIANTS, CLIENTS, k_star_for,
    call_llm_safely, PROMPT_REGISTRY
)

doc = """Step 2 (12 rounds): advise clients with assistant (guardrails configurable via settings)."""


# =========================
# Constants / Models
# =========================
class C(BaseConstants):
    NAME_IN_URL = 'advisor_step2'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 12  # 3 clients × 4 variants each (client-major blocks)

class Subsession(BaseSubsession): pass
class Group(BaseGroup): pass

class AdviceMessage(ExtraModel):
    participant_code = models.StringField()
    session_code     = models.StringField()
    round_number     = models.IntegerField()
    role             = models.StringField()     # 'user' | 'assistant'
    text             = models.LongStringField()
    stage            = models.StringField()     # 'S2'

class Player(BasePlayer):
    # Stage flag (forced via config or inherited from Step 1)
    S2 = models.BooleanField(initial=False)

    # Variant shown this round
    variant_name = models.StringField()
    rf   = models.FloatField()
    r    = models.FloatField()
    sigma= models.FloatField()

    # Client for this round (fixed for each 4-round block)
    client_code  = models.StringField()
    client_label = models.StringField()
    client_p     = models.FloatField()

    # Inputs
    advised_k   = models.IntegerField(label='How many dollars (0–10) do you advise in the risky asset?', min=0, max=10)
    advice_conf = models.IntegerField(label='Confidence in your advice (0–100)', min=0, max=100)

    # Computed metrics
    k_star      = models.FloatField()
    align_error = models.FloatField()
    proj_gap    = models.FloatField()

    # AI usage tracking (per round)
    used_ai    = models.BooleanField(initial=False)
    ai_q_count = models.IntegerField(initial=0)


# =========================
# Clients + Scheduling
# =========================
def _get_clients(session_config: dict) -> List[Dict]:
    """Allow session-level overrides via settings.py: clients_override=[...]"""
    override = session_config.get('clients_override')
    if isinstance(override, list) and override:
        return override
    return CLIENTS

def _make_schedule_client_blocks(clients: List[Dict]):
    """
    Build a 12-row schedule of (variant_idx, client_idx) with CLIENT-MAJOR blocks:
      - Randomize the order of the clients
      - Within each client: fixed V1..V4 (can randomize later if desired)
    """
    order = list(range(len(clients)))   # e.g., [0,1,2]
    random.shuffle(order)
    sched = []
    for c_idx in order:
        for v_idx in range(len(VARIANTS)):     # fixed V1..V4
            sched.append((v_idx, c_idx))
    return sched


# =========================
# Session init
# =========================
def creating_session(subsession: Subsession):
    cfg = subsession.session.config

    if subsession.round_number == 1:
        session_clients = _get_clients(cfg)
        for p in subsession.get_players():
            # Decide S2 once (from settings or participant flags) and store it
            force_s2 = cfg.get('force_s2', None)
            if force_s2 is not None:
                p.S2 = bool(force_s2)
            else:
                flags = p.participant.vars.get('S_flags', {})
                p.S2 = bool(flags.get('S2', False))
            p.participant.vars['S2_enabled'] = p.S2  # persist the decision

            # One-time per participant
            p.participant.vars['step2_clients'] = session_clients
            p.participant.vars['step2_schedule'] = _make_schedule_client_blocks(session_clients)
            p.participant.vars.setdefault("s2_histories", {})
            p.participant.vars.setdefault("s2_prev_resp_ids", {})
            p.participant.vars.setdefault("s2_preface_sent", {})

    # For every round: re-apply the persisted flag to this round's Player
    for p in subsession.get_players():
        p.S2 = bool(p.participant.vars.get('S2_enabled', False))

        clients = p.participant.vars.get('step2_clients') or _get_clients(cfg)
        sched = p.participant.vars.get('step2_schedule')
        if not sched or len(sched) < C.NUM_ROUNDS:
            sched = _make_schedule_client_blocks(clients)
            p.participant.vars['step2_schedule'] = sched

        v_idx, c_idx = sched[subsession.round_number - 1]
        v, c = VARIANTS[v_idx], clients[c_idx]
        p.variant_name = v['name']
        p.rf, p.r, p.sigma = v['rf'], v['r'], v['sigma']
        p.client_code, p.client_label, p.client_p = c['code'], c['label'], float(c['p'])



# =========================
# UI helpers
# =========================
def client_profile(code: str):
    profiles = {
        "C": dict(label="Conservative", summary="Prioritizes stability and protecting principal.", bullets=[
            "Goal: preserve savings for near-term needs",
            "Tolerates modest ups/downs; prefers steadier outcomes",
            "Sleeps better with fewer surprises",
        ]),
        "B": dict(label="Balanced", summary="Wants a mix of steadiness and growth.", bullets=[
            "Goal: grow steadily while avoiding big drawdowns",
            "Comfortable with some variability for potentially higher average outcomes",
            "Open to diversification to smooth the ride",
        ]),
        "A": dict(label="Aggressive", summary="Focuses on growth and accepts larger swings.", bullets=[
            "Goal: pursue higher long-run growth",
            "Accepts notable variability around the average",
            "Understands that higher σ means wider upside/downside",
        ]),
    }
    return profiles.get(code, dict(
        label="General",
        summary="No specific preferences provided.",
        bullets=[
            "Open to trade-offs between safe rf and risky average r",
            "Wants clear explanations rather than specific amounts",
        ],
    ))


# =========================
# Small helpers (history)
# =========================
def _get_hist(player: Player):
    store = player.participant.vars.get("s2_histories", {})
    hist = store.get(player.round_number)
    if hist is None:
        hist = []
        store[player.round_number] = hist
        player.participant.vars["s2_histories"] = store
    return hist

def _push_hist(player: Player, role: str, content: str):
    hist = _get_hist(player)
    hist.append(dict(role=role, content=content))
    if len(hist) > 6:  # keep lean
        del hist[:-6]

def _get_prev_resp_id(player: Player):
    ids = player.participant.vars.get("s2_prev_resp_ids", {})
    return ids.get(player.round_number)

def _set_prev_resp_id(player: Player, rid: str):
    ids = player.participant.vars.get("s2_prev_resp_ids", {})
    if rid:
        ids[player.round_number] = rid
        player.participant.vars["s2_prev_resp_ids"] = ids

def _preface_sent(player: Player) -> bool:
    flags = player.participant.vars.get("s2_preface_sent", {})
    return bool(flags.get(player.round_number, False))

def _mark_preface_sent(player: Player):
    flags = player.participant.vars.get("s2_preface_sent", {})
    flags[player.round_number] = True
    player.participant.vars["s2_preface_sent"] = flags


# Step-specific assistant brief (optional additive to global rules)
STEP2_PROMPT = (
     "You help the participant ADVISE a client. Stay focused on the current screen/task. "
     "Condition on the client’s risk type (Conservative/Balanced/Aggressive) and the participant’s stated goal. "
     "Explain the risk–return trade-off and σ (dispersion) in the client’s terms. Keep it concise and clear."
 )


# =========================
# Pages
# =========================
class AdvicePage(Page):
    form_model  = 'player'
    form_fields = ['advised_k', 'advice_conf']

    @staticmethod
    def vars_for_template(player: Player):
        v = dict(name=player.variant_name, rf=player.rf, r=player.r, sigma=player.sigma)
        prof = client_profile(player.client_code)

        clients = player.participant.vars.get('step2_clients') or _get_clients(player.session.config)
        name_fallbacks = ['Alex', 'Riley', 'Jamie', 'Taylor', 'Casey']
        roster = [
            dict(
                code=c['code'],
                label=c['label'],
                name=name_fallbacks[i % len(name_fallbacks)],
                one_liner=client_profile(c['code'])['summary'],
            )
            for i, c in enumerate(clients)
        ]

        return dict(
            variant=v,
            client=dict(code=player.client_code, label=player.client_label),
            client_profile=prof,
            client_roster=roster,
            # ← policy decided at session creation from settings.force_s2
            show_ai_panel=bool(player.S2),
        )


    @staticmethod
    def error_message(player: Player, values):
        errors = {}
        if values.get('advised_k') is None:
            errors['advised_k'] = "Please enter how many dollars to advise."
        if values.get('advice_conf') is None:
            errors['advice_conf'] = "Please enter your confidence."
        return errors or None

    @staticmethod
    def live_method(player: Player, data):
        # Gate by stage
        if not player.S2:
            return {player.id_in_group: dict(type='answer', text='Assistant is off for this step.')}
        if data.get('type') != 'ask':
            return {player.id_in_group: dict(type='answer', text='(Unrecognized message)')}

        user_text = (data.get('text') or '').strip()
        if not user_text:
            return {player.id_in_group: dict(type='answer', text='Please type a question.')}

        # Flat audit log
        AdviceMessage.create(
            participant_code=player.participant.code,
            session_code=player.session.code,
            round_number=player.round_number,
            role='user',
            text=user_text,
            stage='S2',
        )

        # In-round history + first-turn preface
        preface = None
        if not _preface_sent(player):
            client_name = player.client_label or {"C": "Conservative", "B": "Balanced", "A": "Aggressive"}.get(
                player.client_code, "the client")
            preface = (
                f"Hi—I'm your assistant for advising. You're currently advising the {client_name} client. "
                "Briefly tell me the client’s primary goal (protect principal, steady growth, or pursue growth), "
                "and I’ll map it to the risk–return trade-off and σ (dispersion)."
            )
            _push_hist(player, "assistant", preface)
            _mark_preface_sent(player)

        # --- ADD THIS BLOCK (naive goal capture) ---
        lt = user_text.lower()
        if "protect" in lt or "principal" in lt:
            player.participant.vars.setdefault("s2_last_goal", {})[player.round_number] = "protect principal"
        elif "steady" in lt or "balance" in lt:
            player.participant.vars.setdefault("s2_last_goal", {})[player.round_number] = "steady growth"
        elif "growth" in lt or "aggress" in lt:
            player.participant.vars.setdefault("s2_last_goal", {})[player.round_number] = "pursue growth"
        # -------------------------------------------
        # (Optional) Echo the recognized goal into the in-round history so the model sees it
        last_goal = player.participant.vars.get("s2_last_goal", {}).get(player.round_number)

        if last_goal:
            _push_hist(
                player,
            "assistant",
            f"(Context note: The client’s stated goal this round is: {last_goal}.)"
            )
        # Centralized assistant call with continuity
        v = dict(name=player.variant_name, rf=player.rf, r=player.r, sigma=player.sigma)
        history = _get_hist(player)
        prev_id = _get_prev_resp_id(player)

        step_prompt = PROMPT_REGISTRY.get("S2", "")
        # Merge our S2 brief (keeps both)
        if STEP2_PROMPT and step_prompt:
            step_prompt = f"{step_prompt}\n\n{STEP2_PROMPT}"
        elif STEP2_PROMPT:
            step_prompt = STEP2_PROMPT

        answer, resp_id = call_llm_safely(
            session=player.session,
            user_text=user_text,
            stage='S2',
            step_prompt=step_prompt,
            variant=v,
            history=history,
            previous_response_id=prev_id,
        )

        # Compose preface + model answer for first turn
        answer_to_send = answer

        # Persist continuity + audit log
        if resp_id:
            _set_prev_resp_id(player, resp_id)

        _push_hist(player, "assistant", answer)

        AdviceMessage.create(
            participant_code=player.participant.code,
            session_code=player.session.code,
            round_number=player.round_number,
            role='assistant',
            text=answer,
            stage='S2',
        )

        # Round usage flags
        player.used_ai = True
        player.ai_q_count = (player.ai_q_count or 0) + 1

        return {player.id_in_group: dict(type='answer', text=answer_to_send)}

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        v = dict(rf=player.rf, r=player.r, sigma=player.sigma)
        kstar = float(k_star_for(player.client_p, v))  # dollars
        player.k_star = kstar
        player.align_error = abs(float(player.advised_k) - kstar)
        player.proj_gap = float(player.advised_k) - kstar


class AdviceResults(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        rows = []
        rounds = player.in_all_rounds()
        for r in rounds:
            rows.append(dict(
                rnd=r.round_number,
                variant=r.variant_name,
                rf=r.rf, risky_avg=r.r, sigma=r.sigma,
                client=f"{r.client_label} ({r.client_code})",
                p=r.client_p,
                advised_k=r.advised_k,
                k_star=(round(r.k_star, 2) if r.k_star is not None else None),
                align_error=(round(r.align_error, 2) if r.align_error is not None else None),
                proj_gap=(round(r.proj_gap, 2) if r.proj_gap is not None else None),
                conf=r.advice_conf,
                used_ai="Yes" if getattr(r, "used_ai", False) else "No",
                ai_q_count=(getattr(r, "ai_q_count", 0) or 0),
            ))

        total_ai_rounds = sum(1 for r in rounds if getattr(r, "used_ai", False))
        total_ai_qs = sum((getattr(r, "ai_q_count", 0) or 0) for r in rounds)

        return dict(
            rows=rows,
            total_ai_rounds=total_ai_rounds,
            total_ai_qs=total_ai_qs,
        )


def custom_export(players):
    """Long (tidy) export of Step 2 chat messages from AdviceMessage."""
    # Header row
    yield [
        "session_code", "participant_code", "round_number",
        "stage", "role", "text"
    ]
    # Grab all S2 messages from this session
    # (players is a queryset restricted to this app/session)
    if not players:
        return
    session_code = players[0].session.code
    for m in sorted(
            AdviceMessage.filter(session_code=session_code, stage='S2'),
            key=lambda x: (x.round_number,)
    ):
        yield [
            m.session_code,
            m.participant_code,
            m.round_number,
            m.stage,
            m.role,          # 'user' | 'assistant'
            m.text,
        ]
page_sequence = [AdvicePage, AdviceResults]
