# advisor_step3/__init__.py
from otree.api import *
from advisor_common import VARIANTS, call_llm_safely, PROMPT_REGISTRY

doc = """Step 3 (4 rounds): participant revisits OWN allocation with concept-only assistant (no numeric advice)."""

class C(BaseConstants):
    NAME_IN_URL = 'advisor_step3'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 4  # V1..V4

class Subsession(BaseSubsession): pass
class Group(BaseGroup): pass

# --- Unified chat log (same name/shape as Steps 1–2), stage='S3'
class AdviceMessage(ExtraModel):
    participant_code = models.StringField()
    session_code     = models.StringField()
    round_number     = models.IntegerField()
    role             = models.StringField()    # 'user' | 'assistant'
    text             = models.LongStringField()
    stage            = models.StringField()    # 'S3'

class Player(BasePlayer):
    # Stage flag (decided once in round 1)
    S3 = models.BooleanField(initial=False)

    # Variant (deterministic by round)
    variant_name = models.StringField()
    rf   = models.FloatField()
    r    = models.FloatField()
    sigma= models.FloatField()

    # Inputs
    k_own     = models.IntegerField(label='How many dollars (0–10) do you put in the risky asset?', min=0, max=10)
    mc_answer = models.StringField(
        label='Which statement best describes σ (sigma) here?',
        choices=[('A','A'),('B','B'),('C','C'),('D','D')],
        widget=widgets.RadioSelect
    )
    mc_conf   = models.IntegerField(label='Confidence (0–100)', min=0, max=100)

    # AI usage (per-round)
    used_ai    = models.BooleanField(initial=False)
    ai_q_count = models.IntegerField(initial=0)

def creating_session(subsession: Subsession):
    """Set S3 once; prep continuity stores; assign per-round variants."""
    if subsession.round_number == 1:
        force_s3 = subsession.session.config.get('force_s3', None)
        for p in subsession.get_players():
            if force_s3 is not None:
                p.S3 = bool(force_s3)
            else:
                flags = p.participant.vars.get('S_flags', {})
                p.S3 = bool(flags.get('S3', False))

            # continuity stores
            p.participant.vars.setdefault("s3_histories", {})      # {round -> [ {role,content}, ... ]}
            p.participant.vars.setdefault("s3_prev_resp_ids", {})  # {round -> response_id}
            p.participant.vars.setdefault("s3_preface_sent", {})   # {round -> bool}

    # assign variant each round
    for p in subsession.get_players():
        v = VARIANTS[subsession.round_number - 1]
        p.variant_name = v['name']
        p.rf, p.r, p.sigma = v['rf'], v['r'], v['sigma']

# ---- Assistant brief for S3 (merged with registry)
STEP3_PROMPT = (
    "You help the participant reflect on their OWN allocation choice. Explain concepts only (no numbers). "
    "Reinforce rf (safe rate) vs risky average and σ as dispersion; never recommend specific amounts. "
    "Tie explanations back to the participant’s stated goal (e.g., steadier outcomes vs potential growth)."
)

# ---- continuity helpers
def _get_hist(player: Player):
    store = player.participant.vars.get("s3_histories", {})
    hist = store.get(player.round_number)
    if hist is None:
        hist = []
        store[player.round_number] = hist
        player.participant.vars["s3_histories"] = store
    return hist

def _push_hist(player: Player, role: str, content: str):
    hist = _get_hist(player)
    hist.append(dict(role=role, content=content))
    if len(hist) > 6:
        del hist[:-6]

def _get_prev_resp_id(player: Player):
    ids = player.participant.vars.get("s3_prev_resp_ids", {})
    return ids.get(player.round_number)

def _set_prev_resp_id(player: Player, rid: str):
    ids = player.participant.vars.get("s3_prev_resp_ids", {})
    if rid:
        ids[player.round_number] = rid
        player.participant.vars["s3_prev_resp_ids"] = ids

def _preface_sent(player: Player) -> bool:
    flags = player.participant.vars.get("s3_preface_sent", {})
    return bool(flags.get(player.round_number, False))

def _mark_preface_sent(player: Player):
    flags = player.participant.vars.get("s3_preface_sent", {})
    flags[player.round_number] = True
    player.participant.vars["s3_preface_sent"] = flags

class OwnDecision(Page):
    form_model  = 'player'
    form_fields = ['k_own', 'mc_answer', 'mc_conf']

    @staticmethod
    def vars_for_template(player: Player):
        v = dict(name=player.variant_name, rf=player.rf, r=player.r, sigma=player.sigma)
        prompt = "Which statement best describes σ (sigma) here?"
        options = [
            "A. σ is the spread of risky outcomes around the average.",
            "B. σ is the certain payoff of the safe asset.",
            "C. σ is the guaranteed minimum of the risky asset.",
            "D. σ is the probability of a loss in the risky asset.",
        ]
        return dict(variant=v, mc_prompt=prompt, mc_options=options, show_ai_panel=bool(player.S3))

    @staticmethod
    def error_message(player: Player, values):
        errs = {}
        if values.get('k_own') is None:    errs['k_own'] = "Please choose how many dollars to put in the risky asset."
        if values.get('mc_answer') is None: errs['mc_answer'] = "Please answer the question."
        if values.get('mc_conf') is None:   errs['mc_conf'] = "Please report your confidence."
        return errs or None

    @staticmethod
    def live_method(player: Player, data):
        if not player.S3:
            return {player.id_in_group: dict(type='answer', text='Assistant is off for this step.')}
        if data.get('type') != 'ask':
            return {player.id_in_group: dict(type='answer', text='(Unrecognized message)')}

        user_text = (data.get('text') or '').strip()
        if not user_text:
            return {player.id_in_group: dict(type='answer', text='Please type a question.')}

        # Log USER
        AdviceMessage.create(
            participant_code=player.participant.code,
            session_code=player.session.code,
            round_number=player.round_number,
            role='user',
            text=user_text,
            stage='S3',
        )

        # First-turn preface (keep only in history)
        if not _preface_sent(player):
            preface = (
                "Hi—I’m your concept-only assistant for your OWN decision. "
                "What’s your goal—steadier outcomes, potential growth, or in between? "
                "(I’ll explain trade-offs without giving numbers.)"
            )
            _push_hist(player, "assistant", preface)
            _mark_preface_sent(player)

        _push_hist(player, "user", user_text)

        # LLM call with continuity
        v = dict(name=player.variant_name, rf=player.rf, r=player.r, sigma=player.sigma)
        history = _get_hist(player)
        prev_id = _get_prev_resp_id(player)

        step_prompt = PROMPT_REGISTRY.get("S3", "")
        if STEP3_PROMPT and step_prompt:
            step_prompt = f"{step_prompt}\n\n{STEP3_PROMPT}"
        elif STEP3_PROMPT:
            step_prompt = STEP3_PROMPT

        answer, resp_id = call_llm_safely(
            session=player.session,
            user_text=user_text,
            stage='S3',
            step_prompt=step_prompt,
            variant=v,
            history=history,
            previous_response_id=prev_id,
        )

        if resp_id:
            _set_prev_resp_id(player, resp_id)

        _push_hist(player, "assistant", answer)

        # Log ASSISTANT
        AdviceMessage.create(
            participant_code=player.participant.code,
            session_code=player.session.code,
            round_number=player.round_number,
            role='assistant',
            text=answer,
            stage='S3',
        )

        player.used_ai = True
        player.ai_q_count = (player.ai_q_count or 0) + 1
        return {player.id_in_group: dict(type='answer', text=answer)}

class OwnResults(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        rows = []
        for r in player.in_all_rounds():
            rows.append(dict(
                rnd=r.round_number,
                variant=r.variant_name,
                rf=r.rf, risky_avg=r.r, sigma=r.sigma,
                k_own=r.k_own,
                mc=r.mc_answer, conf=r.mc_conf,
                used_ai=("Yes" if r.used_ai else "No"),
                ai_qs=r.ai_q_count or 0,
            ))
        total_ai_rounds = sum(1 for r in player.in_all_rounds() if r.used_ai)
        total_ai_qs = sum((r.ai_q_count or 0) for r in player.in_all_rounds())
        return dict(rows=rows, total_ai_rounds=total_ai_rounds, total_ai_qs=total_ai_qs)

# --- Export just Step 3 chat (tidy long)
def custom_export(players):
    yield ["session_code", "participant_code", "round_number", "stage", "role", "text"]
    if not players:
        return
    session_code = players[0].session.code
    for m in sorted(
        AdviceMessage.filter(session_code=session_code, stage='S3'),
        key=lambda x: (x.round_number,)
    ):
        yield [m.session_code, m.participant_code, m.round_number, m.stage, m.role, m.text]

page_sequence = [OwnDecision, OwnResults]
