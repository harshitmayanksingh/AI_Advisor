# advisor_step1/__init__.py
from otree.api import *
import random

# Centralized logic & LLM utilities
from advisor_common import (
    VARIANTS,                 # list of dicts: name, rf, r, sigma
    call_llm_safely, PROMPT_REGISTRY
)

doc = """Step 1 (4 rounds): own decision with assistant (guardrails configurable via settings)."""

class C(BaseConstants):
    NAME_IN_URL = 'advisor_step1'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 4
    BONUS_PER_DOLLAR = 0.5

class Subsession(BaseSubsession): pass
class Group(BaseGroup): pass

class Player(BasePlayer):
    # Stage toggles (assigned once on round 1)
    S1 = models.BooleanField()
    S2 = models.BooleanField()
    S3 = models.BooleanField()

    # Inputs (per round)
    k_own = models.IntegerField(
        label='How many dollars (0–10) would you put in the risky asset?',
        min=0, max=10
    )
    mc_answer = models.StringField(
        label='Select one option:',
        choices=[('A','A'),('B','B'),('C','C'),('D','D')],
        widget=widgets.RadioSelect
    )
    mc_conf = models.IntegerField(label='Confidence in your answer (0–100)', min=0, max=100)
    mc_correct = models.BooleanField(initial=False)

class AdviceMessage(ExtraModel):
    participant_code = models.StringField()
    session_code     = models.StringField()
    round_number     = models.IntegerField()
    role             = models.StringField()   # 'user' | 'assistant'
    text             = models.LongStringField()
    stage            = models.StringField()   # 'S1'

def creating_session(subsession: Subsession):
    if subsession.round_number == 1:
        players = subsession.get_players()
        cfg = subsession.session.config

        # target proportions (defaults 50/50)
        p_s1 = cfg.get('s1_rate', 0.5)
        p_s2 = cfg.get('s2_rate', 0.5)
        p_s3 = cfg.get('s3_rate', 0.5)

        # optional hard overrides
        force_s1 = cfg.get('force_s1', None)
        force_s2 = cfg.get('force_s2', None)
        force_s3 = cfg.get('force_s3', None)

        n = len(players)
        def balanced_bits(rate: float):
            k = int(round(rate * n))
            bits = [1]*k + [0]*(n-k)
            random.shuffle(bits)
            return bits

        s1_bits = balanced_bits(p_s1)
        s2_bits = balanced_bits(p_s2)
        s3_bits = balanced_bits(p_s3)

        for i, p in enumerate(players):
            s1, s2, s3 = s1_bits[i], s2_bits[i], s3_bits[i]
            if force_s1 is not None: s1 = 1 if force_s1 else 0
            if force_s2 is not None: s2 = 1 if force_s2 else 0
            if force_s3 is not None: s3 = 1 if force_s3 else 0

            p.S1, p.S2, p.S3 = bool(s1), bool(s2), bool(s3)
            p.participant.vars['S_flags'] = dict(S1=p.S1, S2=p.S2, S3=p.S3)

            # per-participant setup
            p.participant.vars.setdefault("rim_round", random.randint(1, C.NUM_ROUNDS))
            p.participant.vars.setdefault("s1_histories", {})      # {round -> [ {role,content}, ... ]}
            p.participant.vars.setdefault("s1_prev_resp_ids", {})  # {round -> response_id}
            p.participant.vars.setdefault("s1_preface_sent", {})   # {round -> bool}

    else:
        # mirror flags from round 1
        for p in subsession.get_players():
            p1 = p.in_round(1)
            p.S1, p.S2, p.S3 = p1.S1, p1.S2, p1.S3

def mc_for_round(round_number: int):
    if round_number == 1:
        return dict(
            prompt="Average per $1 risky is $1.10. If you invest $6 in risky, what is the average risky outcome?",
            options=["$5.40", "$6.60", "$7.10", "I’m not sure"],
            correct='B'
        )
    if round_number == 2:
        return dict(
            prompt=("If you invest $4 in risky and $6 in safe, what is the average total?\n"
                    "(Use average per $1 risky = $1.10; safe per $1 = $1.05.)"),
            options=["$10.40", "$10.70", "$11.00", "I’m not sure"],
            correct='B'
        )
    if round_number == 3:
        return dict(
            prompt="Which number on the chart marks the typical center (P50) per $1?",
            options=["$0.69", "$1.20", "$1.71", "I’m not sure"],
            correct='B'
        )
    if round_number == 4:
        return dict(
            prompt="Increasing rf from 0% to 5% (holding the risky average and σ fixed) mainly:",
            options=["Raises the risky average", "Widens the risky range", "Raises the safe payoff per $1", "I’m not sure"],
            correct='C'
        )
    return dict(prompt="(no question)", options=["A","B","C","D"], correct='D')

# ---- history helpers
def _get_hist(player: Player):
    store = player.participant.vars.get("s1_histories", {})
    hist = store.get(player.round_number)
    if hist is None:
        hist = []
        store[player.round_number] = hist
        player.participant.vars["s1_histories"] = store
    return hist

def _push_hist(player: Player, role: str, content: str):
    hist = _get_hist(player)
    hist.append(dict(role=role, content=content))
    if len(hist) > 6:
        del hist[:-6]

def _get_prev_resp_id(player: Player):
    ids = player.participant.vars.get("s1_prev_resp_ids", {})
    return ids.get(player.round_number)

def _set_prev_resp_id(player: Player, rid: str):
    ids = player.participant.vars.get("s1_prev_resp_ids", {})
    if rid:
        ids[player.round_number] = rid
        player.participant.vars["s1_prev_resp_ids"] = ids

def _preface_sent(player: Player) -> bool:
    flags = player.participant.vars.get("s1_preface_sent", {})
    return bool(flags.get(player.round_number, False))

def _mark_preface_sent(player: Player):
    flags = player.participant.vars.get("s1_preface_sent", {})
    flags[player.round_number] = True
    player.participant.vars["s1_preface_sent"] = flags

def average_total(v: dict, k: int) -> float:
    return k * v['r'] + (10 - k) * v['rf']

class MyPage(Page):
    form_model = 'player'
    form_fields = ['k_own', 'mc_answer', 'mc_conf']

    @staticmethod
    def error_message(player, values):
        errors = {}
        if not values.get('mc_answer'):
            errors['mc_answer'] = "Please select an answer for the quiz."
        if values.get('mc_conf') is None:
            errors['mc_conf'] = "Please enter your confidence."
        return errors or None

    @staticmethod
    def vars_for_template(player: Player):
        v = VARIANTS[player.round_number - 1]
        q = mc_for_round(player.round_number)
        return dict(
            variant=v,
            show_ai_panel=bool(player.S1),
            mc_prompt=q['prompt'],
            mc_options=q['options'],
        )

    @staticmethod
    def live_method(player: Player, data):
        # Gate by stage
        if not player.S1:
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
            stage='S1',
        )

        # Preface on first turn
        if not _preface_sent(player):
            preface = (
                "Hi—I’m your assistant. Tell me your goal (protect principal, steady growth, or pursue growth). "
                "I’ll explain rf (safe rate), the risky average (expected value), and σ (dispersion) so you can decide."
            )
            _push_hist(player, "assistant", preface)
            _mark_preface_sent(player)

        _push_hist(player, "user", user_text)

        # Build answer
        v = VARIANTS[player.round_number - 1]
        history = _get_hist(player)
        prev_id = _get_prev_resp_id(player)

        step_prompt = PROMPT_REGISTRY.get("S1", "")
        answer, resp_id = call_llm_safely(
            session=player.session,
            user_text=user_text,
            stage='S1',
            step_prompt=step_prompt,
            variant=v,
            history=history,
            previous_response_id=prev_id,
        )

        if resp_id:
            _set_prev_resp_id(player, resp_id)

        _push_hist(player, "assistant", answer)

        # Log ASSISTANT (same table as user so export sees both)
        AdviceMessage.create(
            participant_code=player.participant.code,
            session_code=player.session.code,
            round_number=player.round_number,
            role='assistant',
            text=answer,
            stage='S1',
        )

        return {player.id_in_group: dict(type='answer', text=answer)}

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        correct = mc_for_round(player.round_number)['correct']
        player.mc_correct = (player.mc_answer == correct)

class Results(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        rows = []
        for r in player.in_all_rounds():
            v = VARIANTS[r.round_number - 1]
            rows.append(dict(
                rnd=r.round_number,
                v_name=v['name'],
                rf=v['rf'],
                r=v['r'],
                sigma=v['sigma'],
                k=r.k_own,
                mc=r.mc_answer,
                correct=r.mc_correct,
            ))
        total_correct = sum(1 for r in player.in_all_rounds() if r.mc_correct)

        rim_round = player.participant.vars.get('rim_round', 1)
        rim_player = player.in_round(rim_round)
        v = VARIANTS[rim_round - 1]
        total_dollars = average_total(v, rim_player.k_own)
        bonus_points = total_dollars * C.BONUS_PER_DOLLAR

        rim_k = rim_player.k_own
        rim_safe = 10 - rim_k

        return dict(
            rows=rows,
            total_correct=total_correct,
            rim_round=rim_round,
            rim_variant=v['name'],
            rim_k=rim_k,
            rim_rf=v['rf'],
            rim_r=v['r'],
            rim_total=round(total_dollars, 2),
            rim_bonus=round(bonus_points, 2),
            rim_safe=rim_safe,
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        rim_round = player.participant.vars.get('rim_round', 1)
        if player.round_number == rim_round:
            v = VARIANTS[player.round_number - 1]
            total_dollars = average_total(v, player.k_own)
            player.payoff = total_dollars * C.BONUS_PER_DOLLAR
        else:
            player.payoff = 0

def custom_export(players):
    """Long (tidy) export of Step 1 chat messages from AdviceMessage."""
    yield ["session_code", "participant_code", "round_number", "stage", "role", "text"]
    if not players:
        return
    session_code = players[0].session.code
    for m in sorted(
        AdviceMessage.filter(session_code=session_code, stage='S1'),
        key=lambda x: (x.round_number,)
    ):
        yield [m.session_code, m.participant_code, m.round_number, m.stage, m.role, m.text]

page_sequence = [MyPage, Results]
