# management_assessment/__init__.py
from otree.api import *


doc = "Collects management levers and generates a simple AI-style confidence narrative."


class C(BaseConstants):
    NAME_IN_URL = 'management_assessment'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    scope_clarity = models.IntegerField(label='Scope clarity (0=unclear, 10=precise)', min=0, max=10)
    stakeholder_trust = models.IntegerField(label='Stakeholder trust (0=low, 10=high)', min=0, max=10)
    team_momentum = models.IntegerField(label='Team momentum (0=stalled, 10=strong)', min=0, max=10)
    risk_appetite = models.IntegerField(label='Leader risk appetite (0=aversion, 10=bold)', min=0, max=10)
    ai_confidence = models.FloatField(initial=0)
    ai_recommendation = models.LongStringField(initial='')


def _compute_weighted_score(player: Player) -> float:
    # Emphasize alignment and support, with a softer nudge for risk appetite.
    weights = dict(scope_clarity=0.35, stakeholder_trust=0.3, team_momentum=0.25, risk_appetite=0.1)
    total = (
        player.scope_clarity * weights['scope_clarity']
        + player.stakeholder_trust * weights['stakeholder_trust']
        + player.team_momentum * weights['team_momentum']
        + player.risk_appetite * weights['risk_appetite']
    )
    return round(total, 2)


def _draft_recommendation(score: float) -> str:
    if score >= 8:
        return (
            "Execution posture: Green. The weighted multipliers signal a high likelihood of progress; "
            "push ambitious milestones, maintain weekly signals, and formalize feedback cadences."
        )
    if score >= 5:
        return (
            "Execution posture: Yellow. Momentum is promising but fragile; focus on stakeholder clarity and "
            "fast risk spikes (pilots, shadow launches) to build confidence before scaling."
        )
    return (
        "Execution posture: Red. Foundations need attention; reduce scope, sequence decisions, and ask the "
        "LLM to map low-risk experiments that rebuild trust and momentum."
    )


class Assessment(Page):
    form_model = 'player'
    form_fields = ['scope_clarity', 'stakeholder_trust', 'team_momentum', 'risk_appetite']

    def vars_for_template(player: Player):
        return dict(
            factor_help=[
                ("Scope clarity", "Strong definitions act as multipliers for every downstream decision."),
                ("Stakeholder trust", "Predicts sponsorship durability and risk tolerance."),
                ("Team momentum", "Captures throughput and morale; dips often precede delays."),
                ("Risk appetite", "Adjusts how boldly to commit and when to pause.")
            ]
        )

    def before_next_page(player: Player, timeout_happened=False):
        player.ai_confidence = _compute_weighted_score(player)
        player.ai_recommendation = _draft_recommendation(player.ai_confidence)


class Results(Page):
    def vars_for_template(player: Player):
        return dict(
            score=player.ai_confidence,
            narrative=player.ai_recommendation,
            checklist=[
                "Identify one multiplier per week to nudge upward (scope, trust, or momentum).",
                "Log a short LLM brief asking for scenarios where the current score succeeds or fails.",
                "Quantify progress with one observable signal per factor (e.g., response time, unblock rate).",
            ]
        )


page_sequence = [Assessment, Results]
