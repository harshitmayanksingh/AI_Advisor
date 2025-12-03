# management_intro/__init__.py
from otree.api import *


doc = "Introductory page for the AI in Management demo."


class C(BaseConstants):
    NAME_IN_URL = 'management_intro'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    saw_management_intro = models.BooleanField(initial=False)


class Intro(Page):
    def vars_for_template(player: Player):
        return dict()

    def before_next_page(player: Player, timeout_happened=False):
        player.saw_management_intro = True


page_sequence = [Intro]
