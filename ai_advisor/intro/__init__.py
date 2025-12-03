# intro/__init__.py
from otree.api import *

doc = "Intro page shown before Step 1."

class C(BaseConstants):
    NAME_IN_URL = 'intro'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    # If you ever want to toggle AI visibility from this page, you can store flags here.
    saw_intro = models.BooleanField(initial=False)

class Intro(Page):
    def vars_for_template(player: Player):
        return dict()

    def before_next_page(player: Player, timeout_happened=False):
        # Small breadcrumb you can export later if desired.
        player.saw_intro = True

page_sequence = [Intro]
