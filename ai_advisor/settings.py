# settings.py
from os import environ
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name('.env'))

# ---- Project-wide defaults (you can override per-session if needed)
SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
    doc="",
    # AI defaults
    ai_mode='model',                # 'model' or 'deterministic'
    openai_model='gpt-5-mini',
    ai_max_msgs_per_round=0,        # 0 = unlimited (you can enforce later)
    ai_free_mode=True,              # relaxed guidance
    force_s1=True, force_s2=True, force_s3=False,
)

# ---- Production session (the only one your participants should see)
SESSION_CONFIGS = [
    dict(
        name='ai_financial_advisor',
        display_name='AI Financial Advisor Experiment',
        num_demo_participants=10,
        app_sequence=[
            'intro',
            'advisor_step1',
            'advisor_step2',
            'advisor_step3',
        ],
    ),
]


# ---- Optional: bring back single-step debug sessions with an env toggle
if environ.get('ADVISOR_DEBUG_STEPS', '').lower() in {'1','true','yes'}:
    SESSION_CONFIGS += [
        dict(
            name='advisor_step1_only',
            display_name='[DEBUG] Step 1 only',
            num_demo_participants=1,
            app_sequence=['advisor_step1'],
            ai_mode='deterministic', force_s1=True,
        ),
        dict(
            name='advisor_step2_only',
            display_name='[DEBUG] Step 2 only',
            num_demo_participants=1,
            app_sequence=['advisor_step2'],
            ai_mode='deterministic', force_s2=True,
        ),
        dict(
            name='advisor_step3_only',
            display_name='[DEBUG] Step 3 only',
            num_demo_participants=1,
            app_sequence=['advisor_step3'],
            ai_mode='deterministic', force_s3=True,
        ),
    ]

# ---- Persistent vars carried across apps (already used in your code)
PARTICIPANT_FIELDS = [
    'S_flags', 'rim_round', 'ai_chat_counts', 'ai_chat_counts_step2',
    'step2_schedule', 'step2_clients',
]
SESSION_FIELDS = []

LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

# Rooms are optional, but handy for a single link in classes/labs
ROOMS = [
    # dict(name='lab', display_name='Behavior Lab', participant_label_file='_room_labels.txt'),
]

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')  # set on Heroku

# Nice, simple landing text on /demo (when OTREE_AUTH_LEVEL=DEMO)
DEMO_PAGE_INTRO_HTML = """
<h3>AI Financial Advisor Experiment</h3>
<p>Click to start a demo session or use links from the admin panel for data collection.</p>
"""

# Replace this before deploying
SECRET_KEY = environ.get('SECRET_KEY', 'replace-me')
INSTALLED_APPS = ['otree']
