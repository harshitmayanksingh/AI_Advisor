# AI_Advisor

This repository now hosts two demo experiences built with oTree:

- **AI Financial Advisor Experiment** – the original multi-step flow for investment decisions.
- **AI in Management Experiment** – a sandbox that mirrors the financial test site but focuses on leadership and project-readiness signals. Participants score management levers, and the app produces an AI-style confidence narrative and checklist.

## Run a live demo locally
1) Install dependencies (Python 3.11+ recommended):
   ```bash
   cd ai_advisor
   pip install -r requirements.txt
   ```
2) Create your environment file:
   ```bash
   cp .env.example .env
   # adjust OTREE_ADMIN_PASSWORD and SECRET_KEY as needed
   ```
3) Launch the site:
   ```bash
   otree devserver 0.0.0.0:8000
   ```
4) Visit <http://localhost:8000/demo> to click into either session configuration:
   - `AI Financial Advisor Experiment` (name: `ai_financial_advisor`)
   - `AI in Management Experiment` (name: `ai_management_advisor`)

You can also use the oTree admin at <http://localhost:8000/admin> (sign in with the password you set) to create study sessions and monitor participants.
