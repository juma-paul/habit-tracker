"""Help content for the habit-tracker app."""

HELP_CONTENT = """# habit-tracker - Your AI Habit Tracking Assistant

Type `/help` anytime to see this guide.

---

## Quick Commands

| Action | How to Ask | Example |
|--------|-----------|---------|
| **Create Habit** | "create a [name] habit" | "create a running habit for 5 km daily" |
| **List Habits** | "show my habits" | "what habits do I have?" |
| **Log Activity** | "I did [activity]" | "I ran 3 km today" |
| **View Progress** | "how am I doing with [habit]?" | "show my running progress" |
| **Update Habit** | "change [habit] to [new value]" | "update my running goal to 10 km" |
| **Delete Habit** | "delete [habit]" | "remove my running habit" |
| **Get Help** | "/help" | "type /help anytime" |

---

## Example Conversations

Try saying:

- "Create a reading habit for 30 min daily"
- "I walked 8,000 steps"
- "Show my habits"
- "How am I doing with running?"
- "Update my reading goal to 45 min"

---

## Tips for Best Results

- **Be Natural** — Say "I walked 8,000 steps" instead of "log 8000 for walking"
- **Use Voice** — Tap the microphone to speak your updates
- **Check Progress** — Ask "how am I doing?" for an overview
- **Set Targets** — Include goals like "5 km daily" when creating habits

---

## Habit Types

| Frequency | Best For | Example |
|-----------|----------|---------|
| **Daily** | Regular activities | Exercise, meditation, reading |
| **Weekly** | Less frequent goals | Deep cleaning, meal prep |
| **Monthly** | Long-term tracking | Budget reviews, goal check-ins |

---

## Understanding Your Progress

When you ask about progress, you'll see:

- **Total logs** — Number of times you've logged activity
- **Completion rate** — Percentage of your target achieved
- **Average value** — Your typical performance
- **Last logged** — When you last updated

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift + Enter` | New line in message |

---

## Need More Help?

Just ask naturally! Try:

- "What can you help me with?"
- "How do I track my reading habit?"
- "Show me an example of logging exercise"
"""

HELP_DATA = {
    "commands": [
        {
            "id": "create",
            "action": "Create Habit",
            "how": "create a [name] habit",
            "example": "create a running habit for 5 km daily",
        },
        {
            "id": "list",
            "action": "List Habits",
            "how": "show my habits",
            "example": "what habits do I have?",
        },
        {
            "id": "log",
            "action": "Log Activity",
            "how": "I did [activity]",
            "example": "I ran 3 km today",
        },
        {
            "id": "progress",
            "action": "View Progress",
            "how": "how am I doing with [habit]?",
            "example": "show my running progress",
        },
        {
            "id": "update",
            "action": "Update Habit",
            "how": "change [habit] to [new value]",
            "example": "update my running goal to 10 km",
        },
        {
            "id": "delete",
            "action": "Delete Habit",
            "how": "delete [habit]",
            "example": "remove my running habit",
        },
        {
            "id": "help",
            "action": "Get Help",
            "how": "/help",
            "example": "type /help anytime",
        },
    ],
    "tips": [
        "Be Natural: Say 'I walked 8,000 steps' instead of 'log 8000 for walking'",
        "Use Voice: Tap the microphone to speak your updates",
        "Check Progress: Ask 'how am I doing?' for an overview",
        "Set Targets: Include goals like '5 km daily' when creating habits",
    ],
    "frequencies": [
        {
            "type": "Daily",
            "best_for": "Regular activities",
            "examples": "Exercise, meditation, reading",
        },
        {
            "type": "Weekly",
            "best_for": "Less frequent goals",
            "examples": "Deep cleaning, meal prep",
        },
        {
            "type": "Monthly",
            "best_for": "Long-term tracking",
            "examples": "Budget reviews, goal check-ins",
        },
    ],
}
