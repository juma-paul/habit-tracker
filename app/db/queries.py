from datetime import datetime, timedelta

from app.db.connection import get_conn

# USERS


async def get_user_by_id(user_id: int) -> dict | None:
    """Get user by ID."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """SELECT id, email, COALESCE(name, '') as name, created_at 
               FROM users 
               WHERE id = %s""",
            (user_id,),
        )

        return await cur.fetchone()


async def get_or_create_user(external_id: str, email: str, name: str = "") -> dict:
    """Get existing user by AuthKit UUID or create on first login."""
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT * FROM users WHERE external_id = %s", (external_id,)
        )
        user = await cur.fetchone()
        if user:
            return dict(user)
        cur = await conn.execute(
            "INSERT INTO users (external_id, email, name) VALUES (%s, %s, %s) RETURNING *",
            (external_id, email, name),
        )
        return dict(await cur.fetchone())


# HABITS


async def create_habit(
    user_id: int,
    name: str,
    target: float | None = None,
    unit: str | None = None,
    frequency: str = "daily",
) -> dict:
    """Create a new habit."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            INSERT INTO  habits (user_id, name, target, unit, frequency)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (user_id, name.lower().strip(), target, unit, frequency),
        )

        return await cur.fetchone()


async def get_habits(user_id: int) -> list[dict]:
    """Get all active habits for a user."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM habits
            WHERE user_id = %s AND is_deleted = FALSE
            ORDER BY created_at DESC
            """,
            (user_id,),
        )

        return await cur.fetchall()


async def get_habit(habit_id: int, user_id: int) -> dict | None:
    """Get a single habit(with ownership check)."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM habits
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            """,
            (habit_id, user_id),
        )

        return await cur.fetchone()


async def update_habit(habit_id: int, user_id: int, **fields) -> dict | None:
    """Update habit fields. Only updates provided fields."""
    if not fields:
        return await get_habit(habit_id, user_id)

    sets = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [habit_id, user_id]

    async with get_conn() as conn:
        cur = await conn.execute(
            f"""
            UPDATE habits SET {sets}
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            RETURNING *
            """,
            values,
        )

        return await cur.fetchone()


async def delete_habit(habit_id: int, user_id: int) -> dict | None:
    """Soft delete a habit."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            UPDATE habits SET is_deleted = TRUE
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            RETURNING *
            """,
            (habit_id, user_id),
        )

        return await cur.fetchone()


async def find_habit_by_name(user_id: int, search: str) -> dict | None:
    """Find habit by name (fuzzy match)."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM habits
            WHERE user_id = %s AND is_deleted = FALSE
            AND name ILIKE %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, f"%{search.lower()}%"),
        )

        return await cur.fetchone()


# HABIT LOGS


async def create_log(
    habit_id: int, user_id: int, value: float, notes: str | None = None
) -> dict:
    """Create a habit log entry (with ownership verification)."""

    habit = await get_habit(habit_id, user_id)

    if not habit:
        raise ValueError("Habit not found or access denied")

    async with get_conn() as conn:
        cur = await conn.execute(
            """
            INSERT INTO habit_logs (habit_id, value, notes)
            VALUES (%s, %s, %s)
            RETURNING *
            """,
            (habit_id, value, notes),
        )

        return await cur.fetchone()


async def get_logs(habit_id: int, user_id: int, days: int = 7) -> list[dict]:
    """Get logs for a habit within date range."""
    habit = await get_habit(habit_id, user_id)

    if not habit:
        raise ValueError("Habit not found or access denied")

    since = datetime.now() - timedelta(days=days)

    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM habit_logs
            WHERE habit_id = %s AND logged_at >= %s
            ORDER BY logged_at DESC
            """,
            (habit_id, since),
        )

        return await cur.fetchall()


async def update_log(log_id: int, user_id: int, **fields) -> dict | None:
    """Update a log entry (with ownership check via habit)"""
    if not fields:
        return None

    sets = ", ".join(f"l.{k} = %s" for k in fields)
    values = list(fields.values()) + [log_id, user_id]

    async with get_conn() as conn:
        cur = await conn.execute(
            f"""
            UPDATE habit_logs l SET {sets}
            FROM habits h
            WHERE l.id = %s AND l.habit_id = h.id AND h.user_id = %s
            RETURNING l.*
            """,
            values,
        )

        return await cur.fetchone()


async def delete_log(log_id: int, user_id: int) -> dict | None:
    """Delete a log entry (with ownership check via habit)."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            DELETE FROM habit_logs l
            USING habits h
            WHERE l.id = %s AND l.habit_id = h.id AND h.user_id = %s
            RETURNING l.*
            """,
            (log_id, user_id),
        )

        return await cur.fetchone()


async def get_today_logs(habit_id: int, user_id: int) -> list[dict]:
    """Get logs for a habit entered today (calendar date, not last 24 h)."""
    habit = await get_habit(habit_id, user_id)
    if not habit:
        raise ValueError("Habit not found or access denied")
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM habit_logs
            WHERE habit_id = %s AND logged_at::date = CURRENT_DATE
            ORDER BY logged_at DESC
            """,
            (habit_id,),
        )
        return await cur.fetchall()


async def get_progress(habit_id: int, user_id: int, days: int = 7) -> dict:
    """Get aggregated progress stats for a habit."""
    habit = await get_habit(habit_id, user_id)
    if not habit:
        raise ValueError("Habit not found or access denied")

    since = datetime.now() - timedelta(days=days)

    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT
                COUNT(*) AS total_logs,
                COALESCE(SUM(value), 0) AS total_value,
                COALESCE(AVG(value), 0) AS avg_value,
                MAX(logged_at) AS last_log

            FROM habit_logs
            WHERE habit_id = %s AND logged_at >= %s
            """,
            (habit_id, since),
        )
        stats = await cur.fetchone()

    completion_rate = None
    if habit["target"]:
        expected = days if habit["frequency"] == "daily" else days // 7

        completion_rate = min(100, (stats["total_logs"] / max(1, expected)) * 100)

    return {
        "habit": habit,
        "days": days,
        "total_logs": stats["total_logs"],
        "total_value": float(stats["total_value"]),
        "avg_value": float(stats["avg_value"]),
        "last_log": stats["last_log"],
        "completion_rate": completion_rate,
    }


# CONVERSATIONS


async def create_conversation(user_id: int, title: str = "New Chat") -> dict:
    """Create a new conversation."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            INSERT INTO conversations (user_id, title)
            VALUES (%s, %s)
            RETURNING *
            """,
            (user_id, title),
        )

        return await cur.fetchone()


async def get_conversations(user_id: int, limit: int = 50) -> list[dict]:
    """Get all conversations for a user, ordered by most recent."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM conversations
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )

        return await cur.fetchall()


async def get_conversation(conversation_id: int, user_id: int) -> dict | None:
    """Get a conversation with ownership check."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM conversations
            WHERE id =%s AND user_id = %s
            """,
            (conversation_id, user_id),
        )

        return await cur.fetchone()


async def update_conversation(
    conversation_id: int, user_id: int, **fields
) -> dict | None:
    """Update conversation fields (e.g., title)."""
    if not fields:
        return await get_conversation(conversation_id, user_id)

    fields["updated_at"] = datetime.now()
    sets = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [conversation_id, user_id]

    async with get_conn() as conn:
        cur = await conn.execute(
            f"""
            UPDATE conversations SET {sets}
            WHERE id = %s AND user_id = %s
            RETURNING *
            """,
            values,
        )

        return await cur.fetchone()


async def delete_conversation(conversation_id: int, user_id: int) -> int | None:
    """Delete a conversation and it's messages."""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            DELETE FROM conversations
            WHERE id = %s AND user_id = %s
            RETURNING *
            """,
            (conversation_id, user_id),
        )

        return await cur.fetchone()


# MESSAGES


async def add_message(conversation_id: int, role: str, content: str) -> dict:
    """Add a message to a conversation and update conversation timestamp.
    If this is the first user message and title is 'New Chat', auto-generate title.
    """

    async with get_conn() as conn:
        # Insert message
        cur = await conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (%s, %s, %s)
            RETURNING *
            """,
            (conversation_id, role, content),
        )

        result = await cur.fetchone()

        # Update conversation updated_at
        await conn.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
            (conversation_id,),
        )

        # Auto-generate title from first user message
        if role == "user":
            # Check if title is still default
            cur = await conn.execute(
                "SELECT title FROM conversations WHERE id = %s", (conversation_id,)
            )
            conv = await cur.fetchone()

            if conv and conv["title"] == "New Chat":
                # Generate title from first 50 chars of message
                title = content[:50].strip()
                if len(content) > 50:
                    title = title.rsplit(" ", 1)[0] + "..."

                await conn.execute(
                    "UPDATE conversations SET title = %s WHERE id = %s",
                    (title, conversation_id),
                )

        return result


async def get_messages(
    conversation_id: int, user_id: int, limit: int = 100
) -> list[dict]:
    """Get messages for a conversation with ownership check."""
    # Verify ownership
    conv = await get_conversation(conversation_id, user_id)
    if not conv:
        return []

    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (conversation_id, limit),
        )

        return await cur.fetchall()


async def get_recent_messages(
    conversation_id: int, user_id: int, limit: int = 10
) -> list[dict]:
    """Get the most recent messages for context (ordered oldest first for chat)."""
    conv = await get_conversation(conversation_id, user_id)
    if not conv:
        return []

    async with get_conn() as conn:
        # Get last N messages. then reverse to chronological order
        cur = await conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (conversation_id, limit),
        )

        messages = await cur.fetchall()

        return list(reversed(messages))


# USER SETTINGS


async def get_user_settings(user_id: int) -> dict:
    """Get user settings, creating defaults if not exists."""
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT * FROM user_settings WHERE user_id = %s", (user_id,)
        )

        result = await cur.fetchone()

        if not result:
            # Create default settings
            cur = await conn.execute(
                """
                INSERT INTO user_settings (user_id)
                VALUES (%s)
                RETURNING *
                """,
                (user_id,),
            )

            result = await cur.fetchone()

        return result


async def update_user_settings(user_id: int, **fields) -> dict:
    """Update user settings."""
    # Ensure settings exist
    await get_user_settings(user_id)

    if not fields:
        return await get_user_settings(user_id)

    fields["updated_at"] = datetime.now()
    sets = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [user_id]

    async with get_conn() as conn:
        cur = await conn.execute(
            f"""
            UPDATE user_settings SET {sets}
            WHERE user_id = %s
            RETURNING *
            """,
            values,
        )

        return await cur.fetchone()
