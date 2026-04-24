"""Agent tools for habit tracking operations."""


from app.db import queries


async def create_habit(
        user_id: int,
        name: str,
        target: float | None = None,
        unit: str | None = None,
        frequency: str = "daily"
) -> dict:
    """Create a new habit for the user."""
    habit = await queries.create_habit(user_id, name, target, unit, frequency)

    return {"habit": habit, "message": f"Created habit: {name}"}


async def list_habits(user_id: int) -> dict:
    """List all active habits for the user."""
    habits = await queries.get_habits(user_id)
    return {"habits": habits, "count": len(habits)}


async def log_activity(
        user_id: int,
        habit_name: str,
        value: float,
        notes: str | None = None
) -> dict:
    """Log activity for a habit by name."""
    habit = await queries.find_habit_by_name(user_id, habit_name)
    if not habit:
        return {"error": f"No habit found matching '{habit_name}'"}
    
    log = await queries.create_log(habit['id'], user_id, value, notes)
    return {"log": log, "message": f"Logged {value} for {habit['name']}"}


async def get_progress(user_id: int, habit_name: str, days: int = 7) -> dict:
    """Get progress stats for a habit."""
    habit = await queries.find_habit_by_name(user_id, habit_name)
    if not habit:
        return {"error": f"No habit found matching '{habit_name}'"}
    
    return await queries.get_progress(habit['id'], user_id, days)


async def update_habit(
        user_id: int,
        habit_name: str,
        new_name: str | None = None,
        target: str | None = None,
        unit: str | None = None,
        frequency: str | None = None,
) -> dict:
    """Update a habit's settings."""
    habit = await queries.find_habit_by_name(user_id, habit_name)
    if not habit:
        return {"error": f"No habit found matching '{habit_name}'"}
    
    updates = {}
    if new_name:
        updates["name"] = new_name
    if target is not None:
        updates["target"] = target
    if unit:
        updates["unit"] = unit
    if frequency:
        updates["frequency"] = frequency

    if not updates:
        return {"habit": habit, "message": "No changes specified"}
    
    updated = await queries.update_habit(habit["id"], user_id, **updates)
    return {"habit": updated, "message": f"Updated habit: {habit['name']}"}


async def delete_habit(user_id: int, habit_name: str) -> dict:
    """Delete a habit."""
    habit = await queries.find_habit_by_name(user_id, habit_name)
    if not habit:
        return {"habit": habit, "message": f"No habit found matching '{habit_name}'"}
    
    deleted = await queries.delete_habit(habit["id"], user_id)
    return {"habit": deleted, "message": f"Deleted habit: {habit['name']}"}


async def update_log(
    user_id: int, 
    log_id: int, 
    value: float | None = None, 
    notes: str | None = None
) -> dict:
    """Update a log entry."""
    updates = {}
    if value is not None:
        updates["value"] = value
    if notes is not None:
        updates["notes"] = notes
    
    if not updates:
        return {"error": "No updates specified"}
    
    log = await queries.update_log(log_id, user_id, **updates)
    if not log:
        return {"error": "LOg not found or access denied"}

    return {"log": log, "message": "Log updated"}


async def delete_log(user_id: int, log_id: int) -> dict:
    """Delete a log entry."""
    log = await queries.delete_log(log_id, user_id)
    if not log:
        return {"error": "Log not found or access denied"}
    
    return {"log": log, "message": "Log deleted"}