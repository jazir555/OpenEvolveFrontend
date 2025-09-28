import streamlit as st
import uuid
from datetime import datetime
from typing import List, Dict, Optional

if "tasks" not in st.session_state:
    st.session_state.tasks = []


def create_task(
    title: str,
    description: str,
    assignee: Optional[str] = None,
    due_date: Optional[datetime] = None,
) -> str:
    """
    Create a new task.
    """
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "title": title,
        "description": description,
        "assignee": assignee,
        "status": "To Do",
        "due_date": due_date.isoformat() if due_date else None,
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.tasks.append(task)
    return task_id


def get_tasks(
    assignee: Optional[str] = None, status: Optional[str] = None
) -> List[Dict]:
    """
    Get a list of tasks.
    """
    tasks = st.session_state.tasks
    if assignee:
        tasks = [t for t in tasks if t["assignee"] == assignee]
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    return tasks


def update_task(
    task_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    due_date: Optional[datetime] = None,
):
    """
    Update a task.
    """
    for task in st.session_state.tasks:
        if task["id"] == task_id:
            if title:
                task["title"] = title
            if description:
                task["description"] = description
            if assignee:
                task["assignee"] = assignee
            if status:
                task["status"] = status
            if due_date:
                task["due_date"] = due_date.isoformat()
            break

def render_tasks():
    """
    Placeholder function to render the tasks section in the Streamlit UI.
    This would typically display a list of tasks, allow creation, and updates.
    """
    st.header("✅ Tasks")
    st.info("Task management features are under development. Stay tuned!")
    # Example of how you might use the functions:
    # st.subheader("My Tasks")
    # for task in get_tasks(assignee="Current User"): # Assuming a way to get current user
    #     st.write(f"- [{task['status']}] {task['title']} (Due: {task['due_date']})")
    #
    # if st.button("Create New Task"):
    #     # Logic to open a form for new task creation
    #     pass
