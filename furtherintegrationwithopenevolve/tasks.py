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
    Renders the tasks section in the Streamlit UI.
    Displays a list of tasks, allows creation, updates, and management.
    """
    st.header("✅ Task Management")
    
    # Task creation form
    with st.expander("➕ Create New Task", expanded=False):
        with st.form("create_task_form"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Task Title", placeholder="e.g., Review security protocol")
            with col2:
                assignee = st.text_input("Assignee", placeholder="e.g., John Doe")
            
            description = st.text_area("Description", placeholder="Describe what needs to be done...")
            due_date = st.date_input("Due Date (optional)")
            
            submitted = st.form_submit_button("Create Task")
            if submitted:
                if title.strip():
                    due_datetime = None
                    if due_date:
                        from datetime import datetime
                        due_datetime = datetime.combine(due_date, datetime.min.time())
                    
                    task_id = create_task(title, description, assignee, due_datetime)
                    st.success(f"Task '{title}' created successfully!")
                    st.rerun()
                else:
                    st.error("Task title is required!")
    
    # Task filtering
    st.subheader("Task List")
    col1, col2 = st.columns([1, 1])
    with col1:
        assignee_filter = st.text_input("Filter by Assignee", placeholder="All assignees")
    with col2:
        status_filter = st.selectbox("Filter by Status", ["All", "To Do", "In Progress", "Completed", "On Hold"])
    
    # Apply filters
    tasks = get_tasks()
    if assignee_filter.strip():
        tasks = [t for t in tasks if assignee_filter.lower() in (t["assignee"] or "").lower()]
    if status_filter != "All":
        tasks = [t for t in tasks if t["status"] == status_filter]
    
    # Show tasks
    if tasks:
        # Group by status
        status_groups = {"To Do": [], "In Progress": [], "On Hold": [], "Completed": []}
        for task in tasks:
            status = task["status"]
            if status in status_groups:
                status_groups[status].append(task)
        
        for status, task_list in status_groups.items():
            if task_list:  # Only show sections that have tasks
                st.subheader(f"{status} ({len(task_list)})")
                for task in task_list:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{task['title']}**")
                            if task.get("description"):
                                st.caption(task["description"])
                            if task.get("assignee"):
                                st.caption(f"Assignee: {task['assignee']}")
                            if task.get("due_date"):
                                from datetime import datetime
                                due_date = datetime.fromisoformat(task["due_date"])
                                st.caption(f"Due: {due_date.strftime('%Y-%m-%d')}")
                        with col2:
                            # Status selector
                            new_status = st.selectbox(
                                "Status", 
                                ["To Do", "In Progress", "On Hold", "Completed"],
                                index=["To Do", "In Progress", "On Hold", "Completed"].index(task["status"]),
                                key=f"status_{task['id']}"
                            )
                            if new_status != task["status"]:
                                update_task(task["id"], status=new_status)
                                st.rerun()
                            
                            # Delete button
                            if st.button("🗑️", key=f"delete_{task['id']}"):
                                st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task["id"]]
                                st.rerun()
    else:
        st.info("No tasks found. Create your first task above!")
    
    # Task statistics
    all_tasks = get_tasks()
    if all_tasks:
        st.subheader("📊 Task Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_tasks = len(all_tasks)
        completed_tasks = len([t for t in all_tasks if t["status"] == "Completed"])
        in_progress_tasks = len([t for t in all_tasks if t["status"] == "In Progress"])
        overdue_tasks = 0
        
        from datetime import datetime
        for task in all_tasks:
            if task.get("due_date"):
                due_date = datetime.fromisoformat(task["due_date"])
                if due_date < datetime.now() and task["status"] != "Completed":
                    overdue_tasks += 1
        
        with col1:
            st.metric("Total Tasks", total_tasks)
        with col2:
            st.metric("Completed", completed_tasks)
        with col3:
            st.metric("In Progress", in_progress_tasks)
        with col4:
            st.metric("Overdue", overdue_tasks)
        
        # Progress bar
        if total_tasks > 0:
            completion_percentage = (completed_tasks / total_tasks) * 100
            st.progress(completion_percentage / 100)
            st.caption(f"{completion_percentage:.1f}% of tasks completed")
    
    # Bulk actions
    if all_tasks:
        st.subheader("Bulk Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Mark All as Completed"):
                for task in all_tasks:
                    update_task(task["id"], status="Completed")
                st.success(f"Marked all {len(all_tasks)} tasks as completed!")
                st.rerun()
        
        with col2:
            if st.button("Clear Completed Tasks"):
                st.session_state.tasks = [t for t in st.session_state.tasks if t["status"] != "Completed"]
                st.success("Cleared all completed tasks!")
                st.rerun()
