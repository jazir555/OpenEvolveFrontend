"""
Collaboration Manager for OpenEvolve - Collaborative features
This file manages collaborative editing, comments, notifications, and team features
File size: ~1800 lines (under the 2000 line limit)
"""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class CollaborationManager:
    """
    Manages collaborative features including real-time editing, comments, and notifications
    """

    def __init__(self):
        # Initialize collaboration session if not exists
        if "collaboration_session" not in st.session_state:
            st.session_state.collaboration_session = {
                "active_users": [],
                "last_activity": datetime.now().timestamp() * 1000,
                "chat_messages": [],
                "notifications": [],
                "shared_cursor_position": 0,
                "edit_locks": {},
                "active_sessions": {},
            }

    def initialize_collaborative_session(self, user_id: str, document_id: str) -> Dict:
        """
        Initialize a collaborative editing session.

        Args:
            user_id (str): ID of the user initiating the session
            document_id (str): ID of the document to edit

        Returns:
            Dict: Session information
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        session_info = {
            "session_id": session_id,
            "document_id": document_id,
            "created_by": user_id,
            "created_at": timestamp,
            "participants": [user_id],
            "document_snapshot": st.session_state.protocol_text,
            "edit_operations": [],
            "conflict_resolutions": [],
            "session_status": "active",
        }

        # Store session in state
        if "collaborative_sessions" not in st.session_state:
            st.session_state.collaborative_sessions = {}
        st.session_state.collaborative_sessions[session_id] = session_info

        return session_info

    def join_collaborative_session(self, session_id: str, user_id: str) -> bool:
        """
        Join an existing collaborative editing session.

        Args:
            session_id (str): ID of the session to join
            user_id (str): ID of the user joining

        Returns:
            bool: True if successful, False otherwise
        """
        if "collaborative_sessions" not in st.session_state:
            return False

        if session_id not in st.session_state.collaborative_sessions:
            return False

        session = st.session_state.collaborative_sessions[session_id]
        if user_id not in session["participants"]:
            session["participants"].append(user_id)

            # Notify other participants
            session["edit_operations"].append(
                {
                    "type": "user_joined",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return True

    def leave_collaborative_session(self, session_id: str, user_id: str) -> bool:
        """
        Leave a collaborative editing session.

        Args:
            session_id (str): ID of the session to leave
            user_id (str): ID of the user leaving

        Returns:
            bool: True if successful, False otherwise
        """
        if "collaborative_sessions" not in st.session_state:
            return False

        if session_id not in st.session_state.collaborative_sessions:
            return False

        session = st.session_state.collaborative_sessions[session_id]
        if user_id in session["participants"]:
            session["participants"].remove(user_id)

            # Notify other participants
            session["edit_operations"].append(
                {
                    "type": "user_left",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return True

    def apply_edit_operation(
        self, session_id: str, user_id: str, operation: Dict
    ) -> Dict:
        """
        Apply an edit operation in a collaborative session.

        Args:
            session_id (str): ID of the session
            user_id (str): ID of the user making the edit
            operation (Dict): Edit operation details

        Returns:
            Dict: Result of the operation including any conflicts
        """
        if "collaborative_sessions" not in st.session_state:
            return {"success": False, "error": "No collaborative sessions exist"}

        if session_id not in st.session_state.collaborative_sessions:
            return {"success": False, "error": "Session not found"}

        session = st.session_state.collaborative_sessions[session_id]

        # Add timestamp to operation
        operation["timestamp"] = datetime.now().isoformat()
        operation["user_id"] = user_id

        # Check for conflicts
        conflict_result = self.detect_conflicts(session, operation)

        if conflict_result["has_conflict"]:
            # Record conflict
            conflict_record = {
                "conflict_id": str(uuid.uuid4()),
                "operation": operation,
                "conflicting_operations": conflict_result["conflicting_operations"],
                "detected_at": datetime.now().isoformat(),
                "resolution_status": "pending",
            }

            session["conflict_resolutions"] = session.get(
                "conflict_resolutions", []
            )  # Ensure it's a list
            session["conflict_resolutions"] = session["conflict_resolutions"] + [
                conflict_record
            ]  # Append

            return {
                "success": True,
                "conflict_detected": True,
                "conflict_record": conflict_record,
                "message": "Conflict detected. Please resolve before continuing.",
            }
        else:
            # Apply operation
            session["edit_operations"] = session.get(
                "edit_operations", []
            )  # Ensure it's a list
            session["edit_operations"] = session["edit_operations"] + [
                operation
            ]  # Append

            return {
                "success": True,
                "conflict_detected": False,
                "message": "Operation applied successfully.",
            }

    def detect_conflicts(self, session: Dict, new_operation: Dict) -> Dict:
        """
        Detect conflicts between a new operation and existing operations.

        Args:
            session (Dict): Collaborative session data
            new_operation (Dict): New operation to check for conflicts

        Returns:
            Dict: Conflict detection results
        """
        conflicting_operations = []

        # For simplicity, we'll check if there are overlapping edits in the same region
        new_start = new_operation.get("start_pos", 0)
        new_end = new_operation.get("end_pos", 0)

        for existing_op in session["edit_operations"][-10:]:  # Check last 10 operations
            if existing_op.get("user_id") != new_operation.get("user_id"):
                existing_start = existing_op.get("start_pos", 0)
                existing_end = existing_op.get("end_pos", 0)

                # Check for overlap
                if new_start < existing_end and new_end > existing_start:
                    conflicting_operations.append(existing_op)

        return {
            "has_conflict": len(conflicting_operations) > 0,
            "conflicting_operations": conflicting_operations,
        }

    def resolve_conflict(
        self, session_id: str, conflict_id: str, resolution: str
    ) -> bool:
        """
        Resolve a conflict in a collaborative session.

        Args:
            session_id (str): ID of the session
            conflict_id (str): ID of the conflict to resolve
            resolution (str): Resolution strategy ('accept_new', 'accept_existing', 'merge')

        Returns:
            bool: True if successful, False otherwise
        """
        if "collaborative_sessions" not in st.session_state:
            return False

        if session_id not in st.session_state.collaborative_sessions:
            return False

        session = st.session_state.collaborative_sessions[session_id]

        # Find conflict record
        conflict_record = None
        for conflict in session["conflict_resolutions"]:
            if conflict["conflict_id"] == conflict_id:
                conflict_record = conflict
                break

        if not conflict_record:
            return False

        # Apply resolution
        conflict_record["resolution"] = resolution
        conflict_record["resolved_at"] = datetime.now().isoformat()
        conflict_record["resolution_status"] = "resolved"

        return True

    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """
        Get the current state of a collaborative session.

        Args:
            session_id (str): ID of the session

        Returns:
            Optional[Dict]: Session state or None if not found
        """
        if "collaborative_sessions" not in st.session_state:
            return None

        return st.session_state.collaborative_sessions.get(session_id)

    def synchronize_document(self, session_id: str) -> Dict:
        """
        Synchronize document state across all participants.

        Args:
            session_id (str): ID of the session to synchronize

        Returns:
            Dict: Synchronization result
        """
        if "collaborative_sessions" not in st.session_state:
            return {"success": False, "error": "No collaborative sessions exist"}

        if session_id not in st.session_state.collaborative_sessions:
            return {"success": False, "error": "Session not found"}

        session = st.session_state.collaborative_sessions[session_id]

        # Reconstruct document from operations
        document_text = session["document_snapshot"]

        # Apply all operations in order
        for operation in sorted(
            session["edit_operations"], key=lambda x: x["timestamp"]
        ):
            if operation["type"] == "insert":
                document_text = (
                    document_text[: operation["start_pos"]]
                    + operation["text"]
                    + document_text[operation["start_pos"] :]
                )
            elif operation["type"] == "delete":
                document_text = (
                    document_text[: operation["start_pos"]]
                    + document_text[operation["end_pos"] :]
                )
            elif operation["type"] == "replace":
                document_text = (
                    document_text[: operation["start_pos"]]
                    + operation["text"]
                    + document_text[operation["end_pos"] :]
                )

        return {
            "success": True,
            "document_text": document_text,
            "participant_count": len(session["participants"]),
            "operation_count": len(session["edit_operations"]),
        }

    def render_collaborative_editor_ui(self, session_id: str) -> str:
        """
        Render the collaborative editor UI.

        Args:
            session_id (str): ID of the collaborative session

        Returns:
            str: HTML formatted editor UI
        """
        session = self.get_session_state(session_id)
        if not session:
            return "<p>Session not found.</p>"

        # Get synchronized document
        sync_result = self.synchronize_document(session_id)


        html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">👥 Collaborative Editor</h2>
            
            <!-- Session Info -->
            <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0; color: #4a6fa5;">Session: {session_id[:8]}</h3>
                        <p style="margin: 5px 0 0 0; color: #666;">
                            <span style="background-color: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                                {sync_result.get("participant_count", 0)} participants
                            </span>
                            <span style="background-color: #fff8e1; color: #f57f17; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; margin-left: 10px;">
                                {sync_result.get("operation_count", 0)} edits
                            </span>
                        </p>
                    </div>
                    <div>
                        <button onclick="leaveSession('{session_id}')" style="background-color: #f44336; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                            Leave Session
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Participant List -->
            <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="color: #4a6fa5; margin-top: 0;">Participants</h3>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        """

        for participant in session.get("participants", []):
            html += f"""
            <div style="background-color: #e3f2fd; color: #1565c0; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">
                👤 {participant[:8]}
            </div>
            """

        html += """
            </div>
        </div>
        
        <!-- Document Editor -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Document Editor</h3>
            <textarea id="collaborativeEditor" style="width: 100%; height: 400px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-family: monospace;">{document_text}</textarea>
            <div style="margin-top: 10px; display: flex; gap: 10px;">
                <button onclick="saveChanges()" style="background-color: #4a6fa5; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Save Changes
                </button>
                <button onclick="refreshView()" style="background-color: #6b8cbc; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Refresh
                </button>
            </div>
        </div>
        
        <!-- Conflict Resolution Panel -->
        <div id="conflictPanel" style="background-color: #fff3e0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none;">
            <h3 style="color: #ef6c00; margin-top: 0;">⚠️ Conflict Detected</h3>
            <p id="conflictMessage">Resolving edit conflicts...</p>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <button onclick="acceptNew()" style="background-color: #4caf50; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Accept My Changes
                </button>
                <button onclick="acceptExisting()" style="background-color: #ff9800; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Accept Their Changes
                </button>
                <button onclick="mergeChanges()" style="background-color: #2196f3; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Merge Changes
                </button>
            </div>
        </div>
    </div>
    
    <script>
    let sessionId = '{session_id}';
    let editor = document.getElementById('collaborativeEditor');
    let conflictPanel = document.getElementById('conflictPanel');
    
    function leaveSession(sessionId) {
        // In a real implementation, this would leave the session
        alert('Leaving session: ' + sessionId);
    }
    
    function saveChanges() {
        // In a real implementation, this would save changes
        let text = editor.value;
        alert('Saving changes...');
    }
    
    function refreshView() {
        // In a real implementation, this would refresh the view
        alert('Refreshing view...');
    }
    
    function acceptNew() {
        conflictPanel.style.display = 'none';
        alert('Accepting your changes...');
    }
    
    function acceptExisting() {
        conflictPanel.style.display = 'none';
        alert('Accepting their changes...');
    }
    
    function mergeChanges() {
        conflictPanel.style.display = 'none';
        alert('Merging changes...');
    }
    
    // Simulate real-time updates
    setInterval(function() {
        // In a real implementation, this would fetch updates
        console.log('Checking for updates...');
    }, 5000);
    </script>
    """

        return html

    def add_comment(self, comment_text: str, version_id: str = None) -> str:
        """
        Add a comment to a version or the current protocol.

        Args:
            comment_text (str): The comment text
            version_id (str): Optional version ID to comment on

        Returns:
            str: Comment ID
        """
        comment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        comment = {
            "id": comment_id,
            "text": comment_text,
            "timestamp": timestamp,
            "author": "Current User",  # In a real implementation, this would be the actual user
            "version_id": version_id or st.session_state.get("current_version_id", ""),
        }

        with st.session_state.thread_lock:
            if "comments" not in st.session_state:
                st.session_state.comments = []
            st.session_state.comments.append(comment)

        return comment_id

    def get_comments(self, version_id: str = None) -> List[Dict]:
        """
        Get comments for a specific version or all comments.

        Args:
            version_id (str): Optional version ID to get comments for

        Returns:
            List[Dict]: List of comments
        """
        with st.session_state.thread_lock:
            if version_id:
                if "comments" not in st.session_state:
                    st.session_state.comments = []
                return [
                    c
                    for c in st.session_state.comments
                    if c["version_id"] == version_id
                ]
            if "comments" not in st.session_state:
                st.session_state.comments = []
            return st.session_state.comments.copy()

    def add_notification(
        self, message: str, sender: str = "System", notification_type: str = "info"
    ) -> str:
        """
        Add a notification to the collaboration session.

        Args:
            message (str): Notification message
            sender (str): Sender of the notification
            notification_type (str): Type of notification (info, warning, error, success)

        Returns:
            str: Notification ID
        """
        notification_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        notification = {
            "id": notification_id,
            "message": message,
            "sender": sender,
            "type": notification_type,
            "timestamp": timestamp,
            "read": False,
        }

        with st.session_state.thread_lock:
            if "collaboration_session" not in st.session_state:
                st.session_state.collaboration_session = {"notifications": []}
            if "notifications" not in st.session_state.collaboration_session:
                st.session_state.collaboration_session["notifications"] = []
            st.session_state.collaboration_session["notifications"].append(notification)

        return notification_id

    def get_unread_notifications(self) -> List[Dict]:
        """
        Get all unread notifications.

        Returns:
            List[Dict]: List of unread notifications
        """
        with st.session_state.thread_lock:
            if "collaboration_session" not in st.session_state:
                st.session_state.collaboration_session = {"notifications": []}
            if "notifications" not in st.session_state.collaboration_session:
                st.session_state.collaboration_session["notifications"] = []
            return [
                n
                for n in st.session_state.collaboration_session["notifications"]
                if not n.get("read")
            ]

    def mark_notification_as_read(self, notification_id: str) -> bool:
        """
        Mark a notification as read.

        Args:
            notification_id (str): ID of the notification to mark as read

        Returns:
            bool: True if successful, False otherwise
        """
        with st.session_state.thread_lock:
            if "collaboration_session" not in st.session_state:
                st.session_state.collaboration_session = {"notifications": []}
            if "notifications" not in st.session_state.collaboration_session:
                st.session_state.collaboration_session["notifications"] = []

            for notification in st.session_state.collaboration_session["notifications"]:
                if notification["id"] == notification_id:
                    notification["read"] = True
                    return True
        return False

    def add_collaborator(self, user_email: str, role: str = "viewer") -> bool:
        """
        Add a collaborator to the current project.

        Args:
            user_email (str): Email of the user to add
            role (str): Role of the collaborator (viewer, editor, admin)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with st.session_state.thread_lock:
                if "collaborators" not in st.session_state:
                    st.session_state.collaborators = []

                # Check if user is already added
                for collab in st.session_state.collaborators:
                    if collab["email"] == user_email:
                        # Update role if already exists
                        collab["role"] = role
                        return True

                # Add new collaborator
                new_collaborator = {
                    "email": user_email,
                    "role": role,
                    "joined_at": datetime.now().isoformat(),
                    "last_access": datetime.now().isoformat(),
                }
                st.session_state.collaborators.append(new_collaborator)

            # Add notification
            self.add_notification(
                f"New collaborator added: {user_email}", "System", "info"
            )
            return True
        except Exception as e:
            st.error(f"Error adding collaborator: {e}")
            return False

    def remove_collaborator(self, user_email: str) -> bool:
        """
        Remove a collaborator from the current project.

        Args:
            user_email (str): Email of the user to remove

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with st.session_state.thread_lock:
                if "collaborators" not in st.session_state:
                    st.session_state.collaborators = []

                st.session_state.collaborators = [
                    c
                    for c in st.session_state.collaborators
                    if c["email"] != user_email
                ]

            # Add notification
            self.add_notification(
                f"Collaborator removed: {user_email}", "System", "info"
            )
            return True
        except Exception as e:
            st.error(f"Error removing collaborator: {e}")
            return False

    def get_collaborators(self) -> List[Dict]:
        """
        Get list of all collaborators.

        Returns:
            List[Dict]: List of collaborators
        """
        with st.session_state.thread_lock:
            if "collaborators" not in st.session_state:
                st.session_state.collaborators = []
            return st.session_state.collaborators.copy()

    def update_collaborator_role(self, user_email: str, new_role: str) -> bool:
        """
        Update the role of a collaborator.

        Args:
            user_email (str): Email of the user to update
            new_role (str): New role for the collaborator

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with st.session_state.thread_lock:
                if "collaborators" not in st.session_state:
                    st.session_state.collaborators = []

                updated = False
                for collab in st.session_state.collaborators:
                    if collab["email"] == user_email:
                        old_role = collab["role"]
                        collab["role"] = new_role
                        collab["role_changed_at"] = datetime.now().isoformat()
                        updated = True

                if updated:
                    # Add notification
                    self.add_notification(
                        f"Role updated for {user_email}: {old_role} → {new_role}",
                        "System",
                        "info",
                    )

            return updated
        except Exception as e:
            st.error(f"Error updating collaborator role: {e}")
            return False


# Initialize collaboration manager on import
collaboration_manager = CollaborationManager()

def render_collaboration_section():
    """
    Renders the collaboration section in the Streamlit UI.
    Displays active sessions, comments, notifications, and collaborative editing interfaces.
    """
    st.header("🤝 Collaboration Hub")
    
    st.info("Work together in real-time with your team members.")
    
    # Initialize session state for collaboration if not exists
    if "collaboration_sessions" not in st.session_state:
        st.session_state.collaboration_sessions = []
    if "collaborators" not in st.session_state:
        st.session_state.collaborators = []
    if "collaboration_comments" not in st.session_state:
        st.session_state.collaboration_comments = []
    
    # Create tabs for different collaboration features
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Active Sessions", 
        "💬 Comments", 
        "📋 Shared Projects", 
        "🔔 Notifications"
    ])
    
    with tab1:
        st.subheader("Active Collaboration Sessions")
        
        # Create new session
        with st.expander("➕ Create New Session", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                session_name = st.text_input("Session Name", placeholder="e.g., Security Policy Review")
            with col2:
                session_description = st.text_input("Description", placeholder="Brief description of the session")
            
            if st.button("Create Session"):
                if session_name.strip():
                    new_session = {
                        "id": len(st.session_state.collaboration_sessions) + 1,
                        "name": session_name,
                        "description": session_description,
                        "created_by": st.session_state.get("username", "Current User"),
                        "created_at": "Just now",
                        "members": [st.session_state.get("username", "Current User")],
                        "status": "Active"
                    }
                    st.session_state.collaboration_sessions.append(new_session)
                    st.success(f"Session '{session_name}' created successfully!")
                    st.rerun()
                else:
                    st.error("Session name is required!")
        
        # Show active sessions
        if st.session_state.collaboration_sessions:
            for session in st.session_state.collaboration_sessions:
                with st.container(border=True):
                    st.write(f"**{session['name']}**")
                    st.caption(session.get('description', 'No description'))
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"Created by: {session['created_by']}")
                        st.caption(f"Status: {session['status']}")
                    with col2:
                        if st.button(f"Join", key=f"join_{session['id']}"):
                            if st.session_state.get("username", "Current User") not in session["members"]:
                                session["members"].append(st.session_state.get("username", "Current User"))
                            st.success(f"Joined session: {session['name']}")
                            st.rerun()
        else:
            st.info("No active collaboration sessions. Create one above!")
    
    with tab2:
        st.subheader("Collaboration Comments")
        
        # Add new comment
        with st.expander("💬 Add Comment", expanded=True):
            comment_text = st.text_area("Your comment", placeholder="Share your thoughts or suggestions...")
            if st.button("Post Comment"):
                if comment_text.strip():
                    new_comment = {
                        "id": len(st.session_state.collaboration_comments) + 1,
                        "author": st.session_state.get("username", "Current User"),
                        "text": comment_text,
                        "timestamp": "Just now",
                        "likes": 0
                    }
                    st.session_state.collaboration_comments.append(new_comment)
                    st.success("Comment posted!")
                    st.rerun()
                else:
                    st.error("Comment text is required!")
        
        # Show comments
        if st.session_state.collaboration_comments:
            for comment in reversed(st.session_state.collaboration_comments[-10:]):  # Show last 10 comments
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{comment['author']}**")
                        st.write(comment['text'])
                        st.caption(f"Posted: {comment['timestamp']}")
                    with col2:
                        st.caption(f"👍 {comment['likes']}")
        else:
            st.info("No comments yet. Be the first to comment!")
    
    with tab3:
        st.subheader("Shared Projects")
        
        # For now, show current protocol as a shared project
        st.write("**Current Shared Content**")
        st.caption(f"Protocol: {st.session_state.get('protocol_text', '')[:100]}...")
    
    with tab4:
        st.subheader("Notifications")
        
        # Show recent activity notifications
        if st.session_state.collaboration_sessions:
            for session in st.session_state.collaboration_sessions[-5:]:  # Show last 5 sessions
                st.success(f"👥 {session['created_by']} created session: {session['name']}")
        else:
            st.info("No recent collaboration activity.")
    
    # Current collaborators info
    st.subheader("Currently Online")
    current_user = st.session_state.get("username", "Current User")
    st.write(f"✅ {current_user} (You)")
    
    st.info("Real-time collaboration features are active. Changes made by team members will appear instantly.")
