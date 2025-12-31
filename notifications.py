import asyncio
import streamlit as st # Import streamlit to access session_state


class NotificationManager:
    def __init__(self, smtp_config=None):
        self.smtp_config = smtp_config

    def send_in_app_notification(self, recipient: str, sender: str, message: str):
        payload = {"recipient": recipient, "sender": sender, "message": message}
        loop = asyncio.get_running_loop()
        
        # Access collaboration_server_instance from session_state
        if "collaboration_server_instance" in st.session_state:
            asyncio.run_coroutine_threadsafe(
                st.session_state.collaboration_server_instance.broadcast_notification(payload), loop
            )
        else:
            print("Collaboration server instance not found in session state.")


notification_manager = NotificationManager()


def send_notification(
    recipient: str,
    sender: str,
    message: str,
    notification_type: str = "in-app",
    email_subject: str = None,
    email_body: str = None,
):
    """
    Send a notification to a specific user.
    """
    if notification_type == "in-app":
        notification_manager.send_in_app_notification(recipient, sender, message)
    elif notification_type == "email":
        notification_manager.send_email_notification(
            recipient, email_subject, email_body
        )

def render_notifications():
    """
    Renders notifications in the Streamlit UI.
    Displays a list of recent notifications to the user.
    """
    import streamlit as st
    st.header("🔔 Notifications Center")
    
    # Initialize notifications in session state if not exists
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    
    # Notification sending section
    with st.expander("📤 Send Notification", expanded=True):
        st.subheader("Send New Notification")
        col1, col2 = st.columns(2)
        with col1:
            recipient = st.text_input("Recipient", placeholder="e.g., username or email")
        with col2:
            notification_type = st.selectbox("Type", ["info", "warning", "error", "success"])
        
        message = st.text_area("Message", placeholder="Enter your notification message...")
        sender = st.session_state.get("username", "Current User")  # Could use actual username if available
        
        if st.button("Send Notification"):
            if recipient and message:
                # Create notification object
                notification = {
                    "id": len(st.session_state.notifications) + 1,
                    "sender": sender,
                    "recipient": recipient,
                    "message": message,
                    "type": notification_type,
                    "timestamp": st.session_state.get("last_activity_time", "Just now"),
                    "read": False,
                    "created_at": st.session_state.get("last_activity_time", "Just now")
                }
                
                # Add to notifications
                st.session_state.notifications.append(notification)
                
                # Also send via the notification manager
                send_notification(recipient, sender, message, notification_type)
                
                st.success(f"Notification sent to {recipient}!")
                st.rerun()
            else:
                st.error("Please enter both recipient and message.")
    
    # Notification filtering
    st.subheader("My Notifications")
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_type = st.selectbox("Filter by Type", ["All", "info", "warning", "error", "success"])
    with col2:
        read_filter = st.selectbox("Filter by Read Status", ["All", "Unread", "Read"])
    
    # Get notifications for current user if available
    current_user = st.session_state.get("username", "Current User")
    user_notifications = [n for n in st.session_state.notifications if n.get("recipient", "").lower() == current_user.lower() or n.get("recipient", "").lower() == "all"]
    
    # Apply filters
    if filter_type != "All":
        user_notifications = [n for n in user_notifications if n["type"] == filter_type.lower()]
    if read_filter == "Unread":
        user_notifications = [n for n in user_notifications if not n.get("read", False)]
    elif read_filter == "Read":
        user_notifications = [n for n in user_notifications if n.get("read", False)]
    
    # Display notifications
    if user_notifications:
        # Mark all as read button
        if st.button("Mark All as Read"):
            for notif in user_notifications:
                notif["read"] = True
            st.rerun()
        
        # Show notifications in reverse chronological order (newest first)
        for notification in reversed(user_notifications):
            # Determine icon and color based on type
            type_icons = {
                "info": "ℹ️",
                "warning": "⚠️", 
                "error": "❌",
                "success": "✅"
            }
            type_colors = {
                "info": "#1976d2",
                "warning": "#f57c00", 
                "error": "#d32f2f",
                "success": "#388e3c"
            }
            
            icon = type_icons.get(notification["type"], "🔔")
            color = type_colors.get(notification["type"], "#666666")
            
            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"<div style='font-size: 2em;'>{icon}</div>", unsafe_allow_html=True)
                    if not notification.get("read", False):
                        st.markdown(f"<div style='color: {color}; font-weight: bold; font-size: 0.8em;'>UNREAD</div>", 
                                  unsafe_allow_html=True)
                with col2:
                    st.write(f"**From:** {notification.get('sender', 'System')}")
                    st.write(f"**Message:** {notification['message']}")
                    st.caption(f"Sent at: {notification.get('created_at', 'Unknown time')}")
                    
                    # Mark as read/unread button
                    if notification.get("read", False):
                        if st.button("Mark as Unread", key=f"unread_{notification['id']}"):
                            notification["read"] = False
                            st.rerun()
                    else:
                        if st.button("Mark as Read", key=f"read_{notification['id']}"):
                            notification["read"] = True
                            st.rerun()
    else:
        st.info("No notifications found. Send yourself a notification above!")
    
    # Notification statistics
    all_user_notifs = [n for n in st.session_state.notifications if n.get("recipient", "").lower() == current_user.lower() or n.get("recipient", "").lower() == "all"]
    unread_count = len([n for n in all_user_notifs if not n.get("read", False)])
    total_count = len(all_user_notifs)
    
    st.subheader("📊 Notification Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Notifications", total_count)
    with col2:
        st.metric("Unread", unread_count)
    
    st.info(f"You have {unread_count} unread notification{'s' if unread_count != 1 else ''}.")
