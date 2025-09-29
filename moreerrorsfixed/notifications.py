import asyncio
import smtplib
from email.mime.text import MIMEText
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
    Placeholder function to render notifications in the Streamlit UI.
    This would typically display a list of recent notifications to the user.
    """
    import streamlit as st # Import streamlit here as it's a UI function
    st.header("🔔 Notifications")
    st.info("Notification display features are under development. Stay tuned!")
    # Example of how you might display notifications:
    # if "collaboration_session" in st.session_state and "notifications" in st.session_state.collaboration_session:
    #     st.subheader("Recent Notifications")
    #     for notif in st.session_state.collaboration_session["notifications"]:
    #         st.write(f"- [{notif['type']}] {notif['message']} (from {notif['sender']})")
