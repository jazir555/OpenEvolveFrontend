import asyncio
from collaboration import collaboration_server
import json
import smtplib
from email.mime.text import MIMEText

class NotificationManager:
    def __init__(self, smtp_config=None):
        self.smtp_config = smtp_config

    def send_in_app_notification(self, recipient: str, sender: str, message: str):
        payload = {
            "recipient": recipient,
            "sender": sender,
            "message": message
        }
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(collaboration_server.broadcast_notification(payload), loop)

    def send_email_notification(self, recipient_email: str, subject: str, body: str):
        if not self.smtp_config:
            return

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.smtp_config['sender_email']
        msg['To'] = recipient_email

        with smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port']) as server:
            server.starttls()
            server.login(self.smtp_config['smtp_user'], self.smtp_config['smtp_password'])
            server.send_message(msg)

notification_manager = NotificationManager()

def send_notification(recipient: str, sender: str, message: str, notification_type: str = 'in-app', email_subject: str = None, email_body: str = None):
    """
    Send a notification to a specific user.
    """
    if notification_type == 'in-app':
        notification_manager.send_in_app_notification(recipient, sender, message)
    elif notification_type == 'email':
        notification_manager.send_email_notification(recipient, email_subject, email_body)