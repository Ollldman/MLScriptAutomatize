import logging
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional

logger = logging.getLogger(__name__)


def send_report_via_email(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    recipient_emails: List[str],
    subject: str,
    body: str,
    attachment_paths: Optional[List[str]] = None,
    use_tls: bool = True,
) -> bool:
    """
    Send an email with report attachments (HTML, PDF, images, etc.) via SMTP.

    Args:
        smtp_server (str): SMTP server address (e.g., 'smtp.gmail.com').
        smtp_port (int): SMTP server port (e.g., 587 for TLS).
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password or app-specific password.
        recipient_emails (List[str]): List of recipient email addresses.
        subject (str): Email subject line.
        body (str): Plain text or HTML email body.
        attachment_paths (List[str], optional): List of file paths to attach.
        use_tls (bool): Whether to use TLS encryption. Default is True.

    Returns:
        bool: True if email was sent successfully, False otherwise.

    Raises:
        ValueError: If any required argument is missing or invalid.
        smtplib.SMTPException: If there is an issue with the SMTP server.
    """
    if not all([smtp_server, sender_email, sender_password, recipient_emails]):
        raise ValueError("SMTP server, sender credentials, and recipients must be provided.")

    if not all(os.path.exists(path) for path in (attachment_paths or [])):
        missing = [path for path in (attachment_paths or []) if not os.path.exists(path)]
        raise ValueError(f"Attachment files not found: {missing}")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_emails)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))  # Use 'plain' if body is plain text

    # Attach files
    for file_path in attachment_paths or []:
        with open(file_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(file_path)}'
            )
            msg.attach(part)

    try:
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port)

        if use_tls:
            server.starttls()
            logger.info("TLS encryption enabled.")

        server.login(sender_email, sender_password)
        logger.info("Login successful.")

        text = msg.as_string()
        server.sendmail(sender_email, recipient_emails, text)
        server.quit()

        logger.info(f"Email sent successfully to: {recipient_emails}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication failed. Check email and password.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while sending email: {e}")
        return False