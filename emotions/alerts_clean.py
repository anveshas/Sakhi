import os
import logging

# Optional Twilio integration. If environment variables are not set, send_alert will just log.
try:
    from twilio.rest import Client
    _TWILIO_AVAILABLE = True
except Exception:
    _TWILIO_AVAILABLE = False

TWILIO_SID = os.environ.get('TWILIO_SID')
TWILIO_TOKEN = os.environ.get('TWILIO_TOKEN')
TWILIO_FROM = os.environ.get('TWILIO_FROM')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alerts')


def send_alert(message, contacts=None, lat=None, lon=None):
    """Send alert to configured contacts. Returns True if at least one message was attempted.
    - `contacts` can be a list of phone numbers; if omitted the function will look for `EMERGENCY_CONTACTS` env var (comma-separated).
    - If Twilio creds are missing, the function logs the message and returns False.
    """
    if contacts is None:
        raw = os.environ.get('EMERGENCY_CONTACTS', '')
        contacts = [c.strip() for c in raw.split(',') if c.strip()]

    if not contacts:
        logger.info(f"No contacts configured. Alert message would be: {message}")
        return False

    if _TWILIO_AVAILABLE and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
        try:
            client = Client(TWILIO_SID, TWILIO_TOKEN)
            for to in contacts:
                client.messages.create(body=message, from_=TWILIO_FROM, to=to)
            logger.info(f"Sent alert to {len(contacts)} contacts via Twilio")
            return True
        except Exception as e:
            logger.exception(f"Failed to send alert via Twilio: {e}")
            return False
    else:
        # fallback: just log
        for to in contacts:
            logger.info(f"(stub) Would send to {to}: {message}")
        return False
