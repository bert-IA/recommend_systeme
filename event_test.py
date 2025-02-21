from azure.eventgrid import EventGridPublisherClient, EventGridEvent
from azure.core.credentials import AzureKeyCredential
import os

EVENT_GRID_TOPIC_ENDPOINT = os.getenv("EVENT_GRID_TOPIC_ENDPOINT")
EVENT_GRID_TOPIC_KEY = os.getenv("EVENT_GRID_TOPIC_KEY")

event = EventGridEvent(
    subject="NewUserAdded",
    event_type="NewUserAdded",
    data={"user_id": 123},
    data_version="1.0"
)

credential = AzureKeyCredential(EVENT_GRID_TOPIC_KEY)
client = EventGridPublisherClient(EVENT_GRID_TOPIC_ENDPOINT, credential)
client.send(event)

print("Event sent successfully.")