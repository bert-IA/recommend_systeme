import json
from function_app import event_grid_trigger
from azure.functions import EventGridEvent

# Créer un événement Event Grid simulé
event_data = {
    "id": "1",
    "eventType": "NewUserAdded",
    "subject": "NewUserAdded",
    "eventTime": "2025-02-19T14:10:54Z",
    "data": {
        "user_id": 322884
    },
    "dataVersion": "1.0",
    "topic": "/subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.EventGrid/topics/{topic-name}"
}

event = EventGridEvent(
    id=event_data["id"],
    subject=event_data["subject"],
    data=event_data["data"],
    event_type=event_data["eventType"],
    event_time=event_data["eventTime"],
    data_version=event_data["dataVersion"],
    topic=event_data["topic"]
)

# Appeler directement la fonction Event Grid Trigger
event_grid_trigger(event)