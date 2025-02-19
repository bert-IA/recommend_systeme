import requests
import json

event = {
    "id": "1",
    "eventType": "NewUserAdded",
    "subject": "NewUserAdded",
    "eventTime": "2025-02-19T14:10:54Z",
    "data": {
        "user_id": 322884
    },
    "dataVersion": "1.0"
}

headers = {
    "aeg-sas-key": "7R9WAKSNVkRTbosfzOZiortUGXTxOe3KOyb9NXi10AviZHzFpXv6JQQJ99BBAC5T7U2XJ3w3AAABAZEGAiQg",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://myeventgridtopic.francecentral-1.eventgrid.azure.net/api/events",
    headers=headers,
    data=json.dumps([event])
)

print(response.status_code)
print(response.text)