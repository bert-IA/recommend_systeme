from azure.storage.blob import BlobServiceClient
import os

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

try:
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    print("Connection to Azure Blob Storage successful.")
except Exception as e:
    print(f"Error connecting to Azure Blob Storage: {e}")