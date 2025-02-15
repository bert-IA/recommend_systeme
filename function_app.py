import logging
import joblib
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
import azure.functions as func
from scipy.sparse import csr_matrix, vstack, hstack
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)

# Variables globales

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "input"

# Charger le modèle, les données utilisateur-article et les mappings
def load_model_and_data():
    logging.info("Loading model and data from Azure Blob Storage")
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    
    # Charger le modèle
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="als_model.pkl")
    with open("/tmp/als_model.pkl", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    model = joblib.load("/tmp/als_model.pkl")
    logging.info("Model loaded successfully")
    
    # Charger user_item_matrix
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_item_matrix.npz")
    with open("/tmp/user_item_matrix.npz", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    user_item_matrix = csr_matrix(np.load("/tmp/user_item_matrix.npz"))
    logging.info("User-item matrix loaded successfully")
    
    # Charger user_id_map
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_id_map.json")
    with open("/tmp/user_id_map.json", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    with open("/tmp/user_id_map.json", "r") as f:
        user_id_map = {int(k): v for k, v in json.load(f).items()}
    logging.info("User ID map loaded successfully")
    
    # Charger article_id_map
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="article_id_map.json")
    with open("/tmp/article_id_map.json", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    with open("/tmp/article_id_map.json", "r") as f:
        article_id_map = {int(k): v for k, v in json.load(f).items()}
    logging.info("Article ID map loaded successfully")
    
    # Charger article_idx_map
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="article_idx_map.json")
    with open("/tmp/article_idx_map.json", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    with open("/tmp/article_idx_map.json", "r") as f:
        article_idx_map = {int(k): v for k, v in json.load(f).items()}
    logging.info("Article IDX map loaded successfully")
    
    return model, user_item_matrix, user_id_map, article_id_map, article_idx_map

# Sauvegarder la matrice mise à jour
def save_user_item_matrix(user_item_matrix):
    logging.info("Saving updated user-item matrix to Azure Blob Storage")
    np.savez("/tmp/user_item_matrix.npz", data=user_item_matrix.data, indices=user_item_matrix.indices, indptr=user_item_matrix.indptr, shape=user_item_matrix.shape)
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_item_matrix.npz")
    
    with open("/tmp/user_item_matrix.npz", "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.info("User-item matrix saved successfully")

# Fonction pour ajouter un nouvel utilisateur
def add_new_user(user_id, article_clicks, user_item_matrix, article_id_map):
    logging.info(f"Adding new user with ID {user_id}")
    # Calculer la moyenne des clics de l'utilisateur
    mean_clicks = np.mean([clicks for article_id, clicks in article_clicks.items()])
    # Créer une nouvelle ligne pour le nouvel utilisateur
    new_user_row = np.zeros(user_item_matrix.shape[1])
    for article_id, clicks in article_clicks.items():
        article_idx = article_id_map.get(article_id, None)
        if article_idx is not None:
            new_user_row[article_idx] = clicks / mean_clicks
    
    # Convertir la nouvelle ligne en matrice sparse
    new_user_row_sparse = csr_matrix(new_user_row)
    
    # Ajouter la nouvelle ligne à la matrice user_item_matrix
    user_item_matrix = vstack([user_item_matrix, new_user_row_sparse])
    logging.info(f"New user with ID {user_id} added successfully")
    
    return user_item_matrix

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    action = req.params.get('action')
    user_id = req.params.get('user_id')
    article_id = req.params.get('article_id')
    article_clicks = req.get_json().get('article_clicks', {}) if action == 'add_user' else {}

    model, user_item_matrix, user_id_map, article_id_map, article_idx_map = load_model_and_data()

    if action == 'recommend' and user_id:
        logging.info(f"Recommending items for user ID {user_id}")
        user_idx = user_id_map.get(int(user_id), None)
        if user_idx is None:
            logging.error(f"User {user_id} not found")
            return func.HttpResponse(f"User {user_id} not found", status_code=404)
        
        user_items = user_item_matrix.getrow(user_idx)
        recommended_items = model.recommend(user_idx, user_items, N=5)
        recommended_items = [article_idx_map[item] for item, score in recommended_items]
        logging.info(f"Recommended items for user {user_id}: {recommended_items}")
        return func.HttpResponse(f"Recommended items for user {user_id}: {recommended_items}")

    elif action == 'add_user' and user_id:
        logging.info(f"Adding user with ID {user_id}")
        user_id = int(user_id)
        # Ajouter le nouvel utilisateur
        user_item_matrix = add_new_user(user_id, article_clicks, user_item_matrix, article_id_map)
        save_user_item_matrix(user_item_matrix)
        return func.HttpResponse(f"User {user_id} added successfully.")

    elif action == 'add_article' and article_id:
        logging.info(f"Adding article with ID {article_id}")
        article_id = int(article_id)
        # Ajouter une nouvelle colonne pour le nouvel article
        new_article_col = csr_matrix((user_item_matrix.shape[0], 1))
        user_item_matrix = hstack([user_item_matrix, new_article_col])
        save_user_item_matrix(user_item_matrix)
        return func.HttpResponse(f"Article {article_id} added successfully.")

    else:
        logging.error("Invalid action or missing parameters")
        return func.HttpResponse(
             "Please pass a valid action (recommend, add_user, add_article) and corresponding parameters (user_id, article_id)",
             status_code=400
        )