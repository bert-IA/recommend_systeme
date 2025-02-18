import logging
import pickle
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
import azure.functions as func
from scipy.sparse import csr_matrix, vstack, hstack
import os
import io
from dotenv import load_dotenv
from implicit.als import AlternatingLeastSquares
from azure.eventgrid import EventGridPublisherClient, EventGridEvent
from azure.core.credentials import AzureKeyCredential

# Load environment variables from .env file
load_dotenv()

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)

# Variables globales
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
EVENT_GRID_TOPIC_ENDPOINT = os.getenv("EVENT_GRID_TOPIC_ENDPOINT")
EVENT_GRID_TOPIC_KEY = os.getenv("EVENT_GRID_TOPIC_KEY")

if not CONNECTION_STRING:
    logging.error("Azure Storage connection string is not set.")
else:
    logging.info(f"Azure Storage connection string: {CONNECTION_STRING}")

if not EVENT_GRID_TOPIC_ENDPOINT or not EVENT_GRID_TOPIC_KEY:
    logging.error("Event Grid topic endpoint or key is not set.")
else:
    logging.info(f"Event Grid topic endpoint: {EVENT_GRID_TOPIC_ENDPOINT}")

CONTAINER_NAME = "input"

# Définir l'application Azure Functions
app = func.FunctionApp()

# Charger le modèle, les données utilisateur-article et les mappings
def load_model_and_data():
    logging.info("Loading model and data from Azure Blob Storage")
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    
    # Charger le modèle
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="als_model.pkl")
    model_data = blob_client.download_blob().readall()
    model = pickle.loads(model_data)
    logging.info("Model loaded successfully")
    
    # Charger user_item_matrix
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_item_matrix.npz")
    user_item_matrix_data = blob_client.download_blob().readall()
    user_item_matrix_buffer = io.BytesIO(user_item_matrix_data)
    loader = np.load(user_item_matrix_buffer)
    user_item_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    logging.info(f"User-item matrix loaded successfully with shape {user_item_matrix.shape}")
    
    # Charger user_id_map
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_id_map.json")
    user_id_map_data = blob_client.download_blob().readall()
    user_id_map = json.loads(user_id_map_data)
    user_id_map = {int(k): v for k, v in user_id_map.items()}  # Convertir les clés en entiers
    logging.info("User ID map loaded successfully")
    
    # Charger article_id_map
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="article_id_map.json")
    article_id_map_data = blob_client.download_blob().readall()
    article_id_map = json.loads(article_id_map_data)
    article_id_map = {int(k): v for k, v in article_id_map.items()}  # Convertir les clés en entiers
    logging.info("Article ID map loaded successfully")
    
    # Charger article_idx_map
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="article_idx_map.json")
    article_idx_map_data = blob_client.download_blob().readall()
    article_idx_map = json.loads(article_idx_map_data)
    article_idx_map = {int(k): v for k, v in article_idx_map.items()}  # Convertir les clés en entiers
    logging.info("Article IDX map loaded successfully")
    
    return model, user_item_matrix, user_id_map, article_id_map, article_idx_map

def recommend(user_id, sparse_user_item, model, user_id_map, article_idx_map, num_items=5):
    user_idx = user_id_map.get(user_id, None)
    if user_idx is None:
        logging.error(f"User {user_id} not found in user_id_map")
        raise ValueError(f"User {user_id} not found")
    
    logging.info(f"Recommending items for user ID {user_id} with index {user_idx}")
    logging.info(f"user_item_matrix shape: {sparse_user_item.shape}")
    
    # Créer une matrice sparse pour l'utilisateur
    user_interactions_csr = csr_matrix(sparse_user_item[user_idx, :])
    logging.info(f"user_interactions_csr shape: {user_interactions_csr.shape}")
    logging.info(f"user_interactions_csr data: {user_interactions_csr.data}")
    
    # Appeler la méthode recommend avec filter_already_liked_items=True
    item_ids, scores = model.recommend(user_idx, user_interactions_csr, N=num_items, filter_already_liked_items=True)
    logging.info(f"Recommended item IDs: {item_ids}")
    logging.info(f"Scores: {scores}")
    
    item_ids = np.array(item_ids[:num_items])
    recommendations = np.vectorize(article_idx_map.get)(item_ids)
    logging.info(f"Recommendations: {recommendations}")
    
    return recommendations

# Sauvegarder la matrice mise à jour
def save_user_item_matrix(user_item_matrix, user_id_map):
    logging.info("Saving updated user-item matrix and user ID map to Azure Blob Storage")
    logging.info(f"user_item_matrix shape before saving: {user_item_matrix.shape}")
    
    # Convertir la matrice user_item_matrix en fichier .npz en mémoire
    user_item_matrix_buffer = io.BytesIO()
    np.savez_compressed(
        user_item_matrix_buffer, 
        data=user_item_matrix.data, 
        indices=user_item_matrix.indices, 
        indptr=user_item_matrix.indptr, 
        shape=user_item_matrix.shape
    )
    user_item_matrix_buffer.seek(0)  # Revenir au début du buffer
    
    # Convertir user_id_map en JSON en mémoire
    user_id_map_json = json.dumps(user_id_map)
    user_id_map_buffer = io.BytesIO(user_id_map_json.encode('utf-8'))
    
    # Sauvegarder les fichiers dans Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    
    # Sauvegarder user_item_matrix.npz
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_item_matrix.npz")
    blob_client.upload_blob(user_item_matrix_buffer, overwrite=True)
    
    # Sauvegarder user_id_map.json
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_id_map.json")
    blob_client.upload_blob(user_id_map_buffer, overwrite=True)
    
    logging.info("User-item matrix and user ID map saved successfully")

# Réentraîner le modèle
def retrain_and_save_model(user_item_matrix, user_id_map):
    logging.info("Retraining the model with updated user-item matrix")
    model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
    model.fit(user_item_matrix)
    logging.info("Model retrained successfully")

    # Sauvegarder le modèle
    model_buffer = io.BytesIO()
    pickle.dump(model, model_buffer)
    model_buffer.seek(0)

    # Sauvegarder la matrice user_item_matrix au format .npz
    user_item_matrix_buffer = io.BytesIO()
    np.savez(user_item_matrix_buffer, 
             data=user_item_matrix.data, 
             indices=user_item_matrix.indices, 
             indptr=user_item_matrix.indptr, 
             shape=user_item_matrix.shape)
    user_item_matrix_buffer.seek(0)

    # Convertir user_id_map en JSON en mémoire
    user_id_map_json = json.dumps(user_id_map)
    user_id_map_buffer = io.BytesIO(user_id_map_json.encode('utf-8'))

    # Sauvegarder les fichiers dans Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

    # Sauvegarder als_model.pkl
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="als_model.pkl")
    blob_client.upload_blob(model_buffer, overwrite=True)
    logging.info("Model saved successfully in Azure Blob Storage")

    # Sauvegarder user_item_matrix.npz
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_item_matrix.npz")
    blob_client.upload_blob(user_item_matrix_buffer, overwrite=True)
    logging.info("user_item_matrix saved successfully in Azure Blob Storage")

    # Sauvegarder user_id_map.json
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="user_id_map.json")
    blob_client.upload_blob(user_id_map_buffer, overwrite=True)
    logging.info("user_id_map saved successfully in Azure Blob Storage")

    return model

# Fonction pour ajouter un nouvel utilisateur
def add_new_user(article_clicks, user_item_matrix, article_id_map, user_id_map):
    # Générer un nouvel user_id unique
    if user_id_map:
        new_user_id = max(user_id_map.keys()) + 1
    else:
        new_user_id = 1  # Commencer à 1 si user_id_map est vide
    
    logging.info(f"Adding new user with ID {new_user_id}")
    
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
    
    # Ajouter l'utilisateur à user_id_map
    user_id_map[new_user_id] = user_item_matrix.shape[0] - 1
    
    logging.info(f"New user with ID {new_user_id} added successfully")
    logging.info(f"user_id_map[{new_user_id}] = {user_id_map[new_user_id]}")
    logging.info(f"user_item_matrix shape: {user_item_matrix.shape}")
    logging.info(f"New user interactions: {new_user_row_sparse.data}")
    
    return user_item_matrix, user_id_map, new_user_id

def add_new_article(article_id, user_item_matrix, article_id_map, article_idx_map):
    # Vérifier si l'article existe déjà
    if article_id in article_id_map:
        logging.error(f"Article with ID {article_id} already exists")
        raise ValueError(f"Article with ID {article_id} already exists")

    # Générer un nouvel article_id unique
    if article_id_map:
        new_article_idx = max(article_id_map.values()) + 1
    else:
        new_article_idx = 0  # Commencer à 0 si article_id_map est vide

    logging.info(f"Adding new article with ID {article_id} and index {new_article_idx}")

    # Ajouter l'article aux mappings
    article_id_map[article_id] = new_article_idx
    article_idx_map[new_article_idx] = article_id

    # Ajouter une nouvelle colonne pour le nouvel article
    new_article_col = csr_matrix((user_item_matrix.shape[0], 1))
    user_item_matrix = hstack([user_item_matrix, new_article_col])

    logging.info(f"New article with ID {article_id} added successfully")
    logging.info(f"article_id_map[{article_id}] = {article_id_map[article_id]}")
    logging.info(f"user_item_matrix shape: {user_item_matrix.shape}")

    return user_item_matrix, article_id_map, article_idx_map

# Publier des événements dans Event Grid
def publish_event(event_type, data):
    event_grid_client = EventGridPublisherClient(EVENT_GRID_TOPIC_ENDPOINT, AzureKeyCredential(EVENT_GRID_TOPIC_KEY))
    event = EventGridEvent(
        subject="RecommenderSystem",
        event_type=event_type,
        data=data,
        data_version="1.0"
    )
    event_grid_client.send([event])

@app.function_name(name="HttpTrigger")
@app.route(route="recommendation", methods=["GET", "POST"])
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    action = req.params.get('action')
    user_id = req.params.get('user_id')
    article_id = req.params.get('article_id')
    article_clicks = req.get_json().get('article_clicks', {}) if action == 'add_user' else {}

    # Charger les données
    model, user_item_matrix, user_id_map, article_id_map, article_idx_map = load_model_and_data()

    if action == 'recommend' and user_id:
        logging.info(f"Recommending items for user ID {user_id}")
        try:
            recommendations = recommend(int(user_id), user_item_matrix, model, user_id_map, article_idx_map, num_items=5)
            logging.info(f"Recommended items for user {user_id}: {recommendations}")
            return func.HttpResponse(f"Recommended items for user {user_id}: {recommendations}")
        except ValueError as e:
            logging.error(str(e))
            return func.HttpResponse(str(e), status_code=404)

    elif action == 'add_user':
        logging.info(f"Adding new user")
        try:
            # Ajouter le nouvel utilisateur
            user_item_matrix, user_id_map, new_user_id = add_new_user(article_clicks, user_item_matrix, article_id_map, user_id_map)
            save_user_item_matrix(user_item_matrix, user_id_map)
            
            # Publier un événement pour le réentraînement du modèle
            publish_event("NewUserAdded", {"user_id": new_user_id})
            
            logging.info(f"User {new_user_id} added successfully.")
            return func.HttpResponse(f"User {new_user_id} added successfully.")
        except ValueError as e:
            logging.error(str(e))
            return func.HttpResponse(str(e), status_code=400)

    elif action == 'add_article' and article_id:
        logging.info(f"Adding article with ID {article_id}")
        article_id = int(article_id)
        try:
            # Ajouter le nouvel article
            user_item_matrix, article_id_map, article_idx_map = add_new_article(article_id, user_item_matrix, article_id_map, article_idx_map)
            save_user_item_matrix(user_item_matrix, user_id_map)
            
            # Publier un événement pour le réentraînement du modèle
            publish_event("NewArticleAdded", {"article_id": article_id})
            
            logging.info(f"Article {article_id} added successfully.")
            return func.HttpResponse(f"Article {article_id} added successfully.")
        except ValueError as e:
            logging.error(str(e))
            return func.HttpResponse(str(e), status_code=400)

    else:
        logging.error("Invalid action or missing parameters")
        return func.HttpResponse(
             "Please pass a valid action (recommend, add_user, add_article) and corresponding parameters (user_id, article_id)",
             status_code=400
        )