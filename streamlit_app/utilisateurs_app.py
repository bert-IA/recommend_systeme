import streamlit as st
import requests
import logging
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env dans le répertoire parent
load_dotenv(dotenv_path="../.env")

logging.basicConfig(level=logging.DEBUG)

# Récupérer la clé d'API depuis les variables d'environnement
api_key = os.getenv("AZURE_FUNCTIONS_KEY")
logging.debug(f"API Key: {api_key}")  # Ajouter un log pour vérifier la clé d'API

# Définir les fonctions pour chaque page
def menu():
    st.title("Application de Recommandation de Contenu")
    if st.button("Recommendations"):
        logging.debug("Navigating to Recommendations page")
        st.session_state.page = "recommendations"
    if st.button("Ajout d'Utilisateurs"):
        logging.debug("Navigating to Add User page")
        st.session_state.page = "add_user"
    if st.button("Ajout d'Articles"):
        logging.debug("Navigating to Add Article page")
        st.session_state.page = "add_article"

def recommendations():
    st.title("Recommandations pour Utilisateur")
    user_id = st.radio("Choisissez un utilisateur", [15497, 163334, 129204, 46716, 171672])  # échantillon
    if st.button("Obtenir Recommandations"):
        logging.debug(f"Fetching recommendations for user_id: {user_id}")
        headers = {"x-functions-key": api_key}
        response = requests.get(f"https://recommendals.azurewebsites.net/api/recommendation?action=recommend&user_id={user_id}", headers=headers)
        if response.status_code == 200:
            st.write(f"Recommandations pour l'utilisateur {user_id}: {response.json()}")
        else:
            st.write(f"Erreur: {response.text}")
            logging.error(f"Error fetching recommendations: {response.text}")

def add_user():
    st.title("Ajout d'un Nouvel Utilisateur")
    article_clicks = st.text_area("Articles cliqués et nombre de clics (format: {1: 5, 2: 3})")
    if st.button("Ajouter Utilisateur"):
        try:
            article_clicks_dict = eval(article_clicks)
            logging.debug(f"Adding user with article_clicks: {article_clicks_dict}")
            headers = {"x-functions-key": api_key}
            response = requests.post("https://recommendals.azurewebsites.net/api/recommendation?action=add_user", json={"article_clicks": article_clicks_dict}, headers=headers)
            if response.status_code == 200:
                st.write(f"Utilisateur ajouté avec succès: {response.json()}")
            else:
                st.write(f"Erreur: {response.text}")
                logging.error(f"Error adding user: {response.text}")
        except Exception as e:
            st.write(f"Erreur de format: {e}")
            logging.error(f"Format error: {e}")

def add_article():
    st.title("Ajout d'un Nouvel Article")
    article_id = st.text_input("ID de l'article à ajouter")
    if st.button("Ajouter Article"):
        logging.debug(f"Adding article with article_id: {article_id}")
        headers = {"x-functions-key": api_key}
        response = requests.post(f"https://recommendals.azurewebsites.net/api/recommendation?action=add_article&article_id={article_id}", headers=headers)
        if response.status_code == 200:
            st.write(f"Article ajouté avec succès: {response.json()}")
        else:
            st.write(f"Erreur: {response.text}")
            logging.error(f"Error adding article: {response.text}")

# Navigation entre les pages
if "page" not in st.session_state:
    st.session_state.page = "menu"

if st.session_state.page == "menu":
    menu()
elif st.session_state.page == "recommendations":
    recommendations()
elif st.session_state.page == "add_user":
    add_user()
elif st.session_state.page == "add_article":
    add_article()

st.sidebar.button("Menu Principal", on_click=lambda: setattr(st.session_state, "page", "menu"))