import logging
import azure.functions as func
from .function_app import retrain_and_save_model, load_model_and_data

def main(event: func.EventGridEvent):
    logging.info('Python EventGrid trigger function processed an event: %s', event.get_json())
    
    model, user_item_matrix, user_id_map, article_id_map, article_idx_map = load_model_and_data()
    model = retrain_and_save_model(user_item_matrix, user_id_map)
    
    logging.info("Model retrained and saved successfully.")