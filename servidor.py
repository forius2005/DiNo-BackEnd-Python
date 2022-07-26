"""Filename: server.py
"""
import pandas as pd
from flask import Flask, jsonify, request

from utilidades import *

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """API request
    """
    try:
        
        req_json = request.get_json()
        string = req_json['string'] 
    except Exception as e:
        raise e

    if string == '':
        return(bad_request())
    else:
        # Load the saved model
        print("Cargar el modelo...")
        loaded_model = load_model("20210815-183150-full-modelo-entrenado.h5")
        
        print("Convertir base64 a imagen...")
        convert = Base64ToImage(string)
        
        custom_path = "F:/Repositorio/DiNo/DINO.BACKEND/API_DiNo/Python/imagenes/"
        custom_image_paths = [custom_path +
                            fname for fname in os.listdir(custom_path)]
        # Convertir imágenes personalizadas en conjuntos de datos por lotes
        custom_data = create_data_batches(custom_image_paths, test_data=True)
        # Hacer predicciones sobre datos personalizados
        custom_preds = loaded_model.predict(custom_data)
        # Obtener etiquetas personalizadas de predicción de imágenes
        custom_pred_labels = [get_pred_label(
            custom_preds[i]) for i in range(len(custom_preds))]

        print("Enviar respuesta")
        responses = jsonify(  
            predictions= custom_pred_labels)
        responses.status_code = 200
        print("Fin de Peticion")

        return (responses)
