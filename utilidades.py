import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO
import sys


labels_csv = pd.read_csv("F:/Repositorio/DiNo/DINO.BACKEND/API_DiNo/Python/labels.csv")
labels = labels_csv["breed"]
unique_breeds = np.unique(labels)
  

def Base64ToImage(cadena):
    
  #decodeit = open('imagenPerro.jpeg', 'wb')
  imagenPerro = Image.open(BytesIO(base64.b64decode(cadena)))
  imagenPerro.save('./imagenes/imagenPerro.png', 'PNG')
  
  return imagenPerro


# Crear función para preprocesar las imagenes
def process_image(image_path, img_size=224):
  """
  Takes an image file path and turns image into a tensor
  """
  # Leer un archivo de imagen
  image = tf.io.read_file(image_path)
  # Convertir la imagen .jpg en un tensor numérico con 3 canales de color
  image = tf.image.decode_jpeg(image, channels=3)
  # Convertir los valores del canal de color de 0-255 a 0-1 valores
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Cambiar el tamaño de la imagen a nuestro valor deseado (224,224)
  image = tf.image.resize(image, size=[img_size, img_size])

  return image   


# Crear función para convertir datos a lotes
def create_data_batches(X, y=None, batch_size=32, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (X) and label (y) pairs.
  Shuffles the data if it is training data but does not shuffle
  validations data. 
  Also accepts test data as input (no labels).
  """
  # Si prueba datos, no tenemos etiquetas
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # solo rutas de archivo (sin etiquetas)
    data_batch = data.map(process_image).batch(batch_size)
    print("Test data batches created")
    
    return data_batch


# Convertir las probabilidades de predicción en sus respectivas etiquetas (más fácil de entender)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label
  """ 
  return unique_breeds[np.argmax(prediction_probabilities)]


# Crea una función para cargar el modelo
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

def AnalisisHuella():
      # Obtener rutas de archivo de imagen personalizadas
  custom_path = "C:/Users/juanc/Documents/Camilo/backUpPCTeleconet/DINO/DINO.BACKEND/API_DiNo/API_DiNo/API_DiNo/Python/imagenes/"
  custom_image_paths = [custom_path + fname for fname in os.listdir(custom_path)]
  # Convertir imágenes personalizadas en conjuntos de datos por lotes
  custom_data = create_data_batches(custom_image_paths, test_data=True)
  # Hacer predicciones sobre datos personalizados
  custom_preds = loaded_full_model.predict(custom_data)
  # Obtener etiquetas personalizadas de predicción de imágenes
  custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
  
  print (custom_pred_labels)
  
  