import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Cargar modelo
model = load_model("modelo_clasificacion_rostros_fondo_op_3.h5")

st.title("Clasificador de Rostros")

# Subir imagen
archivo = st.file_uploader("Sube una imagen", type=["jpg","jpeg","png"])

def procesar_imagen(img):
    img = cv2.resize(img, (224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

if archivo is not None:

    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Imagen cargada")

    img_procesada = procesar_imagen(img)

    pred = model.predict(img_procesada)

    if pred[0][0] > 0.5:
        resultado = "Nathaly"
    else:
        resultado = "Fondo"

    st.subheader("Resultado:")
    st.write(resultado)