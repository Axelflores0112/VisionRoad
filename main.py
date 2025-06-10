from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import json
import os

app = FastAPI()
model = load_model("modelo_senal.h5")
COUNT_FILE = "signal_counts.json"
LAST_PREDICTION_FILE = "last_prediction.json"

#Permitir las peticiones desde mi app android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#Quien va poder hacer peteciones (todos por el momento)
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

def load_counts():
    if os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_counts(counts):
    with open(COUNT_FILE, "w") as f:
        json.dump(counts, f) 

def save_last_prediction(data):
    with open(LAST_PREDICTION_FILE, "w") as f:
        json.dump(data, f)

def load_last_prediction():
    if os.path.exists(LAST_PREDICTION_FILE):
        with open(LAST_PREDICTION_FILE, "r") as f:
            return json.load(f)
    return {"prediccion": None, "confidence": 0.0, "total_detected": 0}

@app.get("/")
def read_root():
    return {"mensaje": "backend funcionando" }

# Cargar el diccionario de índices a nombres de clases
with open("index_to_class_name.json", "r") as f:
    index_to_class_name = json.load(f)

@app.post("/prediccion/")
async def prediccion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).resize((128, 128))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        class_name = index_to_class_name[str(predicted_class)]
        confidence = float(np.max(prediction))

        counts = load_counts()
        counts[class_name] = counts.get(class_name, 0) + 1
        save_counts(counts)

        # Guardar la última predicción
        last_pred = {
            "prediccion": class_name,
            "confidence": confidence,
            "total_detected": sum(counts.values())
        }
        save_last_prediction(last_pred)

        return last_pred
    except Exception as e:
        print("Error en prediccion:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/conteos/")
def get_counts():
    counts = load_counts()
    return counts

@app.get("/last_prediction/")
def get_last_prediction():
    return load_last_prediction()

@app.post("/reset_counts/")
def reset_counts():
    save_counts({})  # Guarda un diccionario vacío
    return {"mensaje": "Conteos reiniciados"}