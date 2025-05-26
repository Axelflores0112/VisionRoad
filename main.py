from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import json

app = FastAPI()
model = load_model("modelo_senal.h5")

#Permitir las peticiones desde mi app android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#Quien va poder hacer peteciones (todos por el momento)
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

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
        class_name = index_to_class_name [str(predicted_class)]
        return {"prediccion": class_name}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})