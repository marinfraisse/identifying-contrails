from fastapi import FastAPI, File, UploadFile, Request, Response
# from typing import Annotated
# from pydantic import BaseModel
# import numpy as np
from idcontrails.ml_logic.building_models import load_model

import numpy as np

local_url = "http://127.0.0.1:8000"
app = FastAPI()
model=load_model()

@app.post("/upload_image/")
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    X_api = np.frombuffer(contents, dtype = np.float32).reshape(256,256,3)
    X_api = np.expand_dims(X_api , axis=0)
    X_mask = model.predict(X_api)

    print(X_mask.shape)
    X_final = X_mask.tobytes()

    return Response(content=X_final)

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'hello': "world"}
