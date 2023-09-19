from fastapi import FastAPI, File, UploadFile, Request, Response
from typing import Annotated
from pydantic import BaseModel
import numpy as np
from idcontrails.ml_logic.building_models import load_model

import numpy as np
import cv2

# @dataclass
# class TestNumpyArray:
#     image: np.ndarray

# class ImageMeteo(BaseModel):
#     image: np.ndarray
#     class Config:
#         arbitrary_types_allowed = True #si ca fonctionne pas, investiguer le bail "dataclass" de pydantic... Si ca fonctionne tout va bien

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




# @app.post("/files/")
# async def create_file(file: Annotated[bytes | None, File()] = None):
#     if not file:
#         return {"message": "No file sent"}
#     else:
#         return {"file_size": len(file)}

    # if not file:
    #     return {"message": "No file sent"}
    # if not file.image :
    #     return {"message": "No image in file"}

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile | None = None):
#     if not file:
#         return {"message": "No upload file sent"}
#     else:
#         return {"filename": file.filename}

#to run locally, cli from package's root folder is "uvicorn idcontrails.interface.fast_test_api:app --reload"

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}
