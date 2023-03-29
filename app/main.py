from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
from deblur_image import deblur
import os

app = FastAPI()

IMAGE = None

@app.post('/uploadfile')
async def upload_file(image : UploadFile = File(...)):
    extension = image.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    
    if not extension:
        return "Image must be jpg or png format!"
    
    with open(f"blurred_img/{image.filename}", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    global IMAGE
    IMAGE = image.filename

    return "File Uploaded successfully!"

@app.get('/predict')
def predict():
    deblur()
    return FileResponse(f"deblurred_img/{'deblurred ' + IMAGE}")

@app.post('/clear')
def clear():
    os.remove(f"blurred_img/{IMAGE}")
    os.remove(f"deblurred_img/{'deblurred '+IMAGE}")

    return "Cleared!"

    

