from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
from detect_prescription import detect_medicines

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API ROUTE
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    print("Received file:", file.filename)

    file_location = "temp.png"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    medicines = detect_medicines(file_location)

    print("Detected medicines:", medicines)

    return {"detected_medicines": medicines}


# FRONTEND (must be after API)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)