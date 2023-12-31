import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
import json
from image_processor import ImageProcessor
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from json_numpy import default

class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # unfreeze the last two layers and retrain entire model   
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # change final layer of resnet-50 
        self.resnet50.fc = torch.nn.Sequential(torch.nn.Linear(2048,1000))
        self.decoder = decoder

    def forward(self, image):
        x = self.resnet50(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

try:
    with open("image_decoder.pkl", "rb") as file:
        decoder = json.load(file)   
        feature_extractor = FeatureExtractor(decoder=decoder)
        state = torch.load("image_model.pt",map_location=torch.device('cpu'))
        feature_extractor.load_state_dict(state[0])
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    import faiss
    with open("index.pkl", "rb") as file:
        faiss_index = faiss.deserialize_index(pickle.load(file))
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_df = pd.read_csv('training_data.csv',lineterminator ='\n', index_col=0)
image_df = image_df.rename(columns={"label\r" : "category"})
image_df["category"] = image_df["category"].astype("string")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}
  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    image_processor = ImageProcessor()
    processed_image = image_processor.process(pil_image)
    image_embedding = feature_extractor(processed_image.unsqueeze(0))
    json_result = json.dumps(image_embedding.detach().numpy(), default=default)

    return JSONResponse(content={
    "features": json_result # Return the image embeddings here
    
        })
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    image_processor = ImageProcessor()
    processed_image = image_processor.process(pil_image)
    image_embedding = feature_extractor(processed_image.unsqueeze(0))

    D, I = faiss_index.search(image_embedding.detach().numpy(), 5) 
    results = pd.DataFrame({'distances': D[0], 'approximate_nearest_neighbour': I[0]})
    merge = pd.merge(results,image_df,left_on='approximate_nearest_neighbour', right_index=True)    
    print(decoder)
    merge["category"] = merge["category"].replace(decoder)
    print(merge.info())
    return JSONResponse(content={
    "similar_index": merge.to_dict(orient="records"), # Return the index of similar images here
        })

        
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)