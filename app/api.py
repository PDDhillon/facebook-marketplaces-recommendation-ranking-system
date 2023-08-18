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
class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # unfreeze the last two layers and retrain entire model   
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # change final layer of resnet-50 to have an output size = number of possible categories (13 categories)
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
        data = json.load(file)   
        feature_extractor = FeatureExtractor(decoder=data)
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

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
from json_numpy import default
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
    # results = pd.DataFrame({'distances': D[0], 'ann': I[0]})
    # df = pd.read_csv('D:/Documents/AICore/facebook-marketplaces-recommendation-ranking-system/training_data.csv',lineterminator ='\n')
    # merge = pd.merge(results,df,left_on='ann', right_index=True)
    print(I[0].tolist())
    return JSONResponse(content={
    "similar_index": I[0].tolist() # Return the index of similar images here
        })
        
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)