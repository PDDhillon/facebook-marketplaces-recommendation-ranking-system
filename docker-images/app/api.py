import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
import json
##############################################################
# TODO                                                       #
# Import your image processing script here                 #
##############################################################
from image_processor import ImageProcessor
class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################
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

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:
#################################################################
# TODO                                                          #
# Load the Feature Extraction model. Above, we have initialized #
# a class that inherits from nn.Module, and has the same        #
# structure as the model that you used for training it. Load    #
# the weights in it here.                                       #
#################################################################
    with open("index.pkl", "rb") as file:
        data = pickle.load(file)   
        feature_extractor = FeatureExtractor(decoder=data)
        state = torch.load("image_model.pt",map_location=torch.device('cpu'))
        feature_extractor.load_state_dict(state[0])
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
##################################################################
# TODO                                                           #
# Load the FAISS model. Use this space to load the FAISS model   #
# which is was saved as a pickle with all the image embeddings   #
# fit into it.                                                   #
##################################################################
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

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ################################################################
    # TODO                                                         #
    # Process the input and use it as input for the feature        #
    # extraction model image. File is the image that the user      #
    # sent to your API. Apply the corresponding methods to extract #
    # the image features/embeddings.                               #
    ################################################################
    image_processor = ImageProcessor()
    processed_image = image_processor.process(pil_image)
    image_embedding = feature_extractor(processed_image)

    return JSONResponse(content={
    "features": image_embedding, # Return the image embeddings here
    
        })
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    #####################################################################
    # TODO                                                              #
    # Process the input  and use it as input for the feature            #
    # extraction model.File is the image that the user sent to your API #   
    # Once you have feature embeddings from the model, use that to get  # 
    # similar images by passing the feature embeddings into FAISS       #
    # model. This will give you index of similar images.                #            
    #####################################################################
    image_processor = ImageProcessor()
    processed_image = image_processor.process(pil_image)
    image_embedding = feature_extractor(processed_image)
    D, I = faiss_index.search(image_embedding, 1) 
    return JSONResponse(content={
    "similar_index": I, # Return the index of similar images here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)