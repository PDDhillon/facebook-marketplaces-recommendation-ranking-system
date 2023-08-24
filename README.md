# Facebook Marketplace Recommendation Ranking System

Facebook Marketplace is a market leading platform for buying and selling products online. The hallmark of this platform is the speed and efficiency at which is it can reccomend products to a user, based on their previously viewed items. Behind the scenes, facebook is able to process image data alone in order to categorise and rank images. This ranking system provides a mathematical representation of the similarity of two images and stores it in a FAISS index.

This project aims to recreate the ranking system in order to perform the same functionality. A multiclass classification model is trained and used as a feature extraction model, which is fed into Facebook AI Similarity Search (FAISS) index to perform a similarity search. The FAISS index is accessed through API endpoints to provide similar images for a given single input. The API is then consumed by a React App in order to provide a graphical represenation of the results.

## Table of Contents
-[Usage](#usage)

-[File Structure](#file-structure)

-[Explore the Dataset](#explore-the-dataset)

-[Create the vision model, then turn it into a feature extraction model](#create-the-vision-model-then-turn-it-into-a-feature-extraction-model)

-[Create the search index using FAISS](#create-the-search-index-using-faiss)

-[Configure and deploy the model serving API](#configure-and-deploy-the-model-serving-api)

-[Consume the API and provide a user interface](#consume-the-api-and-provide-a-user-interface)


## Usage
To use the final solution, you can access the API directly [here](http://54.170.80.153:8080/docs).

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/API.jpg?raw=true)

Or you can view the deployed react app [here](https://main.d18kreqtn2vh50.amplifyapp.com/). (Currently fixing mixed content issue)

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/darkmode.png?raw=true)

## File Structure

```
├── clean_images.py
├── clean_tabular_data.py
├── FBMClassifier.py
├── FBMDataset.py
├── FBMTrainer.py
├── image_processor.py
├── sandbox.ipynb
├── package-lock.json
├── README.md
├── app
│   ├── api.py
│   ├── dockerfile
│   ├── image_decoder.pkl
│   ├── image_model.pt
│   ├── image_processor.py
│   ├── index.pkl
│   ├── requirements.txt
│   ├── training_data.csv
│   └── utils
│       ├── another.js
│       ├── constants.js
│       └── index.js
|
├── data
│   ├── Images.csv
│   ├── Products.csv
│   └── training_data.csv
|
├── final_model
│   └── image_model.pt
|
├── model_information
│   ├── image_decoder.pkl
│   ├── image_embeddings.json
│   └── index.pkl
|
├── my-app
│   ├── .gitignore
│   ├── package-lock.json
│   ├── package.json
│   ├── README.md
│   ├── public
│   |    ├── favicon.ico
│   |    ├── index.html
│   |    ├── manifest.json
│   |    └── robots.txt
|   |
|   └── src
│       ├── App.css
│       ├── App.js
│       ├── ImageCard.js
│       ├── index.css
│       ├── index.js
│       ├── SocialProfile.js
│       ├── theme.js
│       └── Upload.js  
```

## Explore the dataset

The success of a machine learning model directly correlates to the quality of the data that is provided. For the first step of the journey, exploratory data analysis and minimal data cleaning were performed. In order to perform this processing, a Jupyter notebook ```sandbox.ipynb``` was utilised. 

```clean_tabular_data.py``` and ```clean_images.py``` were created in order to perform the neccessary processing.

Firstly, the Products.csv was read into a pandas dataframe. Processing was performed by the function ```clean_product_data``` from the ```clean_tabular_data.py``` file. Here null values were removed and prices cast to numeric values.

Secondly, the required labels were extracted for classification. Two dictionaries, the encoder and decoder, were created as a means to convert between a string category and its numeric representation. This label was extracted by the category column of the Products.csv. 

The category column stored multiple categories in a string, all seperated by a ```/```. In order to extract our labels, the first category of the string was taken as the label. This data was then merged with the Images.csv, in order to determine the correct category for each image, and then stored in ```training_data.csv```.

Finally, the images provided were passed through the function ```clean_image_data``` inside of ```clean_images.py```. Here images were resized and forced to an RGB format and saved into a ```cleaned_images``` directory. This formatting was done in order to have consistent data for our model to use.

## Create the vision model, then turn it into a feature extraction model

Using the concept of a Convulutional Neural Network, a model was trained to classify the category of each product from their images and use it as a feature extraction model for indexing.

The process began by defining a custom PyTorch dataset ```FBMDataset```. This image dataset would contain the representations of all of the data we want to train/validate/test our models on. This would be mapped against a category of the image.

The model itself could have been created from scratch, but this would take a lot of time to finetune each layer to get the best result. Instead, transfer learning was used to utilise a pre trained model that solved a similar issue. Resnet 50 was the model of choice due to its success at classifying over 1000 categories. Much less than 13 categories of this problem.

In order to make this model a little more accurate for our specific problem, the architecture of the model needed to be updated slightly. All the weights of the exisiting model were frozen, so as to keep the pretrained weights of all the initial layers. Two new linear layers were added which diverged in output down to 13 neurons (the amount of labels we are training for). These two layers were unfrozen to allow the weights of the model to be optimised to our problem.

The optimiser chosen to improve the model was Stochastic Gradient Descent (SGD). The general process for SGD is as follows:

- Make a prediction
- Evaluate the loss
- Differentiate the loss with respect to model parameters
- Update the prarameters in the opposite diretion of the gradient

The loss function that was chosen was cross entropy. Cross entropy is commonly used for multiclass classifcation problems. The general idea is that the probability of a correct outcome should be pushed up, whereas the probability of an incorrect outcome should be pushed down. Cross entropy maximises the probability of a correct label, which by default pushed all other outcomes down. The training, validation and accuracy testing was all performed inside of ```FBMTrainer.py```. This process was the foundation of the training loop to train the ```FBMClassifier.py```.

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/Tensorboard.jpg?raw=true)

Although the weights of the model could be optimised, there are parameters that affect the accuracy of the model that can not. These are known as hyperparameters. The two main hyperparameters that were optimised for this model were the learning rate of the optimiser and the batch size of the datasets. Ray Tune helped to solve this problem. Ray Tune is a library that is designed to help optimise the process of hyperparameter tuning. By providing a range of options, Ray Tune is able to run multiple experiments on the training function and output the accuracy and loss of the model, ultimately providing you with a single set of hyperparameters that provided the best outcome. This allows for multiple concurrent experiments, as well as early stopping mechanisms (schedulers), so as to efficiently use resources. For this problem, Ray Tune helped to determine that a batch size of 8 and a learning rate around 1-e^1 worked the best (60% accuracy on test set).

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/tune_results.jpg?raw=true)

Once the model was trained to an acceptable degree, the architecture of the model was reverted to return 1000 outputs (feature extraction model). This output, referred to as our image embedding, would provide us with a reference to index inside of FAISS.

## Create the search index using FAISS

Facebook AI Similarity Search (FAISS) is a bespoke solution from Facebook to solve the issue of searching for multimedia documents that are similar to one another. It uses nearest-neighbour search implementations that can be utilised by billion scale data sets. For this reason, its a perfect solution to our problem.

Using the feature extraction model that was created previously, a dictionary was created to represent all of the images in our dataset. The key was the images filename and the value was the image embedding produced by the feature extraction model. This was added to a FAISS index and saved to a pkl file ```index.pkl```. This meant that the index could be recreated again with the exact same values.

## Configure and deploy the model serving API
Once all the components had been created, it was time to link them together to be consumed. Two endpoints were created using FastAPI and uvicorn. A GET endpoint to retrieve an image embedding from a single image and a POST to get the index of similar images from the one posted. These endpoint were containerised and stored inside of a docker image. 

Docker images were the perfect solution to deploy our application to our AWS EC2 instance. Ultimately, if this was a production application, we could scale the containers based on usage to provide load balancing. This would prevent users from getting a downturn in performance.

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/json_response.png?raw=true)

The JSON response provided from my API provides the user with the following four properties: Euclidian Distance (a measure of distance from the origial provided input), the index of the approximate nearest neighbour, the filename associated for that result and the category that the result falls under. This provided enough information to be consumed and displayed inside of a user interface.

## Consume the API and provide a user interface
The final component of the project was to provide a sufficient user interface in order to interact with our model. This interface would allow a user to provide an image, pass that image to our API and run it through our model, then provide the user with a graphical representation of images in our FAISS Index (along with suplimentary information).

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/laptop_result.png?raw=true)

The choice of frontend framework was React, due to its popularity and support. I was able to utilise Chakra UI in order to produce a useable interface without the complexity of creating my own components. A variety of react components were created such as ```ImageCard.js```, ```SocialProfile.js``` and ```Upload.js```. The latter being the main area the user would interact with when uploading the input for the API.

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/network.png?raw=true)


![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/network2.png?raw=true)

The process was to first ask for a user to upload an image. This would be the input for our API in order to find images similar to. The API responds with a JSON structure that provides more information on the related image and how it is stored inside of FAISS. Based on the result, the file name was used to query an S3 bucket which hosted all the images stored in our FAISS model. Once returned, the images were displayed alongside the rest of the information returned from the first API call.

## Improvements and lessons learned
Of course with a project of this magnitude, there was never going to be a complete replica of facebook's system. However, I do believe there are a lot of areas where I could have improved on. The best accuracy I was able to achieve after multiple rounds of experiments was 60%. The usage of Resnet50 greatly reduced the amount of time it would have taken to finetune a models structure, however, given the time I definitely would have implemented my own Convulutional Neural Network. This would mean that I would be able to train all layers of the CNN to improve on accuracy. Although 12k images is a sizeable dataset, I would have loved to have had access to even more data to train my model to improve its accuracy.

I believe the usage of RayTune was really beneficial to my experiment stage of this project. It allowed me to perform a multitude of experiments in order to test a variety of batch sizes and learning rates. With more time, I could experiment with different optimisers in order to improve accuracy. I was also constrained with the limitations of my hardware. Despite having access to CUDA, if I was to do this again, I would host my application in the cloud and utilise the larger amount of resoures and processing power. CUDA helped me to improve my epoch times from 22 minutes down to 2. With the help of the cloud, i'm certain this could improve even more.
