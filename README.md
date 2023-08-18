# Facebook Marketplace Recommendation Ranking System

Facebook Marketplace is a platform for buying and selling products on Facebook. This is an implementation of the system behind the marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

# Explore the dataset

The success of a machine learning model directly correlates to the quality of the data that is provided. For the first step of the journey, exploratory data analysis and minimal data cleaning were performed. In order to perform this processing, a Jupyter notebook "sandbox.ipynb" was utilised. 

"clean_tabular_data.py" and "clean_images.py" were created in order to perform the neccessary processing.

Firstly, the Products.csv was read into a pandas dataframe. Processing was performed by the function "clean_product_data" from the "clean_tabular_data.py" file. Here null values were removed and prices cast to numeric values.

Secondly, the required labels were extracted for classification. Two dictionaries, the encoder and decoder, were created as a means to convert between a string category and its numeric representation. This label was extracted by the category column of the Products.csv. 

The category column stored multiple categories in a string, all seperated by a "/". In order to extract our labels, the first category of the string was taken as the label. This data was then merged with the Images.csv, in order to determine the correct category for each image, and then stored in "training_data.csv".

Finally, the images provided were passed through the function "clean_image_data" inside of "clean_images.py". Here images were resized and forced to an RGB format and saved into a "cleaned_images" directory. This formatting was done in order to have consistent data for our model to use.

# Create the vision model, then turn it into a feature extraction model

Using the concept of a Convulutional Neural Network, a model is trained to classify the category of each product from their images and use it as a feature extraction model for indexing.

The process began by defining a custom PyTorch dataset "FBMDataset". This image dataset would contain the representations of all of the data we want to train/validate/test our models on. This would be mapped against a category of the image.

The model itself could have been created from scratch, but this would take a lot of time to finetune each layer to get the best result. Instead, transfer learning was used to utilise a pre trained model that solved a similar issue. Resnet 50 was the model of choice. The resnet50 model is a pretrained model used for multiclass classifcation of the imagenet dataset. The model is trained for 1000 labels, so definitely could handle our scenario with only 13 labels. 

In order to make this model a little more accurate for our specific problem, the architecture of the model needed to be updated slightly. All the weights of the exisiting model were frozen, so as to keep the pretrained weights of all the initial layers. Two new linear layers were added which diverged in output down to 13 neurons (the amount of labels we are training for). These two layers were unfrozen to allow the weights of the model to be optimised to our problem.

The optimiser chosen to improve the model was Stochastic Gradient Descent (SGD). The general process for SGD is as follows:

- Make a prediction
- Evaluaton the loss
- Differentiate the loss with respect to model parameters
- Update the prarameters in the opposite diretion of the gradient

The loss function that was chosen was cross entropy. Cross entropy is commonly used for multiclass classifcation problems. The general idea is that the probability of a correct outcome should be pushed up, whereas the probability of an incorrect outcome should be pushed down. Cross entropy maximises the probability of a correct label, which by default pushed all other outcomes down.

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/Tensorboard.jpg?raw=true)

This process was the foundation of the training loop to train the FBMClassifier. Although the weights of the model could be optimised, the hyperparamters of the model could not be. The two main hyperparameters being the learning rate of the optimiser and the batch size of the datasets. Ray Tune helped to solve this problem. By providing a range of options, Ray Tune was able to run multiple experiments on the training function and output the accuracy and loss of the model, ultimately providing you with a single set of hyperparameters that provided the best outcome. 


![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/tune_results.jpg?raw=true)

Once the model was trained to an acceptable degree, the architecture of the model was reverted to return 1000 neurons. This output, referred to as our image embedding, would provide us with a reference to index inside of FAISS.

# Create the search index using FAISS

Facebook AI Similarity Search (FAISS) is a bespoke solution from Facebook to solve the issue of searching for multimedia documents that are similar to one another. It uses nearest-neighbour search implementations that can be utilised by billion scale data sets. For this reason, its a perfect solution to our problem.

Using the feature extraction model that was created previously, a dictionary was created to represent all of the images in our dataset. The key was the images filename and the value was the image embedding produced by the feature extraction model. This was added to a FAISS index and saved to a pkl file "index.pkl". This meant that the index could be recreated again with the exact same values.

# Configure and deploy the model serving API
Once all the components had been created, it was time to link them together to be consumed. Two endpoints were created using FastAPI and uvicorn. A GET endpoint to retrieve an image embedding from a single image and a POST to get the index of similar images from the one posted. These endpoint were containerised and stored inside of a docker image. 

![alt text](https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/API.jpg?raw=true)

Docker images were the perfect solution to deploy our application to our AWS EC2 instance. Ultimately, if this was a production application, we could scale the containers based on usage to provide load balancing. This would prevent users from getting a downturn in performance.