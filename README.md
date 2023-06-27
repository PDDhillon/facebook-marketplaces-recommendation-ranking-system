# Facebook Marketplace Recommendation Ranking System

Facebook Marketplace is a platform for buying and selling products on Facebook. This is an implementation of the system behind the marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

# Explore the dataset

The success of a machine learning model directly correlates to the quality of the data that is provided. For the first step of the journey, exploratory data analysis and minimal data cleaning were performed. In order to perform this processing, a Jupyter notebook "sandbox.ipynb" was utilised. 

"clean_tabular_data.py" and "clean_images.py" were created in order to perform the neccessary processing.

Firstly, the Products.csv was read into a pandas dataframe. Processing was performed by the function "clean_product_data" from the "clean_tabular_data.py" file. Here null values were removed and prices cast to numeric values.

Secondly, the required labels were extracted for classification. Two dictionaries, the encoder and decoder, were created as a means to convert between a string category and its numeric representation. This label was extracted by the category column of the Products.csv. 

The category column stored multiple categories in a string, all seperated by a "/". In order to extract our labels, the first category of the string was taken as the label. This data was then merged with the Images.csv, in order to determine the correct category for each image, and then stored in "training_data.csv".

Finally, the images provided were passed through the function "clean_image_data" inside of "clean_images.py". Here images were resized and forced to an RGB format and saved into a "cleaned_images" directory. This formatting was done in order to have consistent data for our model to use.