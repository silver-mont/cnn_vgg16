import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
model = VGG16(weights='imagenet', include_top=False)
model.summary() 

img_path = '/content/car_image.jpg'  # Make sure to upload a file to this path

img = image.load_img(img_path, target_size=(224, 224))  # Resize image to fit model input
img_array = image.img_to_array(img)  # Convert img to array
img_array_expanded_dims = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_array_expanded_dims)

# We create a model that will return these outputs, given the model input
layer_outputs = [layer.output for layer in model.layers[1:]]  # Exclude the first layer to match the input size
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_preprocessed)

# Function to display the activations of the layers
def display_activation(activations, col_size, row_size, layer_index): 
    activation = activations[layer_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

# Displaying the 9th layer , VGG will have 16 
display_activation(activations, 8, 8, 9)  # Change the layer index as needed
