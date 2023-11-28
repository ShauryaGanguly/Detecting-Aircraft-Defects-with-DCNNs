import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.models import load_model

'''
Model Testing
'''
model = load_model('crack_classifier.h5')


def classify(test_image_path, true_class):
    # Load and preprocess the test image
    test_image = image.load_img(test_image_path, target_size=(100, 100))
    test_image_array = image.img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)  # Add an extra dimension for batch size
    test_image_array /= 255.0  # Normalize pixel values to [0, 1]
    
    # Make predictions
    predictions = model.predict(test_image_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    
    # Map predicted class index to class label
    class_labels = ['Small', 'Medium', 'Large', 'No']
    predicted_label = class_labels[predicted_class]
    
    # Display the result 
    plt.imshow(test_image)
    plt.axis('off')

    # Display the predicted label and confidence percentage on the image
    plt.text(1, 10, f"True Crack Classification Label: {true_class}\nPredicted Crack Classification Label: {predicted_label}\n", color='white', bbox=dict(facecolor='black', alpha=0.8))
   
    for i, class_label in enumerate(class_labels):
        confidence = predictions[0][i] * 100
        plt.text(5, 77 + i * 6, f"{class_label} Crack: {confidence:.2f}%", color='green', fontsize=10)

    # Show the image
    plt.show()
    return 

classify('Data/Test/Medium/Crack__20180419_06_19_09,915.bmp', 'Medium')
classify('Data/Test/Large/Crack__20180419_13_29_14,846.bmp', 'Large')
