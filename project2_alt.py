import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image
from keras.preprocessing.image import ImageDataGenerator

'''
Data Processing
'''

# Define the input image shape
input_shape = (100, 100, 3)

# Define the directories for train, validation, and test data
train_dir = 'Data/Train/'
validation_dir = 'Data/Validation/'
test_dir = 'Data/Test/'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values to [0, 1]
    shear_range=0.2,    # Shear transformation
    zoom_range=0.2,     # Zoom transformation
    horizontal_flip=True
)

# Data augmentation for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Batch size for training and validation generators
batch_size = 32

# Create the train and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

'''
Neural Network Architechture Design & Hyperparameter Analysis
'''

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Create a sequential model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to transition from convolutional layers to dense layers
model.add(Flatten())

# Dense layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to reduce overfitting

model.add(Dense(4, activation='softmax'))  # 4 neurons for 4 classes and softmax activation for multiclass classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#save model as file to export and use for part 5
model.save('crack_classifier.h5')

# Display the model summary
model.summary()

'''
Model Evaluation
'''

# Set the number of epochs
epochs = 5

# Train the model and store the training history
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')

# Show plots
plt.show()