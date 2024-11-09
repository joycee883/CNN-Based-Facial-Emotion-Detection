# Import necessary libraries
from google.colab import drive
drive.mount('/content/drive')  # Mount Google Drive to access files

# Check GPU availability (optional for Colab environment)
!nvidia-smi  

# Import required libraries for CNN model building, image processing, and visualization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

# Load and display an example image
img = image.load_img('/content/drive/MyDrive/Colab Notebooks/Mood Classification CNN/Training/Not Happy/NH7.jpg')
plt.imshow(img)  # Display the image

# Read the same image with OpenCV to check dimensions and pixel values
i1 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Mood Classification CNN/Training/Not Happy/NH7.jpg')
print(i1.shape)  # Display the shape of the image (height, width, RGB channels)

# Image Data Generators for scaling and loading images
# The rescale parameter divides pixel values by 255 to normalize them to a 0-1 range
train = ImageDataGenerator(rescale=1.0 / 255)
validation = ImageDataGenerator(rescale=1.0 / 255)

# Define the training dataset
train_dataset = train.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/Mood Classification CNN/Training',
    target_size=(200, 200),  # Resize images to 200x200 pixels
    batch_size=32,           # Number of images per batch
    class_mode='binary'       # Binary classification (Happy or Not Happy)
)

# Define the validation dataset
validation_dataset = validation.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/Mood Classification CNN/Validation',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)

# Display the class indices to verify label mapping
print(train_dataset.class_indices)

# CNN Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),  # First Conv layer with 16 filters
    tf.keras.layers.MaxPooling2D(2, 2),                                                # First MaxPooling layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),                             # Second Conv layer with 32 filters
    tf.keras.layers.MaxPooling2D(2, 2),                                                # Second MaxPooling layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),                             # Third Conv layer with 64 filters
    tf.keras.layers.MaxPooling2D(2, 2),                                                # Third MaxPooling layer
    tf.keras.layers.Flatten(),                                                         # Flatten the output for fully connected layers
    tf.keras.layers.Dense(512, activation='relu'),                                     # Dense layer with 512 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')                                     # Output layer with sigmoid for binary classification
])

# Compile the model with binary crossentropy loss and RMSprop optimizer
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model
# Using the validation data to monitor accuracy and loss
model_fit = model.fit(train_dataset, validation_data=validation_dataset, epochs=8)

# Define the directory path for testing images
test_dir_path = '/content/drive/MyDrive/Colab Notebooks/Mood Classification CNN/Testing'

# Loop through each file in the test directory, display the image, and predict mood
for img_name in os.listdir(test_dir_path):
    img_path = os.path.join(test_dir_path, img_name)      # Construct full path for each image
    img = image.load_img(img_path, target_size=(200, 200))  # Load and resize the image
    plt.imshow(img)                                       # Display the image
    plt.show()

    # Preprocess the image for model prediction
    x = image.img_to_array(img)                           # Convert image to array
    x = np.expand_dims(x, axis=0)                         # Expand dimensions to fit model input
    images = np.vstack([x])                               # Stack the image for prediction batch

    # Make a prediction and output the result
    val = model.predict(images)
    if val < 0.5:  # Assuming 0.5 as the threshold for binary classification
        print('I am happy')
    else:
        print('I am not happy')

# Re-check GPU status after execution (optional)
!nvidia-smi  
