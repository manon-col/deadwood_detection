# -*- coding: utf-8 -*-
"""

@author: manon-col
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


directory = 'CNN_data/07_04' # location of image data to train the model

# Generate dataset

image_size = (217, 217) # dimension of image in pixels
num_classes = 2 # number of classes in the dataset
batch_size = 32

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

# Image data augmentation : apply random transformations to artificially
# introduce samples

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        
    ]
)

# # Dataset with augmented data gen
train_ds_augmented = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Create a convolutional base

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(image_size[0], image_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
epochs = 50  # You can adjust the number of epochs as needed
history = model.fit(train_ds_augmented,
                    epochs=epochs,
                    validation_data=val_ds)

# Display learning curves
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Model evaluation
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(test_acc)