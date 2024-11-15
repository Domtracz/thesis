import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as snss
import os

# Paths to data
train_dir = 'images/train'
val_dir = 'images/test'
class_names = ["Angry", "Disgust", "Fear", "Happy","Neutral", "Sad", "Surprise" ]

# Data Augmentation and Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Function to create a data generator
def create_data_generator(directory, datagen, batch_size=64, shuffle=True):
    return datagen.flow_from_directory(
        directory,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=shuffle
    )

# Creates train and validation generators
train_generator = create_data_generator(train_dir, train_datagen)
val_generator = create_data_generator(val_dir, val_datagen, shuffle=False)

# Learning rate scheduler function
def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1) if epoch >= 10 else lr

# Defines the residual block
def residual_block(x, filters):
    residual = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(layers.add([x, residual]))

# Model definition
def create_model():
    input_layer = layers.Input(shape=(48, 48, 1))
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 256)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(7, activation='softmax')(x)
    return models.Model(inputs=input_layer, outputs=output_layer)

# Checks if model weights exist and load them
model_path = "Models/best_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)  # Load the model if weights are saved
    print("Loaded saved model weights.")
else:
    model = create_model()  # Create a new model if no saved weights exist
    print("Created new model.")

# Compiles the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min"),
    LearningRateScheduler(scheduler)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks
)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.title('Training History')
plt.show()

# Evaluate the model on the validation data
val_preds = np.argmax(model.predict(val_generator), axis=1)
val_labels = val_generator.classes
accuracy = np.mean(val_preds == val_labels)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = tf.math.confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(8, 6))
snss.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
