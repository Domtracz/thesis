import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import shutil
import random
from tabnanny import verbose
import pickle
import json
from tensorflow.keras.applications import ResNet50

# Paths to data
train_dir = 'images/train'
val_dir = 'images/test'
class_names = ["Angry", "Disgust", "Fear", "Happy","Neutral", "Sad", "Surprise" ]

def oversample_data(train_dir,target_samples,valid_image_extensions = ('.png', '.jpg', '.jpeg')):

    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir,class_dir)

        images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_image_extensions)]

        if not images:
            if verbose:
                print(f"Warning: No valid images found in class '{class_dir}'. Skipping oversampling.")
            continue

        n_samples = len(images)

        if n_samples < target_samples:
            if verbose:
                print(f"Class '{class_dir}': Oversampling from {n_samples} to {target_samples}.")
            for i in range(target_samples-n_samples):
                try:
                    image_to_copy = random.choice(images)
                    src = os.path.join(class_path , image_to_copy)
                    dst = os.path.join(class_path,f"{image_to_copy}_copy{i}.png")
                    shutil.copyfile(src, dst)

                except Exception as e:
                    if verbose:
                        print(f"Error copying file in class '{class_dir}': {e}")

oversample_data(train_dir,5000)

# Data Augmentation and Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Function to create a data generator
def create_data_generator(directory, datagen, batch_size=32, shuffle=True):
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

def se_block(input_tensor, reduction=16):
    """Squeeze and excitation block for channel attenuation"""
    filters = input_tensor.shape[-1] #Number of input Channels
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([input_tensor, se])


# Defines the residual block
def residual_block(input_tensor, filters):

    residual = input_tensor
    x = layers.BatchNormalization()(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)

    # Adjust the residual if necessary to match dimensions
    if input_tensor.shape[-1] != filters:
        residual = layers.Conv2D(filters, (1, 1), padding='same')(residual)

    x = layers.add([x, residual])

    return layers.ReLU()(x)

# Model definition
def create_model(use_resnet = False):
    input_layer = layers.Input(shape=(48, 48, 1))

    if use_resnet:
        x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_layer)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
        base_model.trainable = False
        x = base_model(x,training=False)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        output_layer = layers.Dense(7, activation='softmax')(x)

    else:
        x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = residual_block(x, 64)

        for filters in [128, 256, 512]:
            x = residual_block(x, filters)
            x = se_block(x, reduction=16)
            x = layers.Dropout(0.2)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)

        output_layer = layers.Dense(7, activation='softmax')(x)


    return models.Model(inputs=input_layer, outputs=output_layer)

def run_model( model, epochs=10, lr=2.5e-4, fine_tune=False):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_labels = train_generator.classes  # Get the labels from the train generator

    if fine_tune:
        for layer in model.layers:
            if 'resnet' in layer.name:
                layer.trainable = True
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001),
    ]

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

model_path = "Models/my_model_weights.keras"

baseline_model = create_model()  # Create a new model if no saved weights exist
history_my_model = run_model(baseline_model, epochs=30, fine_tune=False)
print("Created baseline model.")

tuned_model = create_model(use_resnet = True)  # Create a new model if no saved weights exist
history_tuned = run_model(tuned_model, epochs=30, fine_tune= True)
print("Created finetune model.")


#pickle and json files that store training history
history_file = 'training_history.pkl'
history_json_file = 'training_history.json'

if os.path.exists(history_file):
    with open(history_file, 'rb') as f:
        all_histories = pickle.load(f)
else:
    all_histories = []

all_histories.append({'baseline': history_my_model.history, 'fine_tuned': history_tuned.history})
with open(history_file, 'wb') as f:
    pickle.dump(all_histories, f)
with open(history_json_file, 'w') as f:
    json.dump(all_histories, f, indent=4)

# Plot training history
def plot_history_and_confusion_matrices(histories, labels, models):
    # Plot Training History
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history['accuracy'], label=f'{label} Train Accuracy')
        plt.plot(history['val_accuracy'], label=f'{label} Val Accuracy')
        plt.plot(history['loss'], label=f'{label} Train Loss')
        plt.plot(history['val_loss'], label=f'{label} Val Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()

    # Plot Confusion Matrices
    for model, label in zip(models, labels):
        # Generate predictions
        val_preds = np.argmax(model.predict(val_generator), axis=1)
        val_labels = val_generator.classes

        # Compute confusion matrix
        conf_matrix = tf.math.confusion_matrix(val_labels, val_preds)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {label}")
        plt.show()

plot_history_and_confusion_matrices(
    histories=[history_my_model.history, history_tuned.history],
    labels=['Baseline', 'Fine-Tuned'],
    models=[baseline_model, tuned_model]
)