# Import Required Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import datetime
import os
import sys

# Global Path Variables
DATASET_PATH = "dataset/"  # Training dataset folder
OUTPUT_PATH = "output/"    # Output folder for the trained model
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoint.weights.h5")  # Checkpoint file path

# Ensure output folder exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Global Training Parameters
IMG_SIZE = 224             # Image size for the model input
BATCH_SIZE = 64            # Batch size for training
EPOCHS = 20                # Number of epochs to train
NUM_CLASSES = 4            # Number of Tesla car models
DROPOUT_RATE = 0.5         # Dropout rate to reduce overfitting
LEARNING_RATE = 1e-4       # Initial learning rate for optimizer

# Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 20% of data for validation
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Load Training and Validation Data
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Check Class Indices
print("Class Indices:", train_generator.class_indices)

# Build Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add Custom Layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(DROPOUT_RATE)(x)  # Use global dropout rate
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model using global learning rate
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Define Async Live Plot for Loss and Accuracy
class LivePlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

        # Setup plot for real-time update
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def on_epoch_end(self, epoch, logs=None):
        # Append logs to track progress
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))

        # Clear previous data on the plot
        self.ax1.cla()
        self.ax2.cla()

        # Plot Loss
        self.ax1.plot(self.losses, label="Training Loss")
        self.ax1.plot(self.val_losses, label="Validation Loss")
        self.ax1.legend()
        self.ax1.set_title("Loss over Epochs")

        # Plot Accuracy
        self.ax2.plot(self.accuracy, label="Training Accuracy")
        self.ax2.plot(self.val_accuracy, label="Validation Accuracy")
        self.ax2.legend()
        self.ax2.set_title("Accuracy over Epochs")

        # Update the plots
        plt.pause(0.1)  # Add a short delay to make the update smoother

    def on_train_end(self, logs=None):
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot

# Prepare Callbacks
live_plot_callback = LivePlotCallback()
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, save_weights_only=True)

# Check if checkpoint exists, if so load it
initial_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print("Checkpoint found. Loading weights from checkpoint.")
    model.load_weights(CHECKPOINT_PATH)
    if os.path.exists(os.path.join(OUTPUT_PATH, "epoch.txt")):
        with open(os.path.join(OUTPUT_PATH, "epoch.txt"), "r") as f:
            initial_epoch = int(f.read())
        print(f"Resuming from epoch {initial_epoch + 1}")



# Train the Model with Callbacks
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[live_plot_callback, checkpoint_callback]
)

# Save the final model
model_save_path = os.path.join(OUTPUT_PATH, 'tesla_model_classifier.h5')
model.save(model_save_path)
print(f"Final model saved as {model_save_path}")

# Save the last completed epoch to a file
with open(os.path.join(OUTPUT_PATH, "epoch.txt"), "w") as f:
    f.write(str(EPOCHS))

# Print completion message and exit program
print("Training complete. Program will now close.")
sys.exit()  # Close the program
