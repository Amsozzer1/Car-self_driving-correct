import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 
# Set the directory where your data is located
base_dir = '/home/amsozzer/DATA'

# Directories for your training, validation, test data
train_dir = os.path.join(base_dir, 'Train')  # Ensure you have a train directory with subfolders for each class
validation_dir = base_dir+"/Val"  # Ensure you have a validation directory with subfolders for each class

total_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
total_validation_samples = sum([len(files) for r, d, files in os.walk(validation_dir)])
# Set batch size
batch_size = 20

# Image Data Generator with Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Image Data Generator for Validation (No Augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Configure the Train and Validation Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'  # 'binary' for binary classification
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),














d    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # 1 unit for binary classification
])

# Compile the Model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=total_train_samples // batch_size,  # Ensure proper step size
    epochs=30,
    validation_data=validation_generator,
    validation_steps=total_validation_samples // batch_size  # Ensure proper step size
)

# Model Evaluation
model.evaluate(validation_generator)
try:
    model.save('
    
    /home/amsozzer/Car-self_driving-correct/examples/my_model.keras')
except Exception as e:
    print(f"Error saving model: {e}")

#model.save('my_model.keras')
