from picamera2 import Picamera2
import numpy as np
import tensorflow as tf
import picar_4wd as car
import cv2

# Initialize PiCamera2
picam2 = Picamera2()

# Configure the camera and start the preview
picam2.start_preview()

# Start camera
picam2.start()

def preprocess_image(image):
    # Assuming the model expects a 224x224 image
    processed_image = cv2.resize(image, (150, 150))
    # If the image has 4 channels, convert it to 3 channels (RGB)
    if processed_image.shape[-1] == 4:
        processed_image = processed_image[..., :3]
    processed_image = processed_image / 255.0  # Normalizing
    return processed_image

def control_car(action):
    # Ensure that action number correctly maps to the car's control commands
    if action == 0:  # Forward
        car.backward(50)  # Adjust speed as needed
    elif action == 1:  # Left
        car.turn_left(50)  # Adjust speed as needed
    elif action == 2:  # Right
        car.turn_right(50)  # Adjust speed as needed
    elif action == 3:  # Back
        car.stop()  # Adjust speed as needed
    elif action == 4:  # Stop or another action
        car.forward(50)

try:
    # Load a pre-trained model
    model = tf.keras.models.load_model('/home/amsozzer/picar-4wd/Picar_4wd/model_mine.keras')

    while True:
        # Capture an image
        image = picam2.capture_array()  # Capture an image array

        # Preprocess the image
        image = preprocess_image(image)

        # Expand dimensions as the model expects batches
        image_batch = np.expand_dims(image, axis=0)

        # Make a prediction
        predictions = model.predict(image_batch)
        predicted_action = np.argmax(predictions, axis=1)[0]

        print(f"Predictions: {predictions}, Predicted action: {predicted_action}")  # Print the predictions and action

        # Control the car based on the prediction
        control_car(predicted_action)

except Exception as e:
    print(e)
finally:
    car.stop()  # Ensure the car stops when the script ends
    picam2.stop_preview()  # Stop the camera preview
    picam2.stop()  # Stop the camera
