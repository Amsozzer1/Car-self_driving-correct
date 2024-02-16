from picamera2 import Picamera2
import cv2
import tensorflow as tf
import numpy as np
import picar_4wd as car

# Initialize PiCamera2 and Picar-4WD
camera = Picamera2()


def preprocess_image(image):
    # Assuming the model expects a 224x224 image
    processed_image = cv2.resize(image, (224, 224))
    processed_image = processed_image / 255.0  # Normalizing
    return processed_image

def control_car(action):
    if action == 0:  # Forward
        car.forward(speed=50)  # Adjust speed as needed
    elif action == 1:  # Left
        car.left(speed=50)  # Adjust speed as needed
    elif action == 2:  # Right
        car.right(speed=50)  # Adjust speed as needed
    elif action == 3:  # Stop or another action
        car.stop()

try:
    # Start the camera preview (optional, for testing)
    camera.start_preview()
    
    # Load a pre-trained model
    model = tf.keras.models.load_model('/home/amsozzer/picar-4wd/Picar_4wd/model_mine.h5')

    while True:
        # Capture an image
        image = camera.capture_array()

        # Preprocess the image
        image = preprocess_image(image)

        # Expand dimensions as the model expects batches
        image_batch = np.expand_dims(image, axis=0)

        # Make a prediction
        predictions = model.predict(image_batch)
        predicted_action = np.argmax(predictions, axis=1)[0]

        # Control the car based on the prediction
        control_car(predicted_action)
        
except Exception as e:
    print(e)
finally:
    car.stop()  # Ensure the car stops when the script ends
    camera.stop_preview()  # Stop the camera preview
