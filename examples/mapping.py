import time
import numpy as np
import tensorflow as tf
import picar_4wd as fc
from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start_preview()
picam2.start()

# Set the ultrasonic sensor to face straight forward


# Map parameters
map_size = 1000  # Size of the map
environment_map = np.zeros((map_size, map_size))  # Initialize the map with zeros

# Car's initial position and orientation
car_position = [map_size // 2, map_size // 2]  # Start in the center of the map
car_orientation = 0  # 0: up, 1: right, 2: down, 3: left
speed = 30

# Destination coordinates
destination = [500, 550]
#/home/amsozzer/Car-self_driving-correct/examples/my_model.keras
# Load TensorFlow model for object detection
model = tf.keras.models.load_model('/home/amsozzer/Car-self_driving-correct/examples/my_model.keras')

def preprocess_image(image):
    image = cv2.rotate(image,cv2.ROTATE_180)
    processed_image = cv2.resize(image, (150, 150))
    if processed_image.shape[-1] == 4:
        processed_image = processed_image[..., :3]
    processed_image = processed_image / 255.0
    return processed_image

def control_car(model_prediction, next_move, distance_to_obstacle):
    if model_prediction != 0:
        fc.stop()
        print("Model detected an object, stopping.")
        time.sleep(2)
        control_car_by_move(next_move, distance_to_obstacle)
    else:
        #fc.stop()
        #time.sleep(0.5)
        control_car_by_move(next_move, distance_to_obstacle)

def control_car_by_move(next_move, distance_to_obstacle):
    global car_orientation
    # If very close to an obstacle, reverse and turn
    if distance_to_obstacle is not None and distance_to_obstacle <= 1:  # example threshold, adjust as needed
        fc.backward(speed)
        time.sleep(1)  # reverse for a bit
        fc.turn_right(speed)  # or fc.turn_left(speed) 
        car_orientation = (car_orientation + 1) % 4  # Update orientation to right
        time.sleep(1)  # turn for a bit
    else:
        if next_move == 'up': 
            fc.forward(speed)
        elif next_move == 'left':
            fc.turn_left(speed)
            car_orientation = (car_orientation + 3) % 4  # Update orientation to left
        elif next_move == 'right':
            fc.turn_right(speed)
            car_orientation = (car_orientation + 1) % 4  # Update orientation to right
        elif next_move == 'down':
            fc.forward(speed)
        print(f"Moving {next_move}, Orientation: {car_orientation}")

def update_map(scan_list, car_position, car_orientation):
    global environment_map
    front_half_start = len(scan_list) // 4
    front_half_end = len(scan_list) * 3 // 4
    front_scan_list = scan_list[front_half_start:front_half_end]
    print(scan_list)

    closest_distance = float('inf')  # Initialize with a large number
    for i, distance in enumerate(front_scan_list):
        angle = 0  # Keep the sensor facing forward
        if distance < closest_distance:
            closest_distance = distance

        if distance <= 1:  # If distance indicates an obstacle is close
            x_offset = distance * np.cos(np.radians(angle))
            y_offset = distance * np.sin(np.radians(angle))
            obstacle_position = [int(car_position[0] + x_offset), int(car_position[1] + y_offset)]

            if 0 <= obstacle_position[0] < map_size and 0 <= obstacle_position[1] < map_size:
                environment_map[obstacle_position[1], obstacle_position[0]] = 1

                print(f"Updating map at position: {obstacle_position}")
            else:
                print(f"Obstacle position {obstacle_position} out of map bounds")
    return closest_distance  # Return the distance to the closest obstacle

def get_next_move(current_position, destination, environment_map):
    # A simple approach to move closer to the destination in each step
    next_move = None
    if current_position[0] < destination[0] and environment_map[current_position[1]][current_position[0] + 1] == 0:
        next_move = 'right'
    elif current_position[0] > destination[0] and environment_map[current_position[1]][current_position[0] - 1] == 0:
        next_move = 'left'
    elif current_position[1] < destination[1] and environment_map[current_position[1] + 1][current_position[0]] == 0:
        next_move = 'down'
    elif current_position[1] > destination[1] and environment_map[current_position[1] - 1][current_position[0]] == 0:
        next_move = 'up'
    return next_move

def update_orientation(car_orientation, next_move):
    # Update car orientation based on the next move
    if next_move == 'left':
        car_orientation = (car_orientation + 3) % 4  # Turn left
    elif next_move == 'right':
        car_orientation = (car_orientation + 1) % 4  # Turn right
    return car_orientation

def update_position(car_position, car_orientation):
    # Update car position based on its orientation
    if car_orientation == 0:  # up
        car_position[1] += 1
    elif car_orientation == 1:  # right
        car_position[0] += 1
    elif car_orientation == 2:  # down
        car_position[1] += 1
    elif car_orientation == 3:  # left
        car_position[0] -= 1
    return car_position

def main():
    global car_position, car_orientation

    try:
        while True:
            # Check if destination is reached
            if car_position == destination:
                print("Destination reached!")
                break

            # Update the map with the new scan data
            scan_list = fc.scan_step(35)
            if not scan_list:
                continue

            closest_obstacle_distance = update_map(scan_list, car_position, car_orientation)

            # Capture an image
            image = picam2.capture_array()
            image = preprocess_image(image)
            image_batch = np.expand_dims(image, axis=0)

            # Make a prediction
            predictions = model.predict(image_batch)
            model_prediction = np.argmax(predictions, axis=1)[0]  # Adjusted based on your model
            print(f"Predictions: {predictions}, Model prediction: {model_prediction}")

            # Determine the next move based on the current position, destination, and environment map
            next_move = get_next_move(car_position, destination, environment_map)

            # Control the car based on the model prediction, next move, and closest obstacle distance
            control_car(model_prediction, next_move, closest_obstacle_distance)

            # Update car's position and orientation based on the next move
            car_orientation = update_orientation(car_orientation, next_move)
            car_position = update_position(car_position, car_orientation)

            # Print the map and car's current position and orientation for debugging
            print(f"Next move: {next_move}, Current position: {car_position}, Destination: {destination}")
            print(f"Orientation: {car_orientation}")
            print(environment_map)

    except Exception as e:
        print(e)
    finally:
        fc.stop()  # Ensure the car stops when the script is interrupted
        picam2.stop_preview()  # Stop the camera preview
        picam2.stop()  # Stop the camera

if __name__ == "__main__":
    try: 
        main()
    finally:
        fc.stop()  # Ensure the car stops when the script ends
