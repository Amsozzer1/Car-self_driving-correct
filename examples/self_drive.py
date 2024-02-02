import time
from picamera2 import Picamera2, Preview
import picar_4wd as fc

import tensorflow as tf
import cv2
picam = Picamera2()
def model():
    
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)




def turn_camer_on():
    picam.start_preview(Preview.QTGL)
    picam.start()
    
    
def turn_camer_off():
    picam.close()
    
def take_pic():
    picam.capture_file("test-python.jpg")

def main():
    
    config = picam.create_preview_configuration()
    picam.configure(config)
    turn_camer_on()
    fc.forward(2)
    time.sleep(2)
    turn_camer_off()
    #model()

main()
