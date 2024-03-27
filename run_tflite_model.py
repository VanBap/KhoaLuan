from imutils.video import VideoStream
from imutils.video import FPS
from tensorflow import keras
import numpy as np
from keras.applications.resnet import preprocess_input
import tensorflow as tf
import argparse
import time
import cv2
import draw_label

if __name__ == "__main__":
    # Load model
    print("[INFO] loading model...")
    interpreter = tf.lite.Interpreter(model_path='F:/CODE_PYCHARM/KhoaLuan/saved_model/ResNet50_Weather_epoch20.tflite')

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Resize Tensor Shape
    interpreter.resize_tensor_input(input_details[0]['index'], (1, 244, 244, 3))
    #interpreter.resize_tensor_input(output_details[0]['index'], (1, 7))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Shape:", input_details[0]['shape'])
    print("Input Type:", input_details[0]['dtype'])
    print("Output Shape:", output_details[0]['shape'])
    print("Output Type:", output_details[0]['dtype'])

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    print("input details", input_details)
    print("output details", output_details)

    # Load labels from label.txt file
    label_file = "F:/CODE_PYCHARM/KhoaLuan/saved_model/label.txt"
    CATEGORIES = []

    with open(label_file, "r") as file:
        for line in file:
            category = line.strip()  # Remove any leading/trailing whitespace
            CATEGORIES.append(category)

    # Open the device at the ID 0
    # Use the camera ID based on
    # /dev/videoID needed
    cap = cv2.VideoCapture(0)

    # Check if camera was opened correctly
    if not (cap.isOpened()):
        print("Could not open video device")

    # Fetch one frame at a time from your camera
    while (True):
        # Frame is a numpy array, that you can predict on
        ret, frame = cap.read()

        # Resize frame to (224, 224)
        resized_frame = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(resized_frame, axis=0)

        # Preprocess input data if needed
        input_data = preprocess_input(input_data)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Perform inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process output data as needed
        predicted_weather = CATEGORIES[np.argmax(output_data)]


        #percentage = output_data.flatten()
        #print(percentage)
        #percentage = percentage[np.argmax(output_data)]

        output_data_float = tf.cast(output_data, dtype=tf.float32)
        softmax_output = tf.nn.softmax(output_data_float)
        print("===============================================")
        print("softmax_output la: ", softmax_output)
        predicted_class = np.argmax(softmax_output)
        print("===============================================")
        print("predicted_class la: ", predicted_class)
        predicted_probability = np.max(softmax_output)*100
        print(predicted_probability)
        # Adding the label on frame
        draw_label.__draw_label(frame, 'Label: {}  {:.2f}%'.format(predicted_weather, predicted_probability), (30, 30), (255, 255, 0))


        # Display the resulting frame
        cv2.imshow("preview", frame)

        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
