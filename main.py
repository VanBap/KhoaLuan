from imutils.video import VideoStream
from imutils.video import FPS
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow as tf
import argparse
import imutils
import time
import cv2
from keras.models import load_model
import draw_label




if __name__ == "__main__":
    # load model
    print("[INFO] loading model...")
    model = tf.keras.models.load_model('F:/saved_model/ResNet50_Weather_epoch20.h5')

    # labels array
    CATEGORIES=['CLOUDY','FOG','RAINY','SANDY','SHINE','SNOWY', 'SUNRISE' ]



    # Open the device at the ID 0
    # Use the camera ID based on
    # /dev/videoID needed
    cap = cv2.VideoCapture(0)

    # Check if camera was opened correctly
    if not (cap.isOpened()):
        print("Could not open video device")

    # 2) fetch one frame at a time from your camera
    while (True):
        # frame is a numpy array, that you can predict on
        ret, frame = cap.read()
        print("=============== 2 =================")

        # 3) obtain the prediction
        # Resize frame thành kích thước (224, 224)
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame_expanded = np.expand_dims(resized_frame, axis=0)
        prediction = model(resized_frame_expanded, training=False)

        print("=============== 3 =================")
        predicted_weather = CATEGORIES[np.argmax(prediction)]

        percentage = prediction.numpy()
        percentage = percentage.flatten()
        print(percentage)
        percentage = percentage[np.argmax(percentage)]*100
        print("$$$$$$$$$$$$$$$$$$$")
        # 4) Adding the label on frame
        draw_label.__draw_label(frame, 'Label: {} {:.2f}%'.format(predicted_weather, percentage), (30, 30), (255, 255, 0))
        print("=============== 4 =================")

        # 5) Display the resulting frame
        cv2.imshow("preview", frame)
        print("=============== 5 =================")
        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()