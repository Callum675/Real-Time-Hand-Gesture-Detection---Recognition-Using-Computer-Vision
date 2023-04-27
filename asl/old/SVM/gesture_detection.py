#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2
import numpy as np
import pandas as pd
import os
import pickle
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from utils import CvFpsCalc

# initialising mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load model using pickle
with open('modelSVM_Linear.pkl', 'rb') as f:
    model_svm = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0  
while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            #prediction = model.predict([landmarks])
            # print(prediction)
            #classID = np.argmax(prediction)
            #className = classNames[classID]        

    # show the prediction on the frame using cv2.putText() method
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 50)
    fontScale = 2
    colour = (0, 0, 255) # Blue color in BGR
    thickness = 3 # Line thickness
    #frame = cv2.putText(frame, str(className[0]), org, font, fontScale, colour, thickness, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()