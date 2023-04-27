import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Define the maximum number of landmarks
max_landmarks = 42

# Define the hands object outside the function so it can be reused
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def collect_data(label, num_samples, num_hands=1, show_landmarks=True):
    data = []
    cap = cv2.VideoCapture(0)

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to capture frame from camera")

        # Convert the image to RGB and process it with Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Extract the landmarks for one or two hands, depending on the input parameter
        if num_hands == 1:
            if results.multi_hand_landmarks:
                landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[0].landmark])
            else:
                landmarks = np.zeros((max_landmarks, 3))
        elif num_hands == 2:
            if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) >= 2:
                landmarks_1 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[0].landmark])
                landmarks_2 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[1].landmark])
                landmarks = np.concatenate((landmarks_1, landmarks_2), axis=0)
            else:
                landmarks = np.zeros((max_landmarks, 3))

        # Pad the landmarks array with zeros if necessary
        n_landmarks = landmarks.shape[0]
        if n_landmarks < max_landmarks:
            padding = np.zeros((max_landmarks - n_landmarks, 3))
            landmarks = np.concatenate((landmarks, padding), axis=0)
        elif n_landmarks > max_landmarks:
            landmarks = landmarks[:max_landmarks]

        # Add the landmarks and label to the data list
        data.append((landmarks, label))

        # Display the current frame with landmarks and bounding box
        if show_landmarks:
            # Draw hand landmarks on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the current frame
            cv2.imshow('frame', image)
        else:
            # Display the current frame without landmarks and bounding box
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()

    print(data)

    # Check if the file exists before loading
    if os.path.exists('./model/data.pickle'):
        with open('./model/data.pickle', 'rb') as f:
            existing_data = pickle.load(f)
    else:
        existing_data = []

    # Save the existing data to backup pickle file
    with open('./model/dataOld.pickle', 'wb') as f:
        pickle.dump(existing_data, f)

    # Append new data to the existing data
    existing_data.append(data)

    # Save the updated data back to the pickle file
    with open('./model/data.pickle', 'wb') as f:
        pickle.dump(existing_data, f)

    # Return the data
    return data