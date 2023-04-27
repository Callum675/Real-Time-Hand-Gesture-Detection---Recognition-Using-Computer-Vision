# Load the RNN model
model = tf.keras.models.load_model("rnn_model.h5")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Define the classes and labels
classes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
labels = list(range(len(classes)))

# Define the canvas for displaying the video feed
canvas = Canvas(root, width=640, height=480)
canvas.pack(side=TOP)

 # Define the label for displaying the recognized gesture
 label = Label(root, font=("Helvetica", 24),
                bg="white", padx=10, pady=10)
  label.pack(side=BOTTOM, fill=X)

   # Define the function for processing the video feed
   def process_video():
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture frame from camera")

            # Convert the image to RGB and process it with Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Extract the landmarks for one or two hands, depending on the input parameter
            num_hands = len(results.multi_hand_landmarks)
            if num_hands == 1:
                landmarks = np.array(
                    [[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[0].landmark])
            elif num_hands == 2:
                landmarks_1 = np.array(
                    [[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[0].landmark])
                landmarks_2 = np.array(
                    [[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[1].landmark])
                landmarks = np.concatenate(
                    (landmarks_1, landmarks_2), axis=0)
            else:
                landmarks = np.zeros((42, 3))

            # Preprocess the landmarks and classify the gesture
            landmarks = np.expand_dims(landmarks, axis=0)
            landmarks = (landmarks - landmarks.mean()) / landmarks.std()
            logits = model.predict(landmarks)
            prediction = classes[np.argmax(logits)]
            label.config(text=prediction)

            # Draw hand landmarks on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the current frame
            img = cv2.resize(image, (640, 480))
            img = cv2.cvtColor(img, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('frame', image)

            # Update the previous landmarks for movement tracking
            prev_landmarks = landmarks

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()
