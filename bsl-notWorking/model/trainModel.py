import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the data from the pickle file
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

# initialize arrays to store landmarks and labels
landmarks = []
labels = []

# loop through data and extract landmarks and labels
for datapoint in data:
    # extract data matrix and label from tuple
    data_matrix, label = datapoint
    
    # extract landmarks from data matrix
    landmark_indices = [0, 5, 10, 15, 20, 25]  # indices of landmarks in data matrix
    landmarks_matrix = data_matrix[landmark_indices, :2]  # only keep x,y coordinates of landmarks
    
    # append landmarks and label to arrays
    landmarks.append(landmarks_matrix)
    labels.append(label)

# convert landmarks and labels to numpy arrays
landmarks = np.array(landmarks)
print(landmarks)
labels = np.array(labels)
print(labels)

# print shapes of landmarks and labels arrays
print('Landmarks shape:', landmarks.shape)
print('Labels shape:', labels.shape)

# Encode the labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# One-hot encode the labels
onehot_encoder = OneHotEncoder(sparse=False)
labels_onehot = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    landmarks, labels_onehot, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(
    X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(labels_onehot.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=50, batch_size=64)

# Evaluate the model on the validation set
score, accuracy = model.evaluate(X_val, y_val, batch_size=64)
print('Validation accuracy:', accuracy)
