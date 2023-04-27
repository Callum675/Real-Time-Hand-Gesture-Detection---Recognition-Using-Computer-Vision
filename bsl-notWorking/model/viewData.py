import pickle

# Open the pickle file in read mode
with open('data.pickle', 'rb') as f:
    # Load the contents of the file into a Python object
    data = pickle.load(f)

# View the data
print(data)