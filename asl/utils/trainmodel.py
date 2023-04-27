import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm


def trainModel():

    # Get the absolute path of the file
    file_path = os.path.abspath('keypoint_classification.ipynb')

    # Load the notebook
    with open(file_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Create the preprocessor
    ep = ExecutePreprocessor(timeout=-1)

    # Run the notebook
    ep.preprocess(nb, {'metadata': {'path': '.'}})

    # Save the notebook with the executed code
    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
