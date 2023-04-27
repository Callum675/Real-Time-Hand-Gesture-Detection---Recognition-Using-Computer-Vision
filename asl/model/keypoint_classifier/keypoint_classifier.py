# loads the model, sets up the interpreter, and then setups calls to the model with the input data
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    # Main Model Object
    def __init__(self, 
        model_path='model/keypoint_classifier/keypoint_classifier.tflite', 
        num_threads=1,):
        """
        Loads the model and sets up the interpreter
        
        :param model_path: The path to the model file, defaults to
        model/keypoint_classifier/keypoint_classifier.tflite (optional)
        :param num_threads: The number of threads to use for running the inference, defaults to 1 (optional)
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        """
        Takes in a list of landmarks, and returns a number, which corresponds
        to the gesture that the model thinks the hand is expressing
        
        :param landmark_list: a list of 68 facial landmarks
        :return: The index of the highest probability of the result.
        """
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
