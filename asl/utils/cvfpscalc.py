# Imports
from collections import deque
import cv2 as cv

class CvFpsCalc(object):
    # Object for tracking fps of camera to measure proformance
    def __init__(self, buffer_len=1):
        """
        The function initializes the class by setting the start time to the current time, the frequency
        to the number of ticks per second, and the deque to the length of the buffer.
        
        :param buffer_len: The number of previous times to keep in memory, defaults to 1 (optional)
        """
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    # Getters
    def get(self):
        """
        Function calculates the time difference between the current tick and the start tick, and then appends
        that difference to a list of differences. 
        
        Then, it calculates the average of all the differences in the list, and divides 1000 by that
        average to get the FPS. 
        
        Finally, it rounds the FPS to 2 decimal places and returns it. 
        """
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
