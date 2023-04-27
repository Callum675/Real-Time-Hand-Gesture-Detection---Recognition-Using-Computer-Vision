import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

import tkinter as tk

from utils import CvFpsCalc
from utils import collect_data


class HomePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.home_label = tk.Label(
            self, text="Welcome to the Home Page", font=("Arial", 20))
        self.home_label.pack(pady=50)


class TrainPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.about_label = tk.Label(
            self, text="Add Train Data", font=("Arial", 20))
        self.about_label.pack(pady=30)

        # Create the label and entry for the number of samples to collect
        self.samples_label = tk.Label(
            self, text="Number of samples:").pack(pady=10)
        self.num_samples_entry = tk.Entry(self)
        self.num_samples_entry.pack(pady=10)

        # Create the label and entry for the label to assign to the data
        self.class_Label = tk.Label(self, text="Label:").pack(pady=10)
        self.class_entry = tk.Entry(self)
        self.class_entry.pack(pady=10)

        # Define the number of hands options
        num_hands_options = [1, 2]

        # Create a variable to store the selected option for number of hands
        global var_num_hands
        var_num_hands = tk.StringVar(parent)
        var_num_hands.set(num_hands_options[0])

        # Create a label for the options menu
        self.label_num_hands = tk.Label(
            self, text="Number of hands:").pack(pady=10)
        self.option_menu_num_hands = tk.OptionMenu(
            self, var_num_hands, *num_hands_options).pack(pady=10)

        # Create the button to start collecting data
        self.start_Button = tk.Button(
            self, text="Start", command=self.collect_data).pack(pady=10)

        # Create the video capture object
        self.cap = cv2.VideoCapture(0)

        # Create the Mediapipe hands object
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

        # Create the previous landmarks variable for movement tracking
        self.prev_landmarks = None

    def collect_data(self):
        # Get the number of samples to collect and the label to assign to the data
        num_samples = int(self.num_samples_entry.get())
        label = self.class_entry.get()
        num_hands = int(var_num_hands.get())

        # Collect the data using the collect_data() function
        data = collect_data(label, num_samples, num_hands, show_landmarks=True)

        # Display a message box indicating that data collection is complete
        tk.messagebox.showinfo(
            "Data Collection", "Data collection is complete.")


class ContactPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.contact_label = tk.Label(
            self, text="Contact us at contact@example.com", font=("Arial", 20))
        self.contact_label.pack(pady=50)


class EdirGestureGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.data = None
        self.filtered_data = None
        self.current_class = tk.StringVar()
        self.new_class = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        # Load data button
        load_data_button = tk.Button(
            self.master, text="Load Data", command=self.load_data)
        load_data_button.pack(pady=10)

        # Filter data button
        filter_data_button = tk.Button(
            self.master, text="Filter Data", command=self.filter_data)
        filter_data_button.pack(pady=10)

        # Update class button
        update_class_button = tk.Button(
            self.master, text="Update Class", command=self.update_class)
        update_class_button.pack(pady=10)

        # Current class label and entry
        current_class_label = tk.Label(self.master, text="Current Class:")
        current_class_label.pack()
        current_class_entry = tk.Entry(
            self.master, textvariable=self.current_class, state="readonly")
        current_class_entry.pack(pady=5)

        # New class label and entry
        new_class_label = tk.Label(self.master, text="New Class:")
        new_class_label.pack()
        new_class_entry = tk.Entry(self.master, textvariable=self.new_class)
        new_class_entry.pack(pady=5)

    def load_data(self):
        # Load the data from a numpy file
        file_path = filedialog.askopenfilename(
            filetypes=[("Numpy Files", "*.npz")])
        if file_path:
            npzfile = np.load(file_path)
            self.data = npzfile['data']
            self.filtered_data = None
            self.current_class.set("")

    def filter_data(self):
        # Filter the data by classification label
        if self.data is not None:
            filtered_class = filedialog.askstring(
                "Filter Data", "Enter the classification label to filter by:")
            if filtered_class:
                self.filtered_data = self.data[self.data[:, 1]
                                               == filtered_class]
                self.current_class.set(filtered_class)

    def update_class(self):
        # Update the classification label of the filtered data
        if self.filtered_data is not None:
            new_class = self.new_class.get()
            if new_class:
                self.filtered_data[:, 1] = new_class
                self.new_class.set("")
                self.current_class.set("")


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Hand Gesture Recognition")

        # Create the sidebar frame
        self.sidebar_frame = tk.Frame(self.master, bg='gray', width=100)
        self.sidebar_frame.pack(side='left', fill='y')

        # Create the content frame
        self.content_frame = tk.Frame(self.master, bg='white', width=400)
        self.content_frame.pack(side='right', fill='both', expand=True)

        # Create the navigation buttons
        self.home_button = tk.Button(
            self.sidebar_frame, text="Home", command=self.show_home)
        self.home_button.pack(pady=10)

        self.train_button = tk.Button(
            self.sidebar_frame, text="Train", command=self.show_train)
        self.train_button.pack(pady=10)

        self.contact_button = tk.Button(
            self.sidebar_frame, text="Contact", command=self.show_contact)
        self.contact_button.pack(pady=10)

        # Create the home page
        self.home_page = HomePage(self.content_frame)

    def show_home(self):
        # Remove all content from the content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create the home page
        self.home_page = HomePage(self.content_frame)
        self.home_page.pack(fill='both', expand=True)

    def show_train(self):
        # Remove all content from the content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create the train page
        self.train_page = TrainPage(self.content_frame)
        self.train_page.pack(fill='both', expand=True)

    def show_contact(self):
        # Remove all content from the content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create the contact page
        self.contact_page = ContactPage(self.content_frame)
        self.contact_page.pack(fill='both', expand=True)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    app.home_page.pack(fill='both', expand=True)
    root.mainloop()
