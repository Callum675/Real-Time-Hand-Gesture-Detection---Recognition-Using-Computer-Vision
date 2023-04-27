from tkinter import Frame, Label, Entry, Button, messagebox
import pandas as pd
from tkinter import *
from tkinter import messagebox
import gesture_detection as gest
import csv
from utils import collect_data
from utils import delete_data
from utils import trainModel


def start_cam():
    main_label.config(text="Type Q to quit")
    gest.main()


# GUI
root = Tk()
var = IntVar()
root.title("Gesture Interpreter")
root.config(padx=100, pady=100, bg="#364f6b")

start_button = Button(text="Start Interpreter", command=start_cam,
                      relief=FLAT, bg="white", fg="black", highlightthickness=0)
start_button.grid(column=1, row=1)


class DeletePage(Frame):
    def __init__(self, parent):
        super().__init__(parent)

        about_label = Label(self, text="Delete Data", font=("Arial", 20))
        about_label.grid(pady=30)

        # Create the label and entry for the label to delete
        class_Label = Label(self, text="Label to delete:").grid(pady=10)
        self.class_entry = Entry(self)
        self.class_entry.grid(pady=10)

        # Create the button to delete the data
        self.delete_Button = Button(
            self, text="Delete", command=self.delete_data).grid(pady=10)

    def delete_data(self):
        # Get the label to delete from the entry field
        label = self.class_entry.get()

        delete_data(label)

        # Display a message box indicating that data deletion is complete
        messagebox.showinfo(
            "Data Deletion", "Data deletion is complete.")


class TrainPage(Frame):
    def __init__(self, parent):
        super().__init__(parent)

        vcmd = (self.register(self.validate_entry), '%P')

        def caps(event):
            label.set(label.get().upper())

        about_label = Label(self, text="Add Train Data", font=("Arial", 20))
        about_label.grid(pady=30)

        # Create the label and entry for the number of frames to collect
        frames_label = Label(self, text="Number of frames:").grid(pady=10)
        self.num_frames_entry = Entry(
            self, validate="key", validatecommand=vcmd)
        self.num_frames_entry.grid(pady=10)

        # Create the label and entry for the interval
        interval_Label = Label(
            self, text="Interval between frames(S):").grid(pady=10)
        self.interval_entry = Entry(self, validate="key", validatecommand=vcmd)
        self.interval_entry.grid(pady=10)

        # Create the label and entry for the label to assign to the data
        class_Label = Label(self, text="Label:").grid(pady=10)
        label = StringVar()
        self.class_entry = Entry(self, textvariable=label)
        self.class_entry.grid(pady=10)
        self.class_entry.bind("<KeyRelease>", caps)

        # Create the button to start collecting data
        self.start_Button = Button(
            self, text="Collect", command=self.collect_data).grid(pady=10)
        # Create the button to start collecting data
        self.train_Button = Button(
            self, text="Train", command=self.start_train).grid(pady=10)

    def collect_data(self):
        # Get the number of frames to collect and the label to assign to the data
        num_frames = int(self.num_frames_entry.get())
        label = self.class_entry.get()
        frame_interval = int(self.interval_entry.get())

        # Collect the data using the collect_data() function
        data = collect_data(num_frames, frame_interval, label)

        # Display a message box indicating that data collection is complete
        messagebox.showinfo(
            "Data Collection", "Data collection is complete.")

    def validate_entry(self, new_value):
        if new_value.isdigit() or new_value == "":
            return True
        else:
            messagebox.showerror(
                "Invalid Entry", "Please enter a number.")
            return False

    def start_train(self):
        trainModel()


class LabelPage(Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Read Classification Labels
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        rows = len(keypoint_classifier_labels)
        columns = len(keypoint_classifier_labels[0])

        label = Label(self, text="Model Labels", padx=20, pady=10)
        label.grid(pady=10)

        for x in range(rows):
            for y in range(columns):
                labels_table = Entry(self, width=15, fg='blue', font=(
                    'Timew New Roman', 13, 'italic'))
                labels_table.grid(row=x, column=y, sticky="W")
                labels_table.insert(END, keypoint_classifier_labels[x][y])

        # save button - saves label changes
        def save_labels():
            csv_path = 'model/keypoint_classifier/keypoint_classifier_label_test.csv'
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                for entry in labels_table:
                    writer.writerow(entry.get())

        btn_label = Button(self, text="update labels", font=("Arial", 12),  fg="black",
                           bg="white",  cursor="hand2", activebackground="light grey", command=save_labels)
        btn_label.grid(row=0, column=3, sticky="W")


class editWindow(Toplevel):
    def __init__(self, master):
        super().__init__(master=master)
        self.title("Edit Hand Gesture Recognition")

        # Create the sidebar frame
        self.sidebar_frame = Frame(self, bg='gray', width=100)
        self.sidebar_frame.grid(row=0, column=0, sticky='nsw')
        self.sidebar_frame.rowconfigure(0, weight=1)

        # Create the content frame
        self.content_frame = Frame(self, bg='white', width=400)
        self.content_frame.grid(row=0, column=1, sticky='nsew')

        # Create the navigation buttons
        self.label_button = Button(
            self.sidebar_frame, text="Labels", command=self.show_label)
        self.label_button.grid(pady=10)

        self.train_button = Button(
            self.sidebar_frame, text="Train", command=self.show_train)
        self.train_button.grid(pady=10)

        self.delete_button = Button(
            self.sidebar_frame, text="Delete", command=self.show_delete)
        self.delete_button.grid(pady=10)

        # Create the Starting page
        self.train_page = TrainPage(self.content_frame)

    def show_label(self):
        # Remove all content from the content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create the label page
        self.label_page = LabelPage(self.content_frame)
        self.label_page.grid(sticky='nsew')

    def show_train(self):
        # Remove all content from the content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create the train page
        self.train_page = TrainPage(self.content_frame)
        self.train_page.grid(sticky='nsew')

    def show_delete(self):
        # Remove all content from the content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create the delete page
        self.delete_page = DeletePage(self.content_frame)
        self.delete_page.grid(sticky='nsew')


edit_button = Button(root, text="Edit Gestures", relief=FLAT,
                     bg="white", fg="black", highlightthickness=0)
edit_button.bind("<Button>", lambda e: editWindow(root))
edit_button.grid(column=1, row=2, pady=10)


class helpWindow(Toplevel):

    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Help")
        self.geometry("700x500")
        start_title = Label(self, text="Getting Started\n",
                            justify=LEFT, font=("Arial", 10, "bold"))
        start_title.grid(column=0, row=0, sticky="W")
        start_content = Label(self, text="Click the 'Start Interpreter' button to begin.\n"
                                         "Type 'Q' to quit.\n\n", justify=LEFT, font=("Arial", 10))
        start_content.grid(column=0, row=1, sticky="W")
        video_call_title = Label(
            self, text="Video Call Integration\n", justify=LEFT, font=("Arial", 10, "bold"))
        video_call_title.grid(column=0, row=4, sticky="W")
        video_call_content = Label(self, text="To use this interpreter within a video conferencing application such as Zoom, Skype or Google Duo,\n"
                                              "You must first create a virtual webcam.\n"
                                              "One of the easiest ways to do this is with OBS Studio:\n\n"
                                              "With OBS Studio installed and open, start the sign interpreter\n"
                                              "In sources click '+' and select 'Window Capture'\n"
                                              "You should be able to select 'object detection from the list of window options\n"
                                              "Click 'Start Virtual Camera'. The virtual camera should now start.\n"
                                              "You can then choose this virtual camera in your video conferencing application settings.",
                                   justify=LEFT, font=("Arial", 10))
        video_call_content.grid(column=0, row=5, sticky="W")


help_button = Button(root, text="Help", relief=FLAT,
                     bg="white", fg="black", highlightthickness=0, width=4)
help_button.bind("<Button>", lambda e: helpWindow(root))
help_button.grid(column=1, row=3, pady=10)

main_label = Label(text="Type 'Q' to quit", bg="#364f6b", fg="#eeeeee")
main_label.grid(column=1, row=4)


root.mainloop()
