import csv
import pandas as pd


def getLabel(label):
    row_number = None
    found_label = False
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', 'r', newline='') as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            # Check if the label is present in the current row
            if label.strip().upper() in [cell.strip().upper() for cell in row]:
                row_number = i  # If so, record the index of the row
                found_label = True
                break
    # Return the index of the row containing the label (whether it was pre-existing or newly added).
    return row_number


def delete_data(label):

    label_row_number = getLabel(label)

    # Open the CSV file and read its contents
    with open('../model/keypoint_classifier/keypoint.csv', 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # Find the rows that contain the value you want to delete
    value_to_delete = label_row_number
    rows_to_delete = [row for row in data if row[0] == value_to_delete]

    # Remove the rows from the data
    for row in rows_to_delete:
        data.remove(row)

    # Write the updated data back to the CSV file
    with open('../model/keypoint_classifier/keypoint.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)
