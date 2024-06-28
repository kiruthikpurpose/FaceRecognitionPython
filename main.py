import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import pyttsx3

# Function to load all face images from a folder
def load_faces_from_folder(folder_path):
    faces = []
    valid_extensions = ['.png']

    for filename in os.listdir(folder_path): 
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            face_name = os.path.splitext(filename)[0]  # Remove the file extension
            face_image = face_recognition.load_image_file(os.path.join(folder_path, filename))
            face_encoding = face_recognition.face_encodings(face_image)[0]
            faces.append({"name": face_name, "encoding": face_encoding})
    return faces

# Load known face encodings and names from the 'faces' folder
known_faces_folder = 'C:\\Users\\kirut\\Desktop\\FACE ATT\\faces'
students = load_faces_from_folder(known_faces_folder)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize CSV file
now = datetime.now()
todate = now.strftime("%d-%m-%Y")
f = open(todate + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

present_students = []  # List to store present students

# Set the tolerance level for face recognition
tolerance = 0.45

engine = pyttsx3.init()
engine.setProperty('voice', engine.getProperty('voices')[0].id) 

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Resize and convert the frame to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Display the frame
    cv2.imshow("Face Attendance System", frame)

    # Recognize faces in the frame
    faces_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, faces_locations)

    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces([student["encoding"] for student in students], face_encoding, tolerance=tolerance)
        name = ""

        if any(matches):
            best_match_index = np.argmax(matches)
            name = students[best_match_index]["name"]
            print("Recognized:", name)
            engine.say(f"Recognized: {name}")
            engine.runAndWait()

            # Update attendance
            if name in [student["name"] for student in students]:
                students = [student for student in students if student["name"] != name]
                print("Remaining:", [student["name"] for student in students])

                # Record attendance with a unique timestamp for each student
                totime = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, "Present", totime])

                # Add the present student to the list with collected timestamp
                present_students.append({"name": name, "timestamp": totime})

    # Exit on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
f.close()

# Sort present and absent students alphabetically
sorted_present_students = sorted(present_students, key=lambda x: x["name"])
sorted_absent_students = sorted([student["name"] for student in students])

# Write the alphabetical order with attendance status to a CSV file
attendance_csv_filename = todate + '.csv'
with open(attendance_csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header
    csvwriter.writerow(["Name", "Attendance Status", "Attendance Time"])

    # Write present students
    for student in sorted_present_students:
        csvwriter.writerow([student["name"], "Present", student["timestamp"]])

    # Write absent students
    for name in sorted_absent_students:
        csvwriter.writerow([name, "Absent", "-"])

# Print the list of present and absent students
print("\nList of Presentees:\n")
print("\n".join(student["name"] for student in sorted_present_students))

print("\nList of Absentees:\n")
print("\n".join(sorted_absent_students))
print("\n")

# Print a message indicating the CSV file location
print(f"Attendance details are saved in '{attendance_csv_filename}'.")
