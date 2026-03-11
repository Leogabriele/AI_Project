import face_recognition
import pickle
import os

dataset_path = "dataset"
encodings = []
names = []

print("[INFO] Loading dataset...")

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        print("[INFO] Processing:", image_path)

        image = face_recognition.load_image_file(image_path)

        boxes = face_recognition.face_locations(image)

        face_encodings = face_recognition.face_encodings(image, boxes)

        for encoding in face_encodings:
            encodings.append(encoding)
            names.append(person_name)

data = {"encodings": encodings, "names": names}

os.makedirs("encodings", exist_ok=True)

with open("encodings/encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encoding completed and saved!")