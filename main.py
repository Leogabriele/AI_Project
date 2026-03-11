import face_recognition
import cv2
import pickle

print("[INFO] Loading encodings...")
data = pickle.loads(open("encodings/encodings.pickle", "rb").read())

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("[INFO] Starting video stream...")

while True:

    ret, frame = video.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces
    boxes = face_recognition.face_locations(rgb)

    # encode faces (let library handle internally)
    encodings = face_recognition.face_encodings(rgb)

    names = []

    for encoding in encodings:

        matches = face_recognition.compare_faces(data["encodings"], encoding)

        name = "Unknown"

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]

            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        y = top - 10 if top - 10 > 10 else top + 10

        cv2.putText(frame, name, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0,255,0), 2)

    cv2.imshow("Crowd Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()