import face_recognition
import os 
import cv2

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] /  100 * percent)
    height = int(frame.shape[0] / 100 * percent)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


KNOWN_FACES_DIR = os.path.join('..', 'data', 'known_faces')
UNKNOWN_FACES_DIR = os.path.join('..', 'data', 'unknown_faces') 
TOLERANCE = 0.4
FRAME_THICKNESS = 3 
FONT_THICKNESS = 2
MODEL = "hog"

print("Loading known faces") # Log

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):

        print(f"Processing {filename} ...") # Log

        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name, filename))
        encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(encoding)
        known_names.append(name)


for filename in os.listdir(os.path.join(UNKNOWN_FACES_DIR)):
    image = face_recognition.load_image_file(os.path.join(UNKNOWN_FACES_DIR, filename))
    locations = face_recognition.face_locations(image, model = MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f"Processing {filename} ...") # Log

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    image = rescale_frame(image, 30) 
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




