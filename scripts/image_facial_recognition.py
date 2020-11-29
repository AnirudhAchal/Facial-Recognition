# Importing Libraries
import face_recognition
import os 
import cv2


# Constant Variables
KNOWN_FACES_DIR = os.path.join('..', 'data', 'known_faces')
UNKNOWN_FACES_DIR = os.path.join('..', 'data', 'unknown_faces') 
TOLERANCE = 0.4 # Set higher if more recall is desired / set lower if more precision is desired
FRAME_THICKNESS = 3 
FONT_THICKNESS = 2
FONT_FAMILY = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
MODEL = "hog" # Use "cnn" if enough computing power is available


# Function that scales the frame (set according to requirement)
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] /  100 * percent)
    height = int(frame.shape[0] / 100 * percent)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':

    print("Loading known faces") # Log

    known_faces = [] # Stores face encodings
    known_names = [] # Stores respective face names

    for name in os.listdir(KNOWN_FACES_DIR):

        print(f"Processing images of {name} ...") # Log

        for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):

            print(f"Processing {filename} ...") # Log

            image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name, filename)) # Loading the image
            encoding = face_recognition.face_encodings(image)[0] # Getting face encodings

            known_faces.append(encoding)
            known_names.append(name)


    for filename in os.listdir(os.path.join(UNKNOWN_FACES_DIR)):

        print(f"Processing {filename} ...") # Log

        image = face_recognition.load_image_file(os.path.join(UNKNOWN_FACES_DIR, filename)) # Loading the test image
        locations = face_recognition.face_locations(image, model = MODEL) # Locating face using selected model
        encodings = face_recognition.face_encodings(image, locations) # Finding the encodings of located faces
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE) # Finding matches 
            match = None

            # Drawing rectangle around the face if match is found
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
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), FONT_FAMILY, FONT_SIZE, (200, 200, 200), FONT_THICKNESS)


        image = rescale_frame(image, 30) # Rescale Frame depending on monitor size
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




