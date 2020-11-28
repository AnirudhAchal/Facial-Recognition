# Facial-Recognition

## Installation of facial_recognition on windows
  1. Install Visual Studio
  2. Install cmake by running 'pip install cmake' in your project environment
  3. Clone the public repository 'git clone git://github.com/ageitgey/face_recognition'
  4. Install face_recognition by running 'python setup.py install' inside the directory of the cloned repo in your project environment.
  For detailed explaination check out https://face-recognition.readthedocs.io/en/latest/installation.html
  
## Using your own Images
  1. Make a directory 'data' in the main directory of the repo
  2. Make 2 sub-directories 'known_faces' and 'unknown_faces'
  3. All test images can be put into the 'unknown_faces' directory
  4. In the 'known_faces' directory make a sub-directory for each indivial person in you dataset. For example, make sub-directoriesperson_1, person_2, ... inside 'known_faces' where person_1 is a directory containing all the images of 'person_1' etc
