import os
import cv2 as cv
import numpy as np

# Opencv's haar_cascade recognizer
characters = [] # Stores the folder name of the characters

for folder in os.listdir(os.path.join(os.getcwd(), 'league')):
    characters.append(folder)
print(characters)

feature = [] # The image array of the faces. Region of interset for the face
labels = [] # Index of the feature in characters, which face does it belong to in characters array

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for character in characters:
        path = os.path.join(os.getcwd(), "league", character)
        label = characters.index(character)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            # Read the image
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            for (x, y, w, h) in face_rect:
                face_roi = gray[y: y+h, x:x+w]
                feature.append(face_roi)
                labels.append(label)
                

if __name__ == '__main__':
    # Train it based on the characters folders
    create_train() 

    # Instanitated face recognizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Convert to numpy array
    feature = np.array(feature, dtype='object')
    labels = np.array(labels)

    # Train the recognizer on the feature list and label
    face_recognizer.train(feature, labels)  

    print("Training done")

    # Save the trained model in a yml file that can be read later
    face_recognizer.save('face_trained.yml')
                
            
            
            
            
            
            
            