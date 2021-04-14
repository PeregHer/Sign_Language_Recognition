import numpy as np
import pickle
import mediapipe as mp
import cv2
from string import ascii_lowercase

model = pickle.load(open('model_2.sav', 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

dico = {k: v for k, v in enumerate(ascii_lowercase)}

cap = cv2.VideoCapture(0)

#Initiate holistic model:
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        #Recolor Feed
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Make detection
        results = holistic.process(img)
        input_model = []
        
        try:
            coords = results.right_hand_landmarks.landmark
        
            for coord in coords:
                input_model.append(coord.x)
                input_model.append(coord.y)
                input_model.append(coord.z)
        
        except: pass
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        #Right hand
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        try: 
            input_model = np.array(input_model).reshape(1, 63)
            pred = model.predict(input_model)
            img = cv2.putText(img, dico[pred[0]],(250,450), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5)
        except: pass
        
        cv2.imshow('Test Model', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()