import argparse
import numpy as np 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imutils import face_utils
import imutils
import dlib
import cv2
import csv

mLEFT_EYE = 'left_eye'
mLEFT_EYEBROW = 'left_eyebrow'
mRIGHT_EYE = 'right_eye'
mMOUTH = 'mouth'
mRIGHT_EYEBROW = 'right_eyebrow'

class Images:
    
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.
        
        self.predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')
        y_features = [];
        x_features = [];
        f = open(location, 'r');
        i = 0;
        reader = csv.DictReader(f);
        for row in reader:
            y_features.append(int(row['Emotion']));
            face = np.array([int(intensity) for intensity in row['Pixels'].split(' ')], np.uint8);
            image = np.reshape(face, (48,48));            
            shape = self.predictor(image, dlib.rectangle(top=0, left=0, bottom=48, right=48))
            shape = face_utils.shape_to_np(shape)
            
            height = dict();
            width = dict();
            
            # loop over the face parts individually
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))                
                width[name] = w;
                height[name] = h;
            
            #print width,'\n', height;
            x_features.append(np.array([width[mLEFT_EYEBROW], height[mLEFT_EYEBROW] , width[mRIGHT_EYEBROW], height[mRIGHT_EYEBROW], 
                                         width[mLEFT_EYE], height[mLEFT_EYE], width[mRIGHT_EYE], height[mRIGHT_EYE],
                                         width[mMOUTH], height[mMOUTH]]));
            
        print type(x_features)
        print len(x_features[0])
        
        x_features = np.array(x_features);
        y_features = np.array(y_features);  
        
        
        shuff = np.arange(x_features.shape[0])
        
        n = len(shuff);
        trainingSamples = n*7/10;    
        
        np.random.shuffle(shuff)
        self.x_train = x_features[shuff[:trainingSamples],:]
        self.y_train = y_features[shuff[:trainingSamples]]
         
        self.x_valid = x_features[shuff[trainingSamples:],:]
        self.y_valid = y_features[shuff[trainingSamples:]]
         
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    
    data = Images("../data/train.csv")
    
    # -----------------------------------
    # Plotting Examples 
    # -----------------------------------
    
    # Display in on screen  
    

