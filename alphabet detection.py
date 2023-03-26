import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

image_data = np.load('image.npz')['arr_0']
labels = pd.read_csv('labels.csv')['labels']

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, random_state=9, train_size=7500, test_size=2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = clf.score(X_test_scaled , y_test)
print("The accuracy is :- ",accuracy)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    cv2.rectangle(frame,(300,300),(100,100),(0,255,0),0)
    crop_image = frame[100:300, 100:300]

    image = Image.fromarray(crop_image)

    image = image.convert('L')

    image = np.array(image)
    image = np.clip(image, 0, 255)

    test_sample = np.array(image).reshape(1, 40000)
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()