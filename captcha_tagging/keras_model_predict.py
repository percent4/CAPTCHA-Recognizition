from keras.models import load_model
import cv2
import numpy as np

model = load_model('best_model.h5')

batch_size = 1
width, height, n_len, n_class = 50, 22, 4, 10

X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

for i in range(batch_size):
    X_test = cv2.imread('./new_image/code12.png', 1)
    X[i] = X_test

y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=2)
print(y_pred)