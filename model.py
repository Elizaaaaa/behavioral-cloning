import numpy as np
import cv2

def load_images():
    import csv
    
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        filename = line[0].split('/')[-1]
        path = './data/IMG/' + filename
        bgr_image = cv2.imread(path)
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurements.append(float(line[3]))
      
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train

    
def train(x, y):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, validition_split=0.2, shuffle=True)
    
    model.save('model.h5')
    
    
if __name__ == "__main__":
    X_train, y_train = load_images()
    train(X_train, y_train)