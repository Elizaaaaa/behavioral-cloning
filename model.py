import numpy as np
from scipy import ndimage
import csv

def load_images():    
    lines = []
    with open('/opt/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        filename = line[0].split('/')[-1]
        path = '/opt/data/IMG/'
        image = ndimage.imread(path+filename)
        steer = float(line[3])
        images.append(image)
        measurements.append(steer)
        # Flip the image and add again
        images.append(np.fliplr(image))
        measurements.append(-steer)
        # Add other 2 camera images
        filename = line[1].split('/')[-1]
        limg = ndimage.imread(path+filename)
        filename = line[2].split('/')[-1]
        rimg = ndimage.imread(path+filename)
        images.append(limg)
        images.append(rimg)
        correction = 0.15
        measurements.append(steer+correction)
        measurements.append(steer-correction)
      
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train

def generator(samples, batch_size=32):
    import sklearn
    from random import shuffle
    
    n = len(samples)
    
    path = './data/IMG/'
    correction = 0.2
    while 1:
        shuffle(samples)    
        for i in range(0, n, batch_size):
            batch_samples = samples[i:i+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                angle = float(sample[3])
                filename = sample[0].split('/')[-1]
                images.append(ndimage.imread(path+filename))
                angles.append(angle)
                filename = sample[1].split('/')[-1]
                images.append(ndimage.imread(path+filename))
                angles.append(angle+correction)
                filename = sample[2].split('/')[-1]
                images.append(ndimage.imread(path+filename))
                angles.append(angle-correction)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    
def train():
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
    from sklearn.model_selection import train_test_split
    from math import ceil
    
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
       
    train_sample, validation_sample = train_test_split(samples, test_size=0.2)
    
    batch_size = 32
    train_generator = generator(train_sample, batch_size=batch_size)
    validation_generator = generator(validation_sample, batch_size=batch_size)
        
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(rate=0.3))
    #model.add(Conv2D(48, kernel_size=(5,5), activation='relu'))
    #model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    #model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    #model.add(Flatten(input_shape=(160,320,3)))
    #model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    #model.fit(x, y, validation_split=0.2, epochs=2, shuffle=True)
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(train_sample)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_sample)/batch_size),
                        epochs=5, verbose=1)

    model.save('model.h5')
    
    
if __name__ == "__main__":
    #X_train, y_train = load_images()
    train()