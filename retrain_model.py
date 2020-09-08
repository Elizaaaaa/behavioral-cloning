import numpy as np
from scipy import ndimage
import csv

def generator(samples, batch_size=32):
    import sklearn
    from random import shuffle
    
    n = len(samples)
    
    path = './data/IMG/'
    correction = 0.15
    while 1:
        shuffle(samples)    
        for i in range(0, n, batch_size):
            batch_samples = samples[i:i+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                angle = float(sample[3])
                filename = sample[0].split('/')[-1]
                image = ndimage.imread(path+filename)
                images.append(image)
                angles.append(angle)
                
                filename = sample[1].split('/')[-1]
                image = ndimage.imread(path+filename)
                images.append(image)
                angles.append(angle+correction)
              #  images.append(np.fliplr(image))
              #  angles.append(-1 * (angle+correction))
                
                filename = sample[2].split('/')[-1]
                image = ndimage.imread(path+filename)
                images.append(image)
                angles.append(angle-correction)
              #  images.append(np.fliplr(image))
              #  angles.append(-1 * (angle-correction))
                
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
    
    from keras.models import load_model
    model = load_model('model2.h5')
    
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(train_sample)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_sample)/batch_size),
                        epochs=2, verbose=1)

    model.save('model3.h5')
    
    
if __name__ == "__main__":
    #X_train, y_train = load_images()
    train()