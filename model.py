import numpy as np
from scipy import ndimage

def load_images():
    import csv
    
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
#         filename = line[1].split('/')[-1]
#         limg = ndimage.imread(path+filename)
#         filename = line[2].split('/')[-1]
#         rimg = ndimage.imread(path+filename)
#         images.append(limg)
#         images.append(rimg)
#         correction = 0.1
#         measurements.append(steer+correction)
#         measurements.append(steer-correction)
      
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train

    
def train(x, y):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(90,320,3)))
    model.add(Conv2D(6, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    
    #model.add(Flatten(input_shape=(160,320,3)))
    #model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, validation_split=0.2, epochs=3, shuffle=True)

    model.save('model.h5')
    
    
if __name__ == "__main__":
    X_train, y_train = load_images()
    train(X_train, y_train)