# Convolutional Neural Network

# Importing the Keras libraries and Packages
from keras.models import  Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Part 1. Building the Convolutional Neural Network (CNN)
# Initialising the CNN
classifier = Sequential()

# 1. Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))

# 2. Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second CNN layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 3. Flattening
classifier.add(Flatten())

# 4. Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))


# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')


# Part 2. Fitting the CNN to the images


from keras_preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit(training_set,
               steps_per_epoch=8000/32,
               epochs=50,
               validation_data=test_set,
               validation_steps=2000/32)


# Part 3. Making new predictions

import numpy as np
from keras.preprocessing import image

test_image1 = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis=0)

result1 = classifier.predict(test_image1)
dog_cat1 = training_set.class_indices

if result1[0][0] >0.5:
    prediction1 = 'Dog'
else:
    prediction1 = 'Cat'
    

test_image2 = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)

result2 = classifier.predict(test_image2)
dog_cat2 = training_set.class_indices

if result1[0][0] ==1:
    prediction2 = 'Dog'
else:
    prediction2 = 'Cat'
    

    
    
    


