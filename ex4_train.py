# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'softmax'))
# Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('CNN_Data/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 2,
                                                    class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('CNN_Data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'categorical')
classifier.fit_generator(training_set,
                            steps_per_epoch = 80,
                            epochs = 10,
                            validation_data = test_set,
                            validation_steps = 20
                            )

# epochs = 4
# batch_size = 128
# # Fit the model weights.
# classifier.fit(training_set,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=test_set)


# Part 3 - Serialize model to JSON
model_json = classifier.to_json()
with open("4thmodel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("4thmodel.h5")
print("Saved model to disk")

