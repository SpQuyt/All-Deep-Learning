# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Making new predictions
import numpy as np
from keras.preprocessing import image

# load json and create model
json_file = open('5thmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("5thmodel.h5")
print("Loaded model from disk")

class_names = ['Cats', 'Dogs']

# test_image = image.load_img('cat2.jpg', target_size=(64, 64))
# test_image = image.load_img('cat.jpg', target_size=(64, 64))
# test_image = image.load_img('cat3.jpg', target_size=(64, 64))
# test_image = image.load_img('cat4.jpg', target_size=(64, 64))
# test_image = image.load_img('cat5.jpg', target_size=(64, 64))
# test_image = image.load_img('cat6.jpg', target_size=(64, 64))
# test_image = image.load_img('cat7.jpeg', target_size=(64, 64))
# test_image = image.load_img('cat8.jpg', target_size=(64, 64))

# test_image = image.load_img('dog3.jpg', target_size=(64, 64))
# test_image = image.load_img('dog2.jpg', target_size=(64, 64))
# test_image = image.load_img('dog.jpg', target_size=(64, 64))
test_image = image.load_img(
    'CNN_Data/test_set/cats/cat.4085.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
# test_image = preprocess_input(test_image)
result_class = loaded_model.predict_classes(test_image)
result_confident = loaded_model.predict(test_image)

print(result_confident)
if (result_confident[0][0] > result_confident[0][1]):
    print("CATS")
else:
    print("DOGS")
