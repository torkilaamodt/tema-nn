from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import sys
import numpy as np

def load_image(path, w, h):
    #datagen = ImageDataGenerator(rescale=1./255)

    img = load_img(path, target_size=(w, h))  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (depth, width, height)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, depth, width, height)
    return x

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def main():
    model = load_model('m.h5', custom_objects={'recall': recall, 'precision':precision})

    path = sys.argv[1]
    image = load_image(path, 128, 128)

    classes = ["Hotdog", "Not-hotdog"]
    prediction = model.predict(image)
    print(classes[np.argmax(prediction)])


if __name__ == "__main__":

    main()
