import numpy as np
from keras.preprocessing.image import load_img
from keras import backend as K

def image(path):
    target_size = (224, 224)
    image = load_img(path, grayscale=False, target_size=target_size)
    data = np.fromstring(image.tobytes(), dtype=np.uint8)
    if 'channels_first' == K.image_data_format():
        data = data.reshape([3] + list(target_size))
    else:
        data = data.reshape(list(target_size) + [3])
    return data.astype(np.float32)