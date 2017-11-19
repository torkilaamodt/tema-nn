import numpy as np
from keras.applications import vgg16
from keras import backend as K

def with_image(image):
    model = default_model()
    return with_image_and_model(image, model)

def with_image_and_model(image, model):
    layer_to_optimize = 'block1_conv1'
    filters_to_optimize = None
    gradients_function = optimize_for(model, layer_to_optimize, feature=filters_to_optimize)
    return augment_image(image, model, gradients_function)

def augment_image(image, model, gradients_function):
    scale = 1.
    niterations = 20

    progress_pattern = 'Dreaming {:3}%'
    nchars = len(progress_pattern.format(0))
    progress_pattern = '{}{}'.format('\b'*nchars, progress_pattern)

    augmented_image = image.reshape([1] + list(image.shape))
    for i in range(niterations):
        progress = int(100 * i / niterations)
        print(progress_pattern.format(progress), end='', flush=True)

        loss, gradients = gradients_function([augmented_image])
        augmented_image += gradients * scale

    print(progress_pattern.format(100))
    return postprocess(augmented_image[0])

def postprocess(x):
    if 'channels_first' == K.image_data_format():
        x = x.transpose((1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)

def optimize_for(model, layer, feature=None):
    layers_by_name = { l.name: l for l in model.layers[1:] }
    activations = layers_by_name[layer].output
    if None == feature:
        loss = K.mean(activations)
    elif 'channels_first' == K.image_data_format():
        loss = K.mean(activations[:, feature, :, :])
    else:
        loss = K.mean(activations[:, :, :, feature])

    gradients = K.gradients(loss, [model.input])[0]
    gradients = normalize(gradients)

    return K.function([model.input], [loss, gradients])


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def default_model():
    return vgg16.VGG16(weights='imagenet', include_top=True)