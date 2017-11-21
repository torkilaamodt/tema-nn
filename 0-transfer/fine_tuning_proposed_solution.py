from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

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

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return metrics.fbeta_score(y_true, y_pred, beta=1)

def plot_training(history, t):
    import matplotlib.pyplot as plt
    plt.plot(history[t])
    plt.plot(history['val_' + t])
    plt.title(t)
    plt.ylabel(t)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot(history):
    plot_training(history, 'acc')
    plot_training(history, 'loss')
    plot_training(history, 'recall')
    plot_training(history, 'precision')


def create_generators(batch_size, w, h):
    train_datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=False,
            rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/training',
            target_size=(w, h),
            batch_size=batch_size,
            class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(w, h),
            batch_size=batch_size,
            class_mode='categorical')

    return train_generator, validation_generator


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Activation


from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Activation
from keras import metrics
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D

def create_model(w, h, number_of_classes):
    vgg16_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape = (w,h,3)
        )

    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    for layer in model.layers:
        layer.trainable = False

    model.add(Flatten())
    model.add(Dense(number_of_classes, activation="softmax"))
    model.summary()
    return model


def main():
    # Images size
    w,h = 128, 128
    batch_size = 10
    # Hotdog and not-hotdog
    number_of_classes = 2
    # Use generators to produce training data https://keras.io/preprocessing/image/
    train_generator, validation_generator = create_generators(batch_size, w, h)

    model = create_model(w, h, number_of_classes)

    model.compile(
        'adam',
        'categorical_crossentropy',
        metrics=['accuracy', recall,precision])

    # Training your model with training generators
    history = model.fit_generator(
        train_generator,
        10,
        epochs=10,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=2)

    plot(history=history.history)

    # Save your model for later use..
    model.save("m.h5")



if __name__ == "__main__":

    main()
