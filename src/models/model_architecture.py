from keras import Sequential
from keras.layers import (
    InputLayer, 
    Conv2D, 
    BatchNormalization, 
    MaxPool2D, 
    Dropout, 
    Flatten, 
    Dense, 
    RandomRotation, 
    RandomFlip, 
    Resizing, 
    Rescaling
)
from keras.regularizers import L2

IMAGE_SIZE = 224
FLOWERS_COUNT = 5

def create_rescale_layers() -> Sequential:
    layers = Sequential([
        Resizing(IMAGE_SIZE, IMAGE_SIZE),
        Rescaling(1.0/255)
    ])

    return layers

def create_augment_layers() -> Sequential:
    layers = Sequential([
        RandomRotation(factor=(0.25, 0.2501)),
        RandomFlip(mode="horizontal")
    ])

    return layers

def create_model() -> Sequential:
    dropout_rate = 0
    regularization_rate = 0.01
    relu_activation = 'relu'
    valid_padding = 'valid'
    regularizer = L2(regularization_rate)

    resize_rescale_layers = create_rescale_layers()
    augment_layers = create_augment_layers()

    model = Sequential([
        InputLayer(input_shape=(None, None, 3)),

        resize_rescale_layers,
        augment_layers,

        Conv2D(filters=6, 
               kernel_size=3, 
               strides=1, 
               padding=valid_padding, 
               activation=relu_activation, 
               kernel_regularizer=regularizer),
        BatchNormalization(),

        MaxPool2D(pool_size=2, strides=2),
        Dropout(rate=dropout_rate),

        Conv2D(filters=16, 
               kernel_size=3, 
               strides=1, 
               padding=valid_padding, 
               activation=relu_activation, 
               kernel_regularizer=regularizer),
        BatchNormalization(),

        MaxPool2D(pool_size=2, strides=2),
        Flatten(),

        Dense(100, activation='relu', kernel_regularizer=regularizer),
        BatchNormalization(),
        Dense(FLOWERS_COUNT, activation='sigmoid')
    ])