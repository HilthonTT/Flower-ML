from keras import Sequential
from keras.regularizers import L2
from keras.layers import (
    InputLayer, 
    Conv2D, 
    MaxPool2D, 
    Dropout, 
    Dense, 
    RandomRotation, 
    RandomFlip, 
    RandomContrast,
    Resizing, 
    Rescaling,
    GlobalAveragePooling2D,
    BatchNormalization,
)

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
        RandomFlip(mode="horizontal"),
        RandomContrast(factor=0.1),
    ])

    return layers

def create_model() -> Sequential:
    resize_rescale_layers = create_rescale_layers()
    augment_layers = create_augment_layers()

    model = Sequential()

    # Input layer
    model.add(InputLayer(input_shape=(None, None, 3)))  # Adjust input shape as needed

    # Rescale layers
    model.add(resize_rescale_layers)
    
    # augment_layers
    model.add(augment_layers)

    # Convolutional Layer 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))

    # Flatten
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(512, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(FLOWERS_COUNT, activation='softmax'))  # Use softmax for multi-class classification

    return model