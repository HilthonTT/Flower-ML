from keras.preprocessing.image import ImageDataGenerator
import os

BATCH_SIZE = 32
IM_SIZE = 224
TARGET_SIZE = (IM_SIZE, IM_SIZE)

def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "flowers")

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir, 
        target_size=TARGET_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        subset='training',
        shuffle=True)

    val_generator = train_datagen.flow_from_directory(
        directory=data_dir, 
        target_size=TARGET_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        subset='validation',
        shuffle=True)
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        directory=data_dir, 
        target_size=TARGET_SIZE, 
        batch_size=1, 
        class_mode='categorical',
        subset='validation')
    
    return train_generator, val_generator, test_generator