from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

BATCH_SIZE = 32
IM_SIZE = 224

def load_data():
    data_dir = "flowers"

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
    
    dataset = train_datagen.flow_from_directory(
        data_dir, target_size=(IM_SIZE, IM_SIZE), batch_size=BATCH_SIZE)
    
    return dataset

def splits(dataset, train_ratio, val_ratio, test_ratio):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(train_ratio*DATASET_SIZE))

    val_test_dataset = dataset.skip(int(train_ratio*DATASET_SIZE))
    val_dataset = val_test_dataset.take(int(val_ratio*DATASET_SIZE))

    test_dataset = val_test_dataset.skip(int(test_ratio*DATASET_SIZE))

    return train_dataset, val_dataset, test_dataset

def create_datasets(dataset, train_ratio, val_ratio, test_ratio):
    train_dataset, val_dataset, test_dataset = splits(dataset[0], train_ratio, val_ratio, test_ratio)

    train_dataset = (
        train_dataset.shuffle(
        buffer_size=8, 
        reshuffle_each_iteration=True).batch(1).prefetch(tf.data.AUTOTUNE)
        )
    
    val_dataset = (
        val_dataset
        .shuffle(buffer_size=8, reshuffle_each_iteration=True)
        .batch(1)
        .prefetch(tf.data.AUTOTUNE)
        )
    
    test_dataset = (
        test_dataset.batch(1)
    )

    return train_dataset, val_dataset, test_dataset