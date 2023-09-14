from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import tensorflow_probability as tfp

BATCH_SIZE = 32
IM_SIZE = 224
TARGET_SIZE = (IM_SIZE, IM_SIZE)

def box(lamda):
    r_x = tf.cast(tfp.distributions.Uniform(0, IM_SIZE).sample(1)[0], dtype=tf.int32)
    r_y = tf.cast(tfp.distributions.Uniform(0, IM_SIZE).sample(1)[0], dtype=tf.int32)

    r_w = tf.cast(IM_SIZE * tf.math.sqrt(1-lamda), dtype=tf.int32)
    r_h = tf.cast(IM_SIZE * tf.math.sqrt(1-lamda), dtype=tf.int32)

    r_x = tf.clip_by_value(r_x - r_w // 2, 0, IM_SIZE)
    r_y = tf.clip_by_value(r_y - r_h // 2, 0, IM_SIZE)

    x_bottom_right = tf.clip_by_value(r_x + r_w // 2, 0, IM_SIZE)
    y_bottom_right = tf.clip_by_value(r_y + r_h // 2, 0, IM_SIZE)

    r_w = x_bottom_right - r_x
    if r_w == 0:
        r_w = 1

    r_h = y_bottom_right - r_y
    if r_h == 0:
        r_h = 1

    return r_y, r_x, r_h, r_w


def cutmix(train_dataset_1, train_dataset_2):
    (image_1, label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

    lamda = tfp.distributions.Beta(0.2, 0.2)
    lamda = lamda.sample(1)[0]

    r_y, r_x, r_h, r_w = box(lamda)

    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, IM_SIZE, IM_SIZE)

    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, IM_SIZE, IM_SIZE)

    image = image_1 - pad_1 + pad_2

    lamda = tf.cast(1 - (r_w*r_h)/(pow(IM_SIZE, 2)), dtype=tf.float32)
    label = lamda * tf.cast(label_1, dtype=tf.float32) + (1-lamda) * tf.cast(label_2, dtype=tf.float32)

    return image, label
    
def apply_cutmix(generator):
    while True:
        batch_x, batch_y = next(generator)

        # Initialize empty lists to store the mixed batch
        mixed_batch_x = []
        mixed_batch_y = []

        # Apply cutmix to each pair of images and labels in the batch
        for i in range(0, len(batch_x), 2):
            image_1, label_1 = batch_x[i], batch_y[i]
            image_2, label_2 = batch_x[i + 1], batch_y[i + 1]

            # Apply cutmix to the pair of images and labels
            mixed_image, mixed_label = cutmix((image_1, label_1), (image_2, label_2))

            # Append the mixed data to the batch
            mixed_batch_x.append(mixed_image)
            mixed_batch_y.append(mixed_label)

        # Stack the mixed data to create a batch with a consistent shape
        mixed_batch_x = tf.stack(mixed_batch_x, axis=0)
        mixed_batch_y = tf.stack(mixed_batch_y, axis=0)

        yield mixed_batch_x, mixed_batch_y

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
    
    train_generator_with_cutmix = apply_cutmix(train_generator)
    
    return train_generator, train_generator_with_cutmix, val_generator, test_generator