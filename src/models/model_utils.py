import os
from keras import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import (
    FalseNegatives, 
    FalsePositives, 
    TrueNegatives, 
    TruePositives, 
    Precision, 
    Recall, 
    AUC, 
    BinaryAccuracy
)

CHECKPOINT_PATH = "checkpoint/"

def compile_model(model: Model):
    metrics = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'), 
               BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=CategoricalCrossentropy(),
                  metrics=metrics)
    
def save_model_checkpoint(model: Model):
    model.save_weights(CHECKPOINT_PATH)

def load_model_checkpoint(model: Model):
    if os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
    else:
        print("[*] The path does not exist...")