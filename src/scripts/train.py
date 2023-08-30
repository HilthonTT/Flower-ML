import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..")
sys.path.append(src_dir)

from models.model_architecture import create_model
from models.model_utils import compile_model, save_model_checkpoint
from data.data_processor import load_data

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def main():
    EPOCHS = 20

    train_generator, val_generator, test_generator = load_data()

    model = create_model()
    compile_model(model)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1,
        validation_steps=len(val_generator))
    
    save_model_checkpoint(model)

if __name__ == "__main__":
    main()