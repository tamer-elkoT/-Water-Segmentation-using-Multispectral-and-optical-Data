# Create Our File path for the Tuned_UNet.h5

import os 
class Settings:
    # We dynamically find where we are currently located on the hard drive
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # We build the path to the model dynamically so when we deploy on docker or AWS the path not crashed
    MODEL_PATH = os.path.join(BASE_DIR,"ml_pipeline","weights", "Tuned_UNet.h5")

# instantiate the settings
settings = Settings()