# This script acts as the "Brain" of my live application.
#  Instead of redefining the whole U-Net architecture,
#  we simply load the pre-trained .h5 file we already created.
#  It uses an Object-Oriented approach (a Python Class)
#  so the heavy model is only loaded into my computer's RAM once when the server starts,
#  not every time a user clicks the button.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


def iou_metric(y_true, y_pred):
    """Calculates the Intersection over Union (IoU) metric."""
    # Convert the predicted probabilities (0.0 to 1.0) into a strict binary mask (0 or 1)
    y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())

    # Calculate intersection and union
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    # Add a tiny epsilon to prevent division by zero
    return (intersection + K.epsilon()) / (union + K.epsilon())

class WaterSegmentationModel:
    def __init__(self, model_path:str):
        """
        Initializes the class and loads the heavy deep learning model into memory. 
        """
        print(f"Loading U-Net model from {model_path}...")
        self.model = load_model(
            model_path,
            custom_objects={"iou_metric": iou_metric}
        )
        print("Model loaded successfully!")


    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        Takes a normalized 12-channel numpy array, runs inference, 
        and returns a 2D binary mask (0s and 1s).
        
        Args:
            image_array (np.ndarray): Shape must be (128, 128, 12)
            
        Returns:
            np.ndarray: Binary mask of shape (128, 128)
        """

        # Expnad the dimension of the image_array from (128, 128, 12) --> (1,128, 128, 12) by but 1 index of 0
        # why ? 
        # because the U-Net take a batch of images for our realtime prediction it expect a batch of images
        # so we tell the U-Net to take a batch of one image 
        input_tensor = np.expand_dims(image_array, axis=0)

        pred_prob = self.model.predict(input_tensor, verbose=0)[0] # Get prediction for one image of the batch (input image)

        # Set a threshold to get the confidence
        binary_mask = (pred_prob > 0.5).astype(np.uint8)

        # Squeeze the dimension of the mask from (128,128,1) --> (128,128)
        return np.squeeze(binary_mask)


# # Test 
# if __name__ == "__main__":
#     import os
#     import tifffile
#     import cv2
#     import matplotlib.pyplot as plt
#     import numpy as np
#     # 1. Setup Paths
#     BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
#     MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "weights", "U-Net_Model.h5")
    
#     # ⚠️ CHANGE THIS to the actual path of one of your test images
#     TEST_IMAGE_PATH = os.path.join(BASE_DIR, "ml_pipeline/Data/preprocessed_data/images/3.tif")

#     print("--- 🧪 Starting AI Engine Test with Real Data ---")
    
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ ERROR: Model not found at {MODEL_PATH}")
#     elif not os.path.exists(TEST_IMAGE_PATH):
#         print(f"❌ ERROR: Please place a test .tif file at: {TEST_IMAGE_PATH}")
#     else:
#         # 2. Initialize the AI
#         ai_engine = WaterSegmentationModel(model_path=MODEL_PATH)
        
#         # 3. Load the Real Image
#         print(f"\nLoading real satellite image from {TEST_IMAGE_PATH}...")
#         raw_image = tifffile.imread(TEST_IMAGE_PATH)
#         print(f"Original Shape: {raw_image.shape}")
        
#         # 4. Preprocess (Resize and Normalize)
#         # Ensure it fits the exact (128, 128) geometry the U-Net expects
#         if raw_image.shape[:2] != (128, 128):
#             print("Resizing image to (128, 128)...")
#             raw_image = cv2.resize(raw_image, (128, 128), interpolation=cv2.INTER_LINEAR)
        
#         # Apply the same Min-Max Normalization used in your training pipeline
          #   raw_float = raw_image.astype(np.float32)
          #   normalized_image = (raw_float - np.min(raw_float)) / (np.max(raw_float) - np.min(raw_float) + 1e-8)        
#         # 5. Run Inference
#         print("Running inference on real data...")
#         prediction_mask = ai_engine.predict(normalized_image)
        
#         # 6. Visualize and Save the Results
#         print("Saving visual comparison to 'test_output.png'...")
#         plt.figure(figsize=(10, 5))
        
#         # Plotting Band 8 (Near Infrared) as it usually highlights water boundaries well
#         plt.subplot(1, 2, 1)
#         plt.title("Original Satellite (Near-Infrared Band)")
#         plt.imshow(normalized_image[:, :, 7], cmap='gray') 
        
#         # Plotting your model's prediction
#         plt.subplot(1, 2, 2)
#         plt.title("U-Net Predicted Water Mask")
#         plt.imshow(prediction_mask, cmap='Blues', vmin=0, vmax=1)
        
#         # Save it to your hard drive so you can open and look at it
#         output_file = os.path.join(BASE_DIR, "assets/test_output.png")
#         plt.savefig(output_file)
#         print(f"✅ Test Complete! Open '{output_file}' to see your AI in action.")
#         print(f"Unique values in mask: {np.unique(prediction_mask)}")