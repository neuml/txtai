import unittest
import tempfile
import os
import numpy as np
from txtai.pipeline import LemurTrainer

class TestLemur(unittest.TestCase):
    """
    Test the LEMUR multi-vector training pipeline.
    """

    def test_pipeline(self):
        """
        Test that the LEMUR pipeline initializes, processes data, and saves the model.
        """
        try:
            # Initialize the true training pipeline
            lemur = LemurTrainer()

            # Generate dummy multi-vector data
            dummy_data = [
                np.random.rand(10, 64).astype(np.float32),
                np.random.rand(12, 64).astype(np.float32)
            ]

            # Create a temporary directory for the output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Execute the pipeline
                lemur(dummy_data, temp_dir)

                # Assert the absolute truth: The forge must have left the weapons on disk
                self.assertTrue(os.path.exists(os.path.join(temp_dir, "model.safetensors")), "Weights were not saved.")
                self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")), "Config was not saved.")
        
        except Exception as e:
            self.fail(f"LEMUR pipeline failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()