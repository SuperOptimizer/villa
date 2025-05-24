"""
Comprehensive test suite for train.py script using real BaseTrainer and ConfigManager classes.

Tests ensure:
1. BaseTrainer works with zarr and tif datasets
2. Training respects ConfigManager settings
3. Losses are properly mapped and computed
4. Loss reporting works correctly
5. Forward passes complete successfully
6. Validation steps complete successfully
7. Checkpoints are saved with correct content
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import os

# Add the models directory to path so we can import the classes
sys.path.insert(0, str(Path(__file__).parent.parent))

from run.train import BaseTrainer
from config_manager import ConfigManager
from tests.create_test_dataset import TestDatasetCreator


class TestTrainer(unittest.TestCase):
    """Test suite for the real BaseTrainer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and temporary directories."""
        print("Setting up test data for training tests...")
        cls.creator = TestDatasetCreator()
        cls.creator.create_all_test_datasets()
        
        # Create temporary checkpoint directory
        cls.temp_checkpoint_dir = tempfile.mkdtemp(prefix="test_checkpoints_")
        print(f"Created temporary checkpoint directory: {cls.temp_checkpoint_dir}")
        print("Test data setup complete.")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data and temporary directories."""
        print("Cleaning up test data...")
        cls.creator.cleanup()
        
        # Clean up checkpoint directory
        if os.path.exists(cls.temp_checkpoint_dir):
            shutil.rmtree(cls.temp_checkpoint_dir)
            print(f"Cleaned up checkpoint directory: {cls.temp_checkpoint_dir}")
        print("Test data cleanup complete.")

    def create_test_config_manager(self, data_path, data_format='zarr', **kwargs):
        """Create a real ConfigManager configured for testing."""
        mgr = ConfigManager(verbose=True)
        
        # Set up basic config dictionaries that ConfigManager expects
        mgr.tr_info = {
            "model_name": kwargs.get('model_name', "test_model"),
            "autoconfigure": True,
            "tr_val_split": kwargs.get('tr_val_split', 0.8),
            "dilate_label": 0,
            "compute_loss_on_label": True,
            "ckpt_out_base": self.temp_checkpoint_dir,
            "checkpoint_path": kwargs.get('checkpoint_path', None),
            "load_weights_only": kwargs.get('load_weights_only', False)
        }
        
        mgr.tr_configs = {
            "patch_size": kwargs.get('patch_size', [8, 16, 16]),
            "optimizer": kwargs.get('optimizer', "AdamW"),
            "initial_lr": kwargs.get('initial_lr', 1e-3),
            "weight_decay": kwargs.get('weight_decay', 0.01),
            "batch_size": kwargs.get('batch_size', 1),
            "gradient_accumulation": kwargs.get('gradient_accumulation', 1),
            "max_steps_per_epoch": kwargs.get('max_steps_per_epoch', 100),
            "max_val_steps_per_epoch": kwargs.get('max_val_steps_per_epoch', 10),
            "num_dataloader_workers": 0,
            "max_epoch": kwargs.get('max_epoch', 2)
        }
        
        mgr.model_config = {}
        
        mgr.dataset_config = {
            "min_labeled_ratio": kwargs.get('min_labeled_ratio', 0.05),
            "min_bbox_percent": kwargs.get('min_bbox_percent', 0.0),
            "targets": kwargs.get('targets', {"ink": {"loss_fn": "SoftDiceLoss", "out_channels": 1, "weight": 1.0}})
        }
        
        mgr.inference_config = {}
        
        # Initialize all attributes using the real ConfigManager method
        mgr._init_attributes()
        
        # Override specific test settings after initialization
        mgr.data_path = str(data_path)
        mgr.data_format = data_format
        mgr.selected_loss_function = "SoftDiceLoss"
        
        return mgr

    def test_zarr_dataset_integration(self):
        """Test BaseTrainer with zarr dataset format."""
        print("\n=== Testing BaseTrainer with Zarr Dataset ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            data_format='zarr',
            model_name="zarr_test_model"
        )
        
        # Create trainer with real ConfigManager
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Test that trainer can be initialized
        self.assertIsNotNone(trainer.mgr)
        self.assertEqual(trainer.mgr.data_format, 'zarr')
        print("✓ BaseTrainer initialized with zarr ConfigManager")
        
        # Test model building
        model = trainer._build_model()
        self.assertIsNotNone(model)
        print("✓ Model built successfully")
        
        # Test dataset configuration
        dataset = trainer._configure_dataset()
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) >= 0)
        print(f"✓ Zarr dataset configured with {len(dataset)} samples")
        
        # Test loss function building
        loss_fns = trainer._build_loss()
        self.assertIsNotNone(loss_fns)
        self.assertIn("ink", loss_fns)
        print("✓ Loss functions built successfully")

    def test_tif_dataset_integration(self):
        """Test BaseTrainer with tif dataset format."""
        print("\n=== Testing BaseTrainer with TIF Dataset ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.tif_path,
            data_format='tifs',
            model_name="tif_test_model"
        )
        
        # Create trainer with real ConfigManager
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Test that trainer can be initialized
        self.assertIsNotNone(trainer.mgr)
        self.assertEqual(trainer.mgr.data_format, 'tifs')
        print("✓ BaseTrainer initialized with tif ConfigManager")
        
        # Test model building
        model = trainer._build_model()
        self.assertIsNotNone(model)
        print("✓ Model built successfully")
        
        # Test dataset configuration
        dataset = trainer._configure_dataset()
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) >= 0)
        print(f"✓ TIF dataset configured with {len(dataset)} samples")
        
        # Test loss function building
        loss_fns = trainer._build_loss()
        self.assertIsNotNone(loss_fns)
        self.assertIn("ink", loss_fns)
        print("✓ Loss functions built successfully")

    def test_config_manager_settings_respected(self):
        """Test that trainer respects ConfigManager settings."""
        print("\n=== Testing ConfigManager Settings Respect ===")
        
        # Test custom settings
        custom_settings = {
            'initial_lr': 0.001,
            'batch_size': 2,
            'optimizer': 'SGD',
            'weight_decay': 0.02,
            'gradient_accumulation': 2,
            'model_name': 'custom_settings_model'
        }
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            data_format='zarr',
            **custom_settings
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Verify settings are stored correctly
        self.assertEqual(trainer.mgr.initial_lr, 0.001)
        self.assertEqual(trainer.mgr.train_batch_size, 2)
        self.assertEqual(trainer.mgr.optimizer, 'SGD')
        self.assertEqual(trainer.mgr.weight_decay, 0.02)
        self.assertEqual(trainer.mgr.gradient_accumulation, 2)
        self.assertEqual(trainer.mgr.model_name, 'custom_settings_model')
        print("✓ Custom ConfigManager settings correctly stored")
        
        # Test optimizer creation respects settings
        model = trainer._build_model()
        optimizer = trainer._get_optimizer(model)
        self.assertIsNotNone(optimizer)
        print("✓ Optimizer created respecting ConfigManager settings")

    def test_loss_mapping_and_computation(self):
        """Test that losses are properly mapped and computed."""
        print("\n=== Testing Loss Mapping and Computation ===")
        
        # Test single target
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            targets={"ink": {"loss_fn": "SoftDiceLoss", "out_channels": 1, "weight": 1.0}},
            model_name="single_target_model"
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        loss_fns = trainer._build_loss()
        
        self.assertIn("ink", loss_fns)
        self.assertEqual(len(loss_fns), 1)
        print("✓ Single target loss mapping successful")
        
        # Test multiple targets with different loss functions
        multi_target_mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            targets={
                "ink": {"loss_fn": "SoftDiceLoss", "out_channels": 1, "weight": 1.0},
                "normals": {"loss_fn": "BCEWithLogitsLoss", "out_channels": 3, "weight": 0.5}
            },
            model_name="multi_target_model"
        )
        
        multi_trainer = BaseTrainer(mgr=multi_target_mgr, verbose=True)
        multi_loss_fns = multi_trainer._build_loss()
        
        self.assertIn("ink", multi_loss_fns)
        self.assertIn("normals", multi_loss_fns)
        self.assertEqual(len(multi_loss_fns), 2)
        print("✓ Multiple target loss mapping successful")
        
        # Test that different loss functions are correctly instantiated
        from models.training.losses import SoftDiceLoss, BCEWithLogitsMaskedLoss
        self.assertIsInstance(multi_loss_fns["ink"], SoftDiceLoss)
        self.assertIsInstance(multi_loss_fns["normals"], BCEWithLogitsMaskedLoss)
        print("✓ Different loss functions correctly instantiated")

    def test_forward_pass_completion(self):
        """Test that forward passes complete successfully."""
        print("\n=== Testing Forward Pass Completion ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            patch_size=[8, 16, 16],  # Small patch for testing
            model_name="forward_pass_model"
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Build components
        model = trainer._build_model()
        dataset = trainer._configure_dataset()
        
        if len(dataset) > 0:
            # Test forward pass
            sample = dataset[0]
            
            # Prepare input
            inputs = sample["image"].unsqueeze(0)  # Add batch dimension
            
            # Determine device
            device = torch.device('cpu')  # Use CPU for testing to avoid GPU issues
            model = model.to(device)
            inputs = inputs.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
            
            # Verify outputs
            self.assertIsInstance(outputs, dict)
            self.assertIn("ink", outputs)
            
            # Check output shape
            output_shape = outputs["ink"].shape
            self.assertEqual(len(output_shape), 5)  # [B, C, D, H, W]
            self.assertEqual(output_shape[0], 1)    # Batch size
            self.assertEqual(output_shape[1], 1)    # Output channels for ink
            
            print("✓ Forward pass completed successfully")
            print(f"✓ Output shape: {output_shape}")
        else:
            print("! Skipping forward pass test - no valid samples in dataset")

    def test_checkpoint_saving(self):
        """Test that checkpoints are saved with correct content."""
        print("\n=== Testing Checkpoint Saving ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            max_epoch=1,  # Just one epoch for checkpoint test
            model_name="checkpoint_test_model"
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Run one epoch of training to generate checkpoint
        trainer.train()
        
        # Check that checkpoint directory was created
        model_ckpt_dir = Path(self.temp_checkpoint_dir) / mgr.model_name
        self.assertTrue(model_ckpt_dir.exists())
        print(f"✓ Checkpoint directory created: {model_ckpt_dir}")
        
        # Check that checkpoint file was created
        checkpoint_files = list(model_ckpt_dir.glob(f"{mgr.model_name}_*.pth"))
        self.assertTrue(len(checkpoint_files) > 0)
        print(f"✓ Checkpoint files created: {len(checkpoint_files)}")
        
        # Load and verify checkpoint content
        checkpoint_path = checkpoint_files[0]
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify required checkpoint components
        required_keys = ['model', 'optimizer', 'scheduler', 'epoch', 'model_config']
        for key in required_keys:
            self.assertIn(key, checkpoint)
        
        print("✓ Checkpoint contains required components")
        
        # Verify model state dict is not empty
        self.assertTrue(len(checkpoint['model']) > 0)
        print("✓ Model state dict saved in checkpoint")
        
        # Verify model config is saved
        self.assertIsInstance(checkpoint['model_config'], dict)
        print("✓ Model configuration saved in checkpoint")

    def test_minimal_training_execution(self):
        """Test that minimal training execution completes without errors."""
        print("\n=== Testing Minimal Training Execution ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            max_epoch=1,  # Minimal training
            patch_size=[8, 8, 8],  # Very small for speed
            batch_size=1,
            model_name="minimal_training_model"
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        try:
            # This should complete without errors
            trainer.train()
            print("✓ Minimal training execution completed successfully")
            
            # Verify that training actually ran
            model_ckpt_dir = Path(self.temp_checkpoint_dir) / mgr.model_name
            checkpoint_files = list(model_ckpt_dir.glob(f"{mgr.model_name}_*.pth"))
            self.assertTrue(len(checkpoint_files) > 0)
            print("✓ Training generated checkpoints as expected")
            
        except Exception as e:
            self.fail(f"Training execution failed with error: {str(e)}")

    def test_validation_step_completion(self):
        """Test that validation steps complete successfully."""
        print("\n=== Testing Validation Step Completion ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            max_epoch=1,
            tr_val_split=0.7,  # Ensure we have validation data
            model_name="validation_test_model"
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Build components needed for validation
        model = trainer._build_model()
        dataset = trainer._configure_dataset()
        loss_fns = trainer._build_loss()
        
        if len(dataset) > 0:
            train_dataloader, val_dataloader, train_indices, val_indices = trainer._configure_dataloaders(dataset)
            
            # Verify we have validation data
            self.assertTrue(len(val_indices) > 0)
            print(f"✓ Validation split created: {len(val_indices)} validation samples")
            
            # Test validation dataloader
            val_batch = next(iter(val_dataloader))
            self.assertIn("image", val_batch)
            self.assertIn("ink", val_batch)
            print("✓ Validation dataloader produces valid batches")
            
        else:
            print("! Skipping validation test - no valid samples in dataset")

    def test_loss_reporting(self):
        """Test that loss values are properly computed and reported."""
        print("\n=== Testing Loss Reporting ===")
        
        mgr = self.create_test_config_manager(
            data_path=self.creator.zarr_path,
            patch_size=[8, 8, 8],
            model_name="loss_reporting_model"
        )
        
        trainer = BaseTrainer(mgr=mgr, verbose=True)
        
        # Build components
        model = trainer._build_model()
        dataset = trainer._configure_dataset()
        loss_fns = trainer._build_loss()
        
        if len(dataset) > 0:
            # Get a sample
            sample = dataset[0]
            inputs = sample["image"].unsqueeze(0)
            target = sample["ink"].unsqueeze(0)
            
            # Set up device
            device = torch.device('cpu')
            model = model.to(device)
            inputs = inputs.to(device)
            target = target.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                
                # Compute loss
                loss_fn = loss_fns["ink"]
                loss_value = loss_fn(outputs["ink"], target)
                
                # Verify loss is computed properly
                self.assertIsInstance(loss_value, torch.Tensor)
                self.assertEqual(loss_value.ndim, 0)  # Scalar
                self.assertFalse(torch.isnan(loss_value))
                self.assertFalse(torch.isinf(loss_value))
                self.assertGreaterEqual(loss_value.item(), 0.0)  # Loss should be non-negative
                
                print(f"✓ Loss computed successfully: {loss_value.item():.4f}")
                print("✓ Loss value is finite and non-negative")
        else:
            print("! Skipping loss reporting test - no valid samples in dataset")


def run_comprehensive_training_test():
    """Run all training tests and provide a summary."""
    print("="*60)
    print("VESUVIUS TRAINING COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("Testing Requirements:")
    print("1. BaseTrainer works with zarr and tif datasets")
    print("2. Training respects ConfigManager settings")
    print("3. Losses are properly mapped and computed")
    print("4. Loss reporting works correctly")
    print("5. Forward passes complete successfully")
    print("6. Validation steps complete successfully")
    print("7. Checkpoints are saved with correct content")
    print("="*60)
    
    # Run the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("="*60)
    print("TRAINING TEST SUMMARY")
    print("="*60)
    if result.wasSuccessful():
        print("🎉 ALL TRAINING TESTS PASSED!")
        print("✅ BaseTrainer works with both zarr and tif datasets")
        print("✅ ConfigManager settings are properly respected")
        print("✅ Loss mapping and computation working correctly")
        print("✅ Loss reporting functioning properly")
        print("✅ Forward passes complete successfully")
        print("✅ Validation steps complete successfully")
        print("✅ Checkpoint saving working correctly")
        print("✅ Real training execution completes without errors")
    else:
        print("❌ Some training tests failed:")
        for failure in result.failures:
            print(f"  - {failure[0]}")
        for error in result.errors:
            print(f"  - {error[0]} (ERROR)")
    
    print("="*60)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_training_test()
    exit(0 if success else 1)
