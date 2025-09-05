import os
import shutil
from pathlib import Path

def merge_augmented_data():
    """
    Merge original data with augmented data for each class.
    Creates train and test folders in merged_data directory.
    """
    
    # Base paths
    datasets_path = Path("/Users/gemwincanete/Thesis /datasets")
    original_data_path = datasets_path / "FinalData"
    augmented_data_path = datasets_path / "PCGmix_Plus_Augmented_Data"
    merged_data_path = datasets_path / "merged_data"
    
    # Classes to process
    classes = ["normal", "murmur", "extra_systole", "extra_heart_audio", "artifact"]
    
    print("Starting data merging process...")
    print(f"Original data path: {original_data_path}")
    print(f"Augmented data path: {augmented_data_path}")
    print(f"Output path: {merged_data_path}")
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Create train and test directories for this class
        class_train_path = merged_data_path / class_name / "train"
        class_test_path = merged_data_path / class_name / "test"
        
        class_train_path.mkdir(parents=True, exist_ok=True)
        class_test_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Copy train data from FinalData directory
        # Try different possible directory structures
        possible_train_paths = [
            original_data_path / class_name / "Train",
            original_data_path / class_name / "train",
            original_data_path / class_name / "Training"
        ]
        
        train_files_copied = 0
        for original_train_path in possible_train_paths:
            if original_train_path.exists():
                print(f"  Copying original train data from {original_train_path}")
                for file_path in original_train_path.glob("*.wav"):
                    if file_path.is_file():
                        dest_path = class_train_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                        train_files_copied += 1
                print(f"  Copied {train_files_copied} original train files")
                break
        else:
            print(f"  Warning: No train directory found for {class_name} in original data")
        
        # Step 2: Copy test data from FinalData directory
        possible_test_paths = [
            original_data_path / class_name / "Test",
            original_data_path / class_name / "test",
            original_data_path / class_name / "Testing"
        ]
        
        test_files_copied = 0
        for original_test_path in possible_test_paths:
            if original_test_path.exists():
                print(f"  Copying original test data from {original_test_path}")
                for file_path in original_test_path.glob("*.wav"):
                    if file_path.is_file():
                        dest_path = class_test_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                        test_files_copied += 1
                print(f"  Copied {test_files_copied} original test files")
                break
        else:
            print(f"  Warning: No test directory found for {class_name} in original data")
        
        # Step 3: Copy augmented data to train folder
        aug_class_path = augmented_data_path / class_name
        aug_files_copied = 0
        
        if aug_class_path.exists():
            print(f"  Copying augmented data from {aug_class_path}")
            for file_path in aug_class_path.glob("*.wav"):
                if file_path.is_file():
                    # Add prefix to avoid filename conflicts
                    dest_path = class_train_path / f"aug_{file_path.name}"
                    shutil.copy2(file_path, dest_path)
                    aug_files_copied += 1
            print(f"  Copied {aug_files_copied} augmented files")
        else:
            print(f"  Warning: No augmented data found for {class_name} at {aug_class_path}")
        
        # Count final files
        train_files = len(list(class_train_path.glob("*.wav")))
        test_files = len(list(class_test_path.glob("*.wav")))
        
        print(f"  Final counts for {class_name}:")
        print(f"    Train: {train_files} files (original: {train_files_copied}, augmented: {aug_files_copied})")
        print(f"    Test: {test_files} files")
    
    print("\nData merging completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total_train = 0
    total_test = 0
    
    for class_name in classes:
        class_train_path = merged_data_path / class_name / "train"
        class_test_path = merged_data_path / class_name / "test"
        
        train_count = len(list(class_train_path.glob("*.wav"))) if class_train_path.exists() else 0
        test_count = len(list(class_test_path.glob("*.wav"))) if class_test_path.exists() else 0
        
        total_train += train_count
        total_test += test_count
        
        print(f"{class_name:20} | Train: {train_count:4d} | Test: {test_count:4d} | Total: {train_count + test_count:4d}")
    
    print("="*60)
    print(f"{'OVERALL TOTAL':20} | Train: {total_train:4d} | Test: {total_test:4d} | Total: {total_train + total_test:4d}")

if __name__ == "__main__":
    merge_augmented_data()