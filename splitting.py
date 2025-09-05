import os
import shutil
import random

def create_final_data_split(source_dir, output_dir, train_split=0.8):
    """Create FinalData with 80/20 train/test split within each class."""
    
    random.seed(42)
    
    final_data_path = os.path.join(output_dir, 'FinalData')
    
    # Remove existing and create fresh structure
    if os.path.exists(final_data_path):
        shutil.rmtree(final_data_path)
    
    os.makedirs(final_data_path)
    
    print(f"Creating FinalData at: {final_data_path}")
    print("=" * 50)
    
    total_train = 0
    total_test = 0
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.WAV', '.MP3', '.FLAC', '.M4A', '.OGG']
    
    # Process each class
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Get audio files
        audio_files = [f for f in os.listdir(class_path) 
                      if os.path.isfile(os.path.join(class_path, f)) and 
                      any(f.endswith(ext) for ext in audio_extensions)]
        
        if not audio_files:
            print(f"⚠️  No audio files in {class_name}")
            continue
        
        # Create class directory with train/test subdirs
        class_final_path = os.path.join(final_data_path, class_name)
        train_class_path = os.path.join(class_final_path, 'train')
        test_class_path = os.path.join(class_final_path, 'test')
        
        os.makedirs(train_class_path)
        os.makedirs(test_class_path)
        
        # Split files
        random.shuffle(audio_files)
        split_point = int(len(audio_files) * train_split)
        train_files = audio_files[:split_point]
        test_files = audio_files[split_point:]
        
        # Copy files
        for file in train_files:
            shutil.copy2(os.path.join(class_path, file), 
                        os.path.join(train_class_path, file))
        
        for file in test_files:
            shutil.copy2(os.path.join(class_path, file), 
                        os.path.join(test_class_path, file))
        
        total_train += len(train_files)
        total_test += len(test_files)
        
        print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")
    
    print("=" * 50)
    print(f"COMPLETE: {total_train} train ({total_train/(total_train+total_test)*100:.1f}%), {total_test} test ({total_test/(total_train+total_test)*100:.1f}%)")
    return final_data_path

def verify_split(final_data_path):
    """Quick verification of the split."""
    
    if not os.path.exists(final_data_path):
        print("❌ FinalData not found")
        return
    
    total_train = 0
    total_test = 0
    
    for class_name in os.listdir(final_data_path):
        class_path = os.path.join(final_data_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        train_path = os.path.join(class_path, 'train')
        test_path = os.path.join(class_path, 'test')
        
        train_count = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        test_count = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
        
        total_train += train_count
        total_test += test_count
        
        print(f"{class_name}: {train_count}/{train_count+test_count} = {train_count/(train_count+test_count)*100:.0f}% train")
    
    print(f"Overall: {total_train/(total_train+total_test)*100:.1f}% train, {total_test/(total_train+total_test)*100:.1f}% test")

if __name__ == "__main__":
    source_directory = "/Users/gemwincanete/Thesis /datasets/Processed_Data_(Resampled)"
    output_directory = "/Users/gemwincanete/Thesis /datasets"
    
    print("Heart Sound Dataset - 80/20 Split")
    print("=" * 50)
    
    try:
        final_data_path = create_final_data_split(source_directory, output_directory, train_split=0.8)
        verify_split(final_data_path)
        print(f"✅ Done: {final_data_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")