import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import shutil

# Set config
target_sr = 8000
n_mfcc = 13
n_fft = 400

# Dataset paths
merged_base = '/Users/gemwincanete/Thesis /datasets/merged_data'
output_base = '/Users/gemwincanete/Thesis /datasets/mfccfeatures'

# Clean output directory before running
if os.path.exists(output_base):
    shutil.rmtree(output_base)
os.makedirs(output_base)

# Initialize metadata list
metadata = []

class_folders = [d for d in os.listdir(merged_base) if os.path.isdir(os.path.join(merged_base, d))]

for class_name in class_folders:
    class_dir = os.path.join(merged_base, class_name)

    # Detect if this class directory contains split subfolders (train/test)
    subdirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
    split_subdirs = [d for d in subdirs if d in ['train', 'test']]

    # If split subdirs exist, iterate over them and maintain split in output
    if len(split_subdirs) > 0:
        for split in split_subdirs:
            input_dir = os.path.join(class_dir, split)
            output_dir = os.path.join(output_base, class_name, split)
            os.makedirs(output_dir, exist_ok=True)

            wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

            for wav_file in tqdm(wav_files, desc=f'Extracting {class_name} [{split}]'):
                try:
                    audio_path = os.path.join(input_dir, wav_file)
                    y, sr = librosa.load(audio_path, sr=target_sr)

                    # Extract MFCC
                    base_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
                    # Compute delta and delta-delta
                    delta = librosa.feature.delta(base_mfcc)
                    delta2 = librosa.feature.delta(base_mfcc, order=2)
                    # Combine all features into one array (shape: (39, time_steps))
                    mfcc = np.vstack([base_mfcc, delta, delta2])

                    # Save MFCC feature (no source prefix, keep name intact)
                    feature_filename = f"{os.path.splitext(wav_file)[0]}.npy"
                    feature_path = os.path.join(output_dir, feature_filename)
                    np.save(feature_path, mfcc)

                    # Record metadata
                    metadata.append({
                        'filename': feature_filename,
                        'label': class_name,
                        'split': split
                    })

                except Exception as e:
                    print(f"Failed: {wav_file} → {e}")
    else:
        # No split subdirs; process files directly in class directory
        input_dir = class_dir
        output_dir = os.path.join(output_base, class_name)
        os.makedirs(output_dir, exist_ok=True)

        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

        for wav_file in tqdm(wav_files, desc=f'Extracting {class_name}'):
            try:
                audio_path = os.path.join(input_dir, wav_file)
                y, sr = librosa.load(audio_path, sr=target_sr)

                # Extract MFCC
                base_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
                # Compute delta and delta-delta
                delta = librosa.feature.delta(base_mfcc)
                delta2 = librosa.feature.delta(base_mfcc, order=2)
                # Combine all features into one array (shape: (39, time_steps))
                mfcc = np.vstack([base_mfcc, delta, delta2])

                # Save MFCC feature
                feature_filename = f"{os.path.splitext(wav_file)[0]}.npy"
                feature_path = os.path.join(output_dir, feature_filename)
                np.save(feature_path, mfcc)

                # Record metadata
                metadata.append({
                    'filename': feature_filename,
                    'label': class_name,
                    'split': ''
                })

            except Exception as e:
                print(f"Failed: {wav_file} → {e}")

# Save metadata to CSV
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(output_base, 'metadata.csv'), index=False)

print(f"\n✅ MFCC feature extraction complete. Metadata saved to {output_base}/metadata.csv")