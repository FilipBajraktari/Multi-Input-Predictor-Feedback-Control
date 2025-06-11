from pathlib import Path
import h5py
import os

def create_downsampled_h5(original_path):

    # Create output filename by appending '_downsampled' before the extension
    dirname, basename = os.path.split(original_path)
    filename, ext = os.path.splitext(basename)
    new_filename = f"{filename}_200K_samples{ext}"
    output_path = os.path.join(dirname, new_filename)
    
    with h5py.File(original_path, 'r') as original_file, \
         h5py.File(output_path, 'w') as new_file:
        
        # Copy all attributes from the original file to the new file
        for attr_name, attr_value in original_file.attrs.items():
            new_file.attrs[attr_name] = attr_value
        
        # Determine which samples to copy (every 5th sample)
        samples_to_copy = [f"sample_{i:06d}" for i in range(0, 1000000, 5)]
        
        # Create a mapping from old sample names to new sample names
        sample_mapping = {
            old_name: f"sample_{i:06d}"
            for i, old_name in enumerate(samples_to_copy)
        }
        
        # Copy each selected sample group to the new file with new names
        for old_name in samples_to_copy:
            if old_name in original_file:
                new_name = sample_mapping[old_name]
                original_file.copy(old_name, new_file, name=new_name)
    
    print(f"Downsampled file created at: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default='data/UnicycleConstDistinctDelays_25_60_dt_001_1M_samples.h5')
    args = parser.parse_args()

    abs_path = (Path(__file__).parent.parent / args.filename).resolve()
    create_downsampled_h5(abs_path)