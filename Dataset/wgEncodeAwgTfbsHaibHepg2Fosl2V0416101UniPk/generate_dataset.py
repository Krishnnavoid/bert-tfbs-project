import os
import glob
import random
from pyfaidx import Fasta

def get_sample_names(narrowpeak_dir):
    """
    Finds all .narrowPeak files in the specified folder and parses filenames.
    """
    peak_files = glob.glob(os.path.join(narrowpeak_dir, "*.narrowPeak"))
    samples = []
    for filepath in peak_files:
        filename = os.path.basename(filepath)
        samples.append({
            'filepath': filepath,
            'filename': filename
        })
    return samples

import random

def generate_samples(narrowpeak_file, fasta_ref, window_size=200):
    records = []
    half_win = window_size // 2
    
    with open(narrowpeak_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            
            # Handle chromosome prefix mismatches
            if chrom not in fasta_ref:
                if chrom.startswith('chr'):
                    chrom_alt = chrom[3:]
                else:
                    chrom_alt = f"chr{chrom}"
                if chrom_alt in fasta_ref:
                    chrom = chrom_alt
                else:
                    continue
            
            # Calculate peak center
            center = (start + end) // 2
            seq_start = max(0, center - half_win)
            seq_end = center + half_win
            
            try:
                # 1. Positive Sequence (Label = 1)
                pos_sequence = str(fasta_ref[chrom][seq_start:seq_end]).upper()
                
                if len(pos_sequence) == window_size and set(pos_sequence).issubset(set('ACGTN')):
                    records.append((chrom, pos_sequence, 1))
                    
                    # 2. Negative Sequence via Random Permutation (Label = 0)
                    seq_list = list(pos_sequence)
                    random.shuffle(seq_list)
                    neg_sequence = "".join(seq_list)
                    
                    records.append((chrom, neg_sequence, 0))
            except KeyError:
                continue

    return records

def write_sample(samples, output_file):
    """
    Writes extracted records in space-delimited format: (name sequence label)
    """
    dirname = os.path.dirname(output_file)
    if dirname:  # Only create directory if a parent path actually exists
        os.makedirs(dirname, exist_ok=True)
        
    with open(output_file, 'w') as f:
        for name, seq, label in samples:
            f.write(f"{name} {seq} {label}\n")
            
    print(f"Successfully saved {len(samples)} samples to: {output_file}")

if __name__ == "__main__":
    fasta_path = r"C:\Users\Anshu\Downloads\hg19.fa\hg19.fa"
    narrowpeak_path = r"C:\Users\Anshu\Downloads\wgEncodeAwgTfbsHaibA549Ctcfsc5916Pcr1xDex100nmUniPk.narrowPeak\wgEncodeAwgTfbsHaibA549Ctcfsc5916Pcr1xDex100nmUniPk.narrowPeak"
    # 1. Load genome and extract samples
    fasta_ref = Fasta(fasta_path)
    samples = generate_samples(narrowpeak_path, fasta_ref)
    
    # 2. Randomly shuffle samples (set seed for reproducibility if needed)
    random.seed(41)  
    random.shuffle(samples)
    
    # 3. Calculate split index (80% Train, 20% Test)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # 4. Save both splits
    write_sample(train_samples, "train_AwgTfbsHaibA549Ctcfsc5916Pcr1xDex100nmUniPk_IN_NIGHT_NO_CONTROL.data")
    write_sample(test_samples, "test_AwgTfbsHaibA549Ctcfsc5916Pcr1xDex100nmUniPk_IN_NIGHT_NO_CONTROL.data")