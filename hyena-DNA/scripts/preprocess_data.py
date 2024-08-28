import utilities as u
import importlib
from functools import partial
from pathlib import Path
import os

def preprocess_data_for_hyenaDNA(data_dir, species_name):
    print("Statrted the pre-processing process")
    data_dir = Path(data_dir)
    #### 1.2 Uncompress the files
    fastq_files = u.convert_directory(data_dir, suffix=".fq.gz",
                                      convertor=partial(u.gunzip_file,
                                                        suffix=".gz"),
                                      delete_orig_file=True)
    
    #### 1.3 Convert each FASTQ into an equivalent FASTA
    fasta_files = u.convert_directory(data_dir, suffix=".fq",
                                      convertor=u.convert_fastq_to_fasta,
                                      delete_orig_file=True)
    
    #### 1.4 Re-jigger the directory hierachy to match what HyenaDNA needs
    for child in data_dir.rglob("**/*"):
        if child.is_file():
            path = str(child).split("/")
            destination = Path("/".join(path[:-2] + [species_name]))
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / path[-1]  
            child.rename(target)  
            #print(f"Moved {child} to {target}")
    print("Completed the pre-processing of the data")
    return data_dir

        