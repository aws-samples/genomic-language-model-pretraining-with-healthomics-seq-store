"""
Commonly-used utilities are here, not in a notebook, to
make them easier to test.

To run unit-tests: python3 -m doctest utilities.py
(You may have to first 'pip3 install doctest'.)
"""

import io
from typing import BinaryIO, TextIO, List, Callable, Optional, Tuple
from pathlib import Path
import gzip
import re
import math
from contextlib import contextmanager
from time import time
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch


@contextmanager
def timing(message: bool, num_iterations: Optional[int] = None):
    start_time: float = time()
    yield
    duration = time() - start_time
    print(f"{message} took {duration/60:.2f} minutes", end="")
    if num_iterations:
        print(f", {(num_iterations/(duration/60)):.2f} iterations/minute")
    else:
        print()


def gunzip_file(gzip_file_path: Path, suffix: str) -> Path:
    assert gzip_file_path.suffix == suffix, gzip_file_path
    result_path = gzip_file_path.with_suffix("")
    result = gunzip_fileobj(io.BytesIO(gzip_file_path.read_bytes()))
    result_path.write_bytes(result.getvalue())
    return result_path


def gunzip_fileobj(gzip_fileobj: BinaryIO) -> BinaryIO:
    """
    >>> in_buf = io.BytesIO(b"abcdef")
    >>> zipped_bytes = gzip.compress(b"abcdef")
    >>> gunzip_fileobj(io.BytesIO(zipped_bytes)).getvalue()
    b'abcdef'
    """
    with gzip.open(gzip_fileobj, 'rb') as f_in:
        return io.BytesIO(f_in.read())


def convert_directory(root_dir: Path,
                      convertor: Callable[[Path], Path],
                      suffix: str = ".fq",
                      delete_orig_file: bool = False) -> List[Path]:
    result = []
    for orig_file in root_dir.rglob(f"*{suffix}"):
        new_file = convertor(orig_file)
        print(f"{str(orig_file)} -> {str(new_file)}")
        result.append(new_file)
        if delete_orig_file:
            orig_file.unlink()
            print(f"Deleted {str(orig_file)}")
    print("Done")
    return result


def gzip_file(file_path: Path) -> Path:
    assert file_path.suffix != ".gz", file_path
    result_path = file_path.parent / f"{file_path.name}.gz"
    # print(f"Compressing {str(file_path)} to {str(result_path)}")
    result = gzip_fileobj(io.StringIO(file_path.read_text()))
    result_path.write_bytes(result.getvalue())
    return result_path


def gzip_fileobj(fileobj: TextIO) -> BinaryIO:
    """
    >>> in_buf = io.StringIO("Hello world")
    >>> zipped = gzip_fileobj(in_buf)
    >>> unzipped = gunzip_fileobj(zipped)
    >>> unzipped.getvalue()
    b'Hello world'
    """
    buf = gzip.compress(fileobj.getvalue().encode("utf-8"), compresslevel=2)
    return io.BytesIO(buf)


def convert_fastq_to_fasta(fastq: Path) -> Path:
    r"""
    >>> fastq = Path("/tmp/test.fq")
    >>> _len = fastq.write_text("@seq1\nacgt\n@\n####\n@seq2\ntgac\n@\n####\n")
    >>> fasta = convert_fastq_to_fasta(fastq)
    >>> print(fasta.read_text())
    >seq1
    acgt
    >seq2
    tgac
    <BLANKLINE>
    """
    fasta = fastq.with_suffix(".fa")
    buf = convert_fastq_to_fasta_fileobj(io.StringIO(fastq.read_text()))
    fasta.write_text(buf.getvalue())
    return fasta


def convert_fastq_to_fasta_fileobj(fasta: TextIO) -> TextIO:
    result = io.StringIO()
    for i, line in enumerate(fasta):
        line = line.strip()
        if (i % 4) == 0 or ((i-1) % 4) == 0:
           if line.startswith("@"):
               line = f">{line[1:]}"
           result.write(line + "\n")
    return result


def convert_fasta_file_to_fastq(fasta: Path) -> Path:
    fastq = fasta.with_suffix(".fq")
    buf = convert_fasta_to_fastq_fileobj(io.StringIO(fasta.read_text()))
    fastq.write_text(buf.getvalue() + "\n")
    return fastq


def convert_fasta_to_fastq_fileobj(fasta: TextIO) -> TextIO:
    r"""

    Main idea is to add in a fake quality line.

    >>> fasta = ">seq1\nacgt\ngtac"
    >>> fastq = convert_fasta_to_fastq_fileobj(io.StringIO(fasta))
    >>> print(fastq.getvalue())
    @seq1
    acgtgtac
    +
    ########

    """
    result = io.StringIO()
    summary: Optional[str] = None
    sequence: List[str] = []

    def output_fastq_entry():
        nonlocal summary, sequence
        result.write(f"@{summary}\n")
        seq_str = "".join(sequence)
        result.write(f"{seq_str}\n")
        result.write("+\n")
        result.write("#" * len(seq_str))
        summary = None
        sequence = []

    for line in fasta:
        line = line.strip()
        if line.startswith(">"):
            if summary is not None:
                output_fastq_entry()
            summary = line[1:]
            sequence = []
        else:
            sequence.append(line)
    if summary is not None:
        output_fastq_entry()
    return result


def deconstruct_s3_uri(s3_url: str) -> Tuple[str, str]:
    """
    Extract the bucket name and prefix
    
    >>> deconstruct_s3_uri("s3://foo/bar.txt")
    ('foo', 'bar.txt')

    """
    patn = re.compile(r"^s3://([^/]+)/(.*)$")
    m = patn.match(s3_url)
    if m:
        return m.group(1), m.group(2)
    else:
        return None, None


def load_model_and_tokenizer(model_id, revision: str = "main",
                             download_always: bool = True
                            ) -> tuple:
    print(f"load_model_and_tokenizer {model_id} {download_always}")
    kwargs = {
        "trust_remote_code": True,
        "revision": revision,
    }
    config = AutoConfig.from_pretrained(model_id, download_always=download_always,
                                        **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config,
                                                 download_always=download_always,
                                                 **kwargs)
    print(f"model {model}")
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model moved to GPU.")
    else:
        print("Using CPU for inference.")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id,
                                              download_always=download_always,
                                              **kwargs)
    print(f"tokenizer: {tokenizer} {type(tokenizer)}")
    return model, tokenizer


def join(sep: str, *strings: List[str]) -> str:
    """
    This is like `sep.join(strings)` except we make sure
    that there are no consecutive separators.
    
    >>> join('/', 'a/', '/b')
    'a/b'
    """
    return sep.join(s.strip(sep) for s in strings)


def RMSerror(arr1: torch.tensor, arr2: torch.tensor) -> float:
    """
    Compute the RMS error of two 1-D tensors.
    
    >>> RMSerror(torch.tensor([1.0, 2.0, 3.0]),
    ...          torch.tensor([1.0, 2.0, 3.0]))
    0.0

    >>> RMSerror(torch.tensor([1.0, 2.0, 3.0]),
    ...          torch.tensor([1.1, 2.1, 3.1]))
    0.09999994188547134

    >>> RMSerror(torch.tensor([1.0, 2.0, 3.0]),
    ...          torch.tensor([2.0, 3.0, 4.0]))
    1.0
    """
    assert arr1.shape == arr2.shape
    assert len(arr1.shape) ==1, "must be 1-dimensional"
    return torch.sqrt(torch.mean((arr1-arr2)**2)).item()


class WeightedAvg:
    """
    >>> wa = WeightedAvg()
    >>> wa.add(1, 10)
    >>> wa.add(1, 20)
    >>> wa.value()
    15.0

    >>> wa = WeightedAvg()
    >>> wa.add(1, 10)
    >>> wa.add(1, 10)
    >>> wa.add(0.1, 20)
    >>> wa.value()
    10.476190476190476
    """
    
    def __init__(self):
        self._sum_weights = 0
        self._sum_values = 0
    
    def add(self, weight: float, value: float):
        self._sum_weights += weight
        self._sum_values += value*weight

    def value(self):
        return self._sum_values / self._sum_weights


class Histogram:
    def __init__(self, n_bins: int, low: float, high: float):
        self._n_bins = n_bins
        self._low = low
        self._high = high
        self._stats = defaultdict(int)
    
    def add(self, value: float):
        if value < self._low:
            print(f"WARNING {value} is too low")
            value = self._low
        if value > self._high:
            print(f"WARNING {value} is too high")
            value = self._high
        bin = math.floor(((value - self._low)/(self._high - self._low)) * self._n_bins)
        if bin > self._n_bins - 1:
            bin = self._n_bins - 1
        self._stats[bin] += 1
    
    def batch_add(self, values: List[float]):
        for value in values:
            self.add(value)
        
    def __str__(self):
        return "[" + " ".join(str(self._stats.get(bin, 0)) for bin in range(self._n_bins))\
                   + "]"
