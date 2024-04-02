"""
Commonly-used utilities are here, not in a notebook, to
make them easier to test.

To run unit-tests: python3 -m doctest utilities.py
(You may have to first 'pip3 install doctest'.)
"""

import io
from typing import BinaryIO, TextIO, List, Callable, Optional
from pathlib import Path
import gzip


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


def convert_directory(dir: Path,
                      convertor: Callable[[Path], Path],
                      suffix: str = ".fq") -> List[Path]:
    result = []
    for orig_file in dir.rglob(f"*{suffix}"):
        new_file = convertor(orig_file)
        print(f"{str(orig_file)} -> {str(new_file)}")
        result.append(new_file)
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


def convert_fastq_to_fasta(fastq_path: str, fasta_path: str):
    with open(fastq_path, "r") as in_f:
        with open(fasta_path, "w") as out_f:
           for i, line in enumerate(in_f):
               if (i % 4) == 0 or ((i-1) % 4) == 0:
                   print(f"outputing line {i}")
                   print(line, file=out_f)


def convert_fasta_file_to_fastq(fasta: Path) -> Path:
    fastq = fasta.with_suffix(".fq")
    buf = convert_fasta_file_obj_to_fastq(io.StringIO(fasta.read_text()))
    fastq.write_text(buf.getvalue() + "\n")
    return fastq


def convert_fasta_file_obj_to_fastq(fasta: TextIO) -> TextIO:
    r"""

    Main idea is to add in a fake quality line.

    >>> fasta = ">seq1\nacgt\ngtac"
    >>> fastq = convert_fasta_file_obj_to_fastq(io.StringIO(fasta))
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
