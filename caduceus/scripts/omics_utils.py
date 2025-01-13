import os
import boto3
from pathlib import Path
from collections import defaultdict
from typing import BinaryIO, TextIO
from datasets import Dataset
from typing import Literal
import gzip
import io


def create_fastq_entry(sequence: str, sequence_id: str, quality_score: int = 40) -> str:
    """Create a FASTQ entry for a single sequence"""
    quality_char = chr(quality_score + 33)
    quality_string = quality_char * len(sequence)
    return f"@{sequence_id}\n{sequence}\n+\n{quality_string}\n"


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


def gunzip_fileobj(gzip_fileobj: BinaryIO) -> BinaryIO:
    """
    >>> in_buf = io.BytesIO(b"abcdef")
    >>> zipped_bytes = gzip.compress(b"abcdef")
    >>> gunzip_fileobj(io.BytesIO(zipped_bytes)).getvalue()
    b'abcdef'
    """
    with gzip.open(gzip_fileobj, 'rb') as f_in:
        return io.BytesIO(f_in.read())


def get_readset_for_task_and_split(
    sequence_store_id: str, 
    task: str, 
    split: Literal["train", "test"],
    region_name: str = "us-east-1",
) -> str:
    omics = boto3.client("omics", region_name=region_name)
    results = omics.list_read_sets(
        sequenceStoreId=sequence_store_id,
        filter={
            "subjectId": task,
            "name": f"{split}_combined.fastq.gz",
        },
    )["readSets"]
    if len(results) == 0:
        raise ValueError(f"Readset not found for task: {task} and split: {split}")
    elif len(results) > 1:
        print(f"{len(results)} readsets found with these conditions. Returning the most recent.")
        results = sorted(results, key=lambda x: x["creationTime"], reversed=True)
    return results[0]["id"]


def load_dataset_from_omics(
    sequence_store_id: str, 
    task: str, 
    split: Literal["train", "test"],
    part_number: int = 1, 
    file: str = "SOURCE1",
    region_name: str = "us-east-1",
) -> Dataset:
    omics = boto3.client("omics", region_name=region_name)
    read_set_id = get_readset_for_task_and_split(sequence_store_id, task, split, region_name)
    response = omics.get_read_set(sequenceStoreId=sequence_store_id, id=read_set_id, partNumber=part_number, file=file)
    decoded_readset: str = gunzip_fileobj(response['payload']).read().decode()

    readset_lines = decoded_readset.splitlines()
    records = []
    for i in range(0, len(readset_lines), 4):
        seq_id, seq = readset_lines[i:i+2]
        label = int(seq_id[1:].split("_")[1])  # label_{label_id}_idx_{idx}
        records.append({"seq": seq, "label": label})
    return Dataset.from_list(records)
    