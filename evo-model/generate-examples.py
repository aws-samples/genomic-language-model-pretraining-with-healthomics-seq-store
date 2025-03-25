import argparse
import json
import sys
import re
from pathlib import Path
from random import sample, choice
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from itertools import zip_longest
import gzip
import shutil

sys.path.append('../')
from utilities import deconstruct_s3_uri, join

import boto3
import omics
import pysam
from omics.transfer.manager import TransferManager
from omics.transfer.config import TransferConfig
from pysam import VariantFile
import requests

# num_threads = 30
s3 = boto3.client("s3", region_name="us-west-2")


def download_s3(in_s3_path, bam_path, bam_tmp_path, chr_filt, num_threads):
    # samtools view -h -b -o HG00553_chr21.bam s3://... chr21
    print(f'Downloading BAM from S3 {in_s3_path} {chr_filt} -> {bam_tmp_path}')

    if not Path(bam_tmp_path).exists():
        pysam.view('-b', '-h', '-@', str(num_threads), '-o', bam_tmp_path,
                   in_s3_path, chr_filt, catch_stdout=False)
    else:
        print("skipping download")

    # samtools sort -O BAM -o HG00096_sort_chr21.bam HG00096_chr21.bam 

    print(f"Sorting S3 -> {bam_path}")
    if not Path(bam_path).exists():
        pysam.sort('-O', 'BAM', '-@', str(num_threads), '-o', bam_path, bam_tmp_path)
    else:
        print("Skipping sort")
    bai_path = bam_path + ".bai"
    print(f"Creating index -> {bai_path}")
    if not Path(bai_path).exists():
        pysam.index('-b', '-@', str(num_threads), bam_path)
    else:
        print("Skipping index")


def download_VCF(bucket: str, key: str, local_file_name: str):
    if not Path(local_file_name).exists():
        s3.download_file(bucket, key, local_file_name)
    print(f"Downloaded VCF to {local_file_name}")
    key = key + ".tbi"
    local_file_name = local_file_name + ".tbi"
    if not Path(local_file_name).exists():
        s3.download_file(bucket, key, local_file_name)
    print(f"Downloaded VCF to {local_file_name}")


def get_reads_overlapping_window(variant_rec, window_size: int, bam_path: str):
    print(f"get_reads_overlapping_window {variant_rec} {window_size} {bam_path}")
    print(f"variant.info: {dict(variant_rec.info)}")
    print(f"variant.id: {variant_rec.id}")
    central_pos = variant_rec.pos - 1 # convert 1-based to 0-based
    leftmost_pos = central_pos - window_size
    rightmost_pos = central_pos + window_size # inclusive
    # for samtools notation, coords are 1-based and end is inclusive
    # (see https://pysam.readthedocs.io/en/stable/glossary.html#term-region):
    region = f"{chr_filt}:{variant_rec.pos - window_size + 1}-{variant_rec.pos + window_size + 1}"
    # print(f"region {region}")
    # results = pysam.view(bam_path, region)
    f = pysam.AlignmentFile(bam_path)
    # print(f"f: {f}")
    k_mers = []
    for read in f.fetch(region=region):
        seq = read.get_forward_sequence()
        # print(f"seq: (len {len(seq):,}) {seq}")
        # print(f"reference_start {read.reference_start}") # 0-based
        # print(f"reference_end: {read.reference_end}") # 0-based, points to one past the end
        if read.reference_start <= leftmost_pos and rightmost_pos <= read.reference_end - 1:
            k_mer = seq[leftmost_pos - read.reference_start:rightmost_pos - read.reference_start+1]
            if len(k_mer) == window_size*2 + 1:
                # if read is shorter than reference sequence we may not come in here
                k_mers.append(k_mer)
    print(f"Found {len(k_mers):,} k-mers, picking one")
    k_mer = choice(k_mers)
    return {
        "k_mer": k_mer,
        "central_position": central_pos # 0-based
    }


def maybe_rm_chr_prefix(chrom: str):
    """
    >>> maybe_rm_chr_prefix("21")
    '21'
    
    >>> maybe_rm_chr_prefix("chr21")
    '21'
    """
    return chrom[3:] if chrom.startswith("chr") else chrom


def maybe_add_chr_prefix(chrom: str):
    """
    >>> maybe_add_chr_prefix("21")
    'chr21'
    
    >>> maybe_add_chr_prefix("chr21")
    'chr21'
    """
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def show_variants_in_range(vcf_f, centroid: int, delta: int = 50, remove_chr: bool = False):
    print(f"+++Indiv variants in [{centroid-delta}..{centroid+delta}]:")
    for var in vcf_f.fetch(region=maybe_rm_chr_prefix(chr_filt) if remove_chr else chr_filt):
        if centroid-delta <= var.pos and var.pos < centroid+delta:
            print(var)
    print("+++ end")


def download_file(URL: str) -> str:
    local_file_name = URL.split("/")[-1]
    if not Path(local_file_name).exists():
        print(f"Downloading {URL}...")
        resp = requests.get(URL)
        with open(local_file_name, "wb") as f:
            f.write(resp.content)
        print(f"Created {local_file_name}")
    return local_file_name
    

def download_reference_clinvar_dbSNP_ids(
    clinvar_VCF_URL: str = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar_20250323.vcf.gz",
    instrument_dbSNP_ids: Set[str] = {}  # for debugging
    ) -> Tuple[dict, VariantFile]:

    """
    We first download the VCF file (and its associated TBI file). We then extract
    all variants that have an associated dbSNP id. We then compute and return
    a mapping from locus (<chrom>:<pos>) to a dict containing info about the variant
    at that position (for now, dbSNP id and effect).
    
    We return a second value, the VariantFile object as a pass-thru.
    
    [Note that they keep changing the URL, so the download may break. If it does,
    go to https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/ and get the latest version
    and update `clinvar_VCF_URL`.]
    
    @return dict(locus => info), VCF file
    """
    print(f"download_reference_clinvar_dbSNP_ids {clinvar_VCF_URL}")
    clinvar_file_name = download_file(clinvar_VCF_URL)
    if clinvar_file_name.endswith(".gz"):
        clinvar_file_name_no_gz = clinvar_file_name[:-3]
        if not Path(clinvar_file_name_no_gz).exists():
            with gzip.open(clinvar_file_name, "rb") as f_in:
                clinvar_file_name = clinvar_file_name
                with open(clinvar_file_name_no_gz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print(f"Decompressed {clinvar_file_name} -> {clinvar_file_name_no_gz}")
    tbi_file_name = download_file(clinvar_VCF_URL+".tbi")

    clinvar_VCF = VariantFile(clinvar_file_name, mode="r")
    clinvar_variants = [var for var in clinvar_VCF.fetch(region=maybe_rm_chr_prefix(chr_filt))]
    clinvar_dbSNP_variants = [
        var for var in clinvar_variants
        if "RS" in var.info
    ]
    for var in clinvar_dbSNP_variants:
        if var.info["RS"][0] in instrument_dbSNP_ids:
            print(f"instrumented: {var}")
    print(f"clinvar: total {len(clinvar_variants):,} variants, {len(clinvar_dbSNP_variants):,} with dbSNP ids")
    region2info = {}
    for var in clinvar_dbSNP_variants:
        # position is 1-based:
        region2info[f"{maybe_add_chr_prefix(var.chrom)}:{var.pos}"] = \
            dict(dbSNPid=var.info["RS"][0],
                 effect=var.info.get('MC', None))
    # print(f"first 20 of region2info: {list(region2info.items())[:20]}")
    return region2info, clinvar_VCF # region is 1-based


def pprint_region(region: str) -> str:
    try:
        chrom, locus = region.split(":")
        return f"{chrom}:{int(locus):,}"
    except:
        print(f"WARNING: failed to parse {region}")
        return region


def generate_examples(
    ref_VCF: VariantFile,
    bam_path: str,
    individual_variant_loci: Set[str], # samtools region string, chrXX
    chr_filt: str,  # chrXX
    num_reads_per_variant: int,
    required_read_len: int
    ):
    """
    Given a set of regions of variants in the individual's VCF,
    `individual_variant_loci`, we compute and return the set of
    examples, where each example corresponds to a variant that is in
    the reference and that has one or more reads.
    """
    print(f"generate_examples {bam_path} {chr_filt}"\
          f" {num_reads_per_variant} {required_read_len}")
    # Path("./ivrs.txt").write_text("\n".join(map(pprint_region, sorted(individual_variant_loci,
    #                                                                   key=lambda ivr: int(ivr.split(":")[1])))))
    ranges = defaultdict(lambda: dict(min=None, max=None))
    for ivr in individual_variant_loci:
        chrom, locus = ivr.split(":")
        locus = int(locus)
        if ranges[chrom]["max"] is None or ranges[chrom]["max"] < locus:
            ranges[chrom]["max"] = locus
        if ranges[chrom]["min"] is None or ranges[chrom]["min"] > locus:
            ranges[chrom]["min"] = locus
    print("Ranges:")
    for chrom, min_max in ranges.items():
        print(f"{chrom}:[{min_max['min']:,}..{min_max['max']:,}]")
    all_effects = set()
    alignment_file = pysam.AlignmentFile(bam_path)
    examples = []
    for ref_variant in ref_VCF.fetch(region=maybe_rm_chr_prefix(chr_filt)):
        ref_locus = f"{maybe_add_chr_prefix(ref_variant.chrom)}:{ref_variant.pos}"
        print(f"ref_locus: {pprint_region(ref_locus)} info {dict(ref_variant.info)}")
        if ref_locus in individual_variant_loci:
            effect = ref_variant.info.get("MC", None) # this is really a Tuple
            if effect:
                for eff in effect:
                    all_effects.add(eff)
        else:
            effect = None
        reads = get_reads_for_locus(alignment_file, ref_variant, num_reads_per_variant,
                                    required_read_len)
        if reads:
            examples.append({
                "chrom": ref_variant.chrom,
                 "pos": ref_variant.pos,
                 "region": ref_locus,
                 "effect": effect,
                 "reads": reads,
                 **SO_term_2_impact_and_score(effect)
                 })
        else:
            print("No reads found")
    print("all effects seen:")
    for eff in all_effects:
        print(f" - {eff}")
    print(f"Created {len(examples):,} examples")
    assert len(examples) > 0, "Check the .bai file: it might be old"
    return examples


def get_reads_for_locus(
    alignment_file: pysam.AlignmentFile,  # internal representation of BAM file
    ref_variant: pysam.VariantRecord,  # a reference Variant
    num_reads_per_variant: int,  # how many to return
    required_read_len: int):
    """
    We truncate or discard reads that are not exactly `required_read_len`
    long. It turns out that we discard less than 1% of reads so it's not
    significant and makes the batch learning easier (no need for padding).
    """
    print(f"get_reads_for_locus {maybe_add_chr_prefix(ref_variant.chrom)}:{int(ref_variant.pos):,}")
    ref_region = f"{maybe_add_chr_prefix(ref_variant.chrom)}:{ref_variant.pos}-{ref_variant.pos}"
    print(f"ref_region {ref_region}")
    raw_reads = list(alignment_file.fetch(region=ref_region))
    print(f"Got {len(raw_reads):,} reads")
    actual_reads = []
    for read in raw_reads:
        if len(read.get_forward_sequence()) < required_read_len:
            pass # discard it
        elif len(read.get_forward_sequence()) == required_read_len:
            actual_reads.append(read)
        else:
            # keep it if the variant is NOT in the discarded part
            if ref_variant.pos - read.reference_start < required_read_len:
                actual_reads.append(read)
    if not actual_reads:
        # If no reads for this variant, discard it
        print("no actual reads")
        return []
    # assert all(read.reference_start <= ref_variant.pos and ref_variant.pos <= read.reference_end
            #   for read in reads)
    if num_reads_per_variant == 1:
        # pick a read in the middle
        read = actual_reads[len(actual_reads)//2]
        seq = read.get_forward_sequence()[:required_read_len]
        return [seq]
    else:
        raise Exception("port me")


high_terms = {
    "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
    "stop_gained", "frameshift_variant", "stop_lost", "start_lost",
    "transcript_amplification", "feature_elongation", "feature_elongation",
    "nonsense" # added by SGH
}
moderate_terms = {
    "inframe_insertion", "inframe_deletion", "missense_variant", 
    "protein_altering_variant",
    "inframe_indel" # added by SGH
}
low_terms = {
    "splice_donor_5th_base_variant", "splice_region_variant", 
    "splice_donor_region_variant", "splice_polypyrimidine_tract_variant",
    "incomplete_terminal_codon_variant", "start_retained_variant",
    "stop_retained_variant", "synonymous_variant"
}
modifier_terms = {
    "coding_sequence_variant", "mature_miRNA_variant", "5_prime_UTR_variant",
    "3_prime_UTR_variant", "non_coding_transcript_exon_variant",
    "intron_variant", "NMD_transcript_variant", "non_coding_transcript_variant",
    "coding_transcript_variant", "upstream_gene_variant", "downstream_gene_variant",
    "TFBS_ablation", "TFBS_amplification", "TF_binding_site_variant", 
    "regulatory_region_ablation", "regulatory_region_amplification",
    "regulatory_region_variant", "intergenic_variant", "sequence_variant",
    "genic_upstream_transcript_variant" # added by SGH
}
none_terms = {
    "no_sequence_alteration"
}

def SO_term_2_impact(so_term: Optional[str]) -> str:
    """
    From.
    https://useast.ensembl.org/info/genome/variation/prediction/predicted_data.html
    
    return: "high", "moderate", "low", "modifier", "none"
    """
    if so_term is None:
        return "none"
    so_term = str(so_term) # ('SO:0001583|missense_variant',) -> "('SO:0001583|missense_variant',)"
    for terms, result in [(high_terms, "high"), (moderate_terms, "moderate"),
                          (low_terms, "low"), (modifier_terms, "modifier"),
                          (none_terms, "none")]:
        for term in terms:
            if term in so_term:
                return result
    print(f"Warning: {so_term} not recognized, defaulting to 'low'")
    return "low"


IMPACT_SCORE_MAPPING = {
    "high":    10,
    "moderate": 5,
    "low":      2,
    "modifier": 1,
    "none":     0
}

def impact_2_impact_score(impact: str) -> float:
    return IMPACT_SCORE_MAPPING[impact]


def SO_term_2_impact_and_score(so_term: Optional[str]) -> dict:
    impact = SO_term_2_impact(so_term)
    impact_score = impact_2_impact_score(impact)
    return dict(impact=impact, impact_score=impact_score)


def generate_bam_stats(bam_path: str, target_len: int = 150):
    print(f"Generating stats for {bam_path}...")
    with pysam.AlignmentFile(bam_path) as alignment_file:
        reads = list(alignment_file.fetch())
        seqs = [r.get_forward_sequence() for r in reads]
        lens = [len(seq) for seq in seqs]
        lens_dict = defaultdict(int)
        for a_len in lens:
            lens_dict[a_len] += 1
        print(f"There are {lens_dict[target_len]} of len {target_len}; "
              f"Exclusing everything else removes {100.0*(len(seqs) - lens_dict[target_len])/len(seqs):.2f}%")
        print(f"There are {len(seqs):,} seqs, min {min(lens)}, max {max(lens)}")
        print(f"Lengths: {dict(lens_dict)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bamOriPath',     type=str, required=True,  dest='bamOriPath')
    parser.add_argument('-d', '--dataFolder',     type=str, required=True,  dest='dataFolder')
    parser.add_argument('-r', '--refFile',        type=str, required=True,  dest='referenceFile')
    parser.add_argument('-l', '--requiredReadLen',type=int, required=False, dest='requiredReadLen',
                        default=150,
                        help="All reads must be this length (shorter ones discarded, "
                             "longer ones truncated) so that we can batch examples on the GPU")
    parser.add_argument('-@', '--numThreads',   type=int, required=False,
                        dest='numThreads', default=30)
    parser.add_argument('-nr','--numReadsPerVariant', type=int, required=False,
                        dest='numReadsPerVariant', default=1)
    parser.add_argument('-N', '--trainTestSize',type=int, required=False,
                        dest='trainTestSize', default=100,
                        help="Number of examples to generate, return all of them if -1")
    parser.add_argument('-s3', '--s3resultsLocation',type=str, required=False,
                        dest='s3resultsLocation', default=None,
                        help="If set, upload results here. Format: 's3://bucket/prefix/'")

    args = parser.parse_args()
    subject = args.bamOriPath.split('/')[-1].split('.')[0]
    chr_filt = 'chr21'
    data_folder = args.dataFolder
    ref_path = args.referenceFile
    num_threads = args.numThreads
    required_read_len = args.requiredReadLen
    num_reads_per_variant = args.numReadsPerVariant
    train_test_size = args.trainTestSize
    s3_results_location = args.s3resultsLocation
    
    # create file names
    bam_tmp = f'{subject}_tmp_{chr_filt}.bam'
    bam = f'{subject}_{chr_filt}.bam'
    cram = f'{subject}_chr21.cram'

    # pull partial file 
    bam_tmp_path = data_folder + bam_tmp
    bam_path = data_folder + bam
    s3_path = args.bamOriPath
    download_s3(s3_path, bam_path, bam_tmp_path, chr_filt, num_threads)

    locus2refInfo, ref_VCF = download_reference_clinvar_dbSNP_ids()

    # locus2refInfo maps "chrom:pos" to an info dict {"dbSNPid": ..., "effect": ...}

    # Download the individual's variants:
    vcf_name = "HG00553.hard-filtered.vcf.gz"
    vcf_path = data_folder + vcf_name
    download_VCF(bucket="1000genomes-dragen-3.7.6",
                 key=f"data/individuals/hg38-graph-based/HG00553/{vcf_name}",
                 local_file_name=vcf_path)
                 
    vcf_f = VariantFile(vcf_path)
    variants = list(vcf_f.fetch(region=chr_filt))
    print(f"we have {len(variants):,} variants in the individual")

    variants_with_dbSNP_id = [
        var for var in variants
        if f"{var.chrom}:{var.pos}" in locus2refInfo
    ]
    print(f"and {len(variants_with_dbSNP_id):,} of those have dbSNP ids")
    print("here are some example dbSNP variants:")
    for region, dbSNPid in list(locus2refInfo.items())[:10]:
        print(f"{dbSNPid} -> {region}")
        
    individual_variant_loci = [f"{var.chrom}:{var.pos}"
                               for var in vcf_f.fetch(region=chr_filt)]
    print(f"Found {len(individual_variant_loci):,} individual variant loci, here is a sample:")
    for x in sample(individual_variant_loci, k=10):
        print(f" - {x}")
    train_test = generate_examples(
        ref_VCF=ref_VCF, bam_path=bam_path,
        individual_variant_loci=individual_variant_loci,
        chr_filt=chr_filt,  # chrXX
        num_reads_per_variant=num_reads_per_variant,
        required_read_len=required_read_len)
    print(f"Generated {len(train_test):,} examples")
    assert len({len(x["reads"][0]) for x in train_test}) == 1
    print("Some examples:")
    for ex in sample(train_test, k=5):
        print(f" - {ex}")
    train_test_w_effect = [ex for ex in train_test if ex["effect"]]
    print(f"{len(train_test_w_effect):,} examples have an effect")
    print("Some examples with effects:")
    for ex in sample(train_test_w_effect, k=5):
        print(f" - {ex}")
    print(f"{(len(train_test_w_effect)/len(train_test))*100:.2f}% of examples have an effect")
    if train_test_size > 0:
        train_test = sample(train_test, k=train_test_size)

    def upload_to_s3(s3_location: str, path: Path):
        print(f"upload_to_s3 {s3_location} {path}")
        bucket, prefix = deconstruct_s3_uri(s3_location)
        while prefix.endswith("/"):
            prefix = prefix[:-1]
        prefix = prefix + "/" + str(path)
        s3.upload_file(path, bucket, prefix)
        print(f"Uploaded to s3://{bucket}/{prefix}")

    path = Path("examples.jsonl")
    path.write_text("\n".join(json.dumps(ex)
                              for ex in train_test) + "\n")
    if s3_results_location:
        upload_to_s3(join("/", s3_results_location, str(train_test_size) if train_test_size > 0 else "all"),
                     path)
