"""Tests for DNA encoding and decoding functions."""

import pytest
import random
from typing import List, Any # Corrected import

# Make sure the package root is in sys.path for imports
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.encoding import (
    dna_to_int, 
    int_to_dna,
    random_dna_sequence,
    find_genes_in_dna,
    decode_gene,
    decode_chromosome,
    scale_value,
    scale_value_int,
    map_value_to_choices,
    START_CODON, 
    STOP_CODONS,
    encode_gene,
    encode_chromosome
)
from nucleotide_strategy_evolution.core.structures import (
    DNASequence,
    Gene,
    Chromosome,
    NUCLEOTIDES
)

# --- Test Codon/Int Conversions ---

@pytest.mark.parametrize("dna, expected_int", [
    ("AAA", 0),
    ("AAC", 1),
    ("AAG", 2),
    ("AAT", 3),
    ("ACA", 4),
    ("AGA", 8),
    ("ATA", 12),
    ("CAA", 16),
    ("GAA", 32),
    ("TAA", 48),
    ("CCC", 21),
    ("GGG", 42),
    ("TTT", 63),
    ("GCT", 39), # Example from PLAN (corrected from 37)
])
def test_dna_to_int(dna: str, expected_int: int):
    assert dna_to_int(dna) == expected_int

@pytest.mark.parametrize("value", ["AA", "AATT", "ABC"])
def test_dna_to_int_invalid_length(value: str):
    with pytest.raises(ValueError):
        dna_to_int(value)

def test_dna_to_int_invalid_nucleotide():
    with pytest.raises(ValueError):
        dna_to_int("AXT")

@pytest.mark.parametrize("value, expected_dna", [
    (0, "AAA"),
    (1, "AAC"),
    (2, "AAG"),
    (3, "AAT"),
    (4, "ACA"),
    (8, "AGA"),
    (12, "ATA"),
    (16, "CAA"),
    (32, "GAA"),
    (48, "TAA"),
    (21, "CCC"),
    (42, "GGG"),
    (63, "TTT"),
    (39, "GCT"), # Corresponds to 39
])
def test_int_to_dna(value: int, expected_dna: str):
    assert int_to_dna(value) == expected_dna

@pytest.mark.parametrize("value", [-1, 64, 100])
def test_int_to_dna_invalid_value(value: int):
    with pytest.raises(ValueError):
        int_to_dna(value)

# --- Test Random DNA Sequence ---
def test_random_dna_sequence():
     """Test the random sequence generation."""
     length = 50
     rand_seq = random_dna_sequence(length)
     assert isinstance(rand_seq, DNASequence)
     assert len(rand_seq) == length
     # Check if all nucleotides are valid
     valid_nucleotides_chars = {'A', 'C', 'G', 'T'}
     for char in rand_seq.sequence:
         assert char in valid_nucleotides_chars

     rand_seq_zero = random_dna_sequence(0)
     assert len(rand_seq_zero) == 0
     assert rand_seq_zero.sequence == ""

     with pytest.raises(ValueError):
         random_dna_sequence(-1)

# --- Test Scaling/Mapping Helpers ---

@pytest.mark.parametrize("codon_val, min_val, max_val, expected", [
    (0, 0.0, 100.0, 0.0),
    (63, 0.0, 100.0, 100.0),
    (31, 0.0, 100.0, pytest.approx(49.2063)), # Corrected approx (31/63) * 100
    (0, -10.0, 10.0, -10.0),
    (63, -10.0, 10.0, 10.0),
    (15, 1.0, 5.0, pytest.approx(1.9523)), # ~1.9523
])
def test_scale_value(codon_val: int, min_val: float, max_val: float, expected: float):
    assert scale_value(codon_val, min_val, max_val) == expected

@pytest.mark.parametrize("codon_val, min_val, max_val, expected", [
    (0, 0, 100, 0),
    (63, 0, 100, 100),
    (31, 0, 100, 49), # round(49.2063)
    (0, -10, 10, -10),
    (63, -10, 10, 10),
    (15, 1, 5, 2), # round(1.9523)
    (45, 10, 20, 17), # round(10 + 10 * (45/63)) = round(17.1428)
])
def test_scale_value_int(codon_val: int, min_val: int, max_val: int, expected: int):
    assert scale_value_int(codon_val, min_val, max_val) == expected

@pytest.mark.parametrize("codon_val, choices, expected", [
    (0, ["A", "B", "C"], "A"),
    (20, ["A", "B", "C"], "A"), # Segment size = 64/3 = 21.33. 20 // 21.33 = 0
    (21, ["A", "B", "C"], "B"),
    (42, ["A", "B", "C"], "B"),
    (43, ["A", "B", "C"], "C"),
    (63, ["A", "B", "C"], "C"),
    (0, [10, 20, 30, 40], 10), # Seg size = 16
    (15, [10, 20, 30, 40], 10),
    (16, [10, 20, 30, 40], 20),
    (31, [10, 20, 30, 40], 20),
    (32, [10, 20, 30, 40], 30),
    (63, [10, 20, 30, 40], 40),
    (30, [True, False], True), # Seg size = 32
    (32, [True, False], False),
])
def test_map_value_to_choices(codon_val: int, choices: List[Any], expected: Any):
    assert map_value_to_choices(codon_val, choices) == expected

def test_map_value_to_choices_empty():
    assert map_value_to_choices(30, []) is None

# --- Tests for Gene Finding --- 
# (More complex, start with basic cases)

def test_find_genes_no_start():
    dna = DNASequence("AAACCCTTTGGGTAATAGTAG")
    assert find_genes_in_dna(dna) == []

def test_find_genes_start_no_stop():
    dna = DNASequence("AAACCCTTTATGCCCGGGTTT")
    assert find_genes_in_dna(dna) == []

def test_find_genes_simple():
    #             START      STOP
    dna = DNASequence("XXATGAAACCCGGGXXXTAGYYY")
    expected_start = 2
    expected_end = 20 # ATG(3) + AAA(3) + CCC(3) + GGG(3) + XXX(3) + TAG(3) = 18. End index is start + len = 2 + 18 = 20
    expected_gene = "ATGAAACCCGGGXXXTAG"
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    start, end, raw, promo = genes[0]
    assert start == expected_start
    assert end == expected_end
    assert raw == expected_gene
    assert promo is None

def test_find_genes_multiple():
    #             START GENE 1 STOP    START GENE 2 STOP
    dna = DNASequence("XXATGAAACCCGGGTAGYYYYATGTTTAAATAAZZZ")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 2
    # Gene 1
    assert genes[0][0] == 2  # Start index
    assert genes[0][1] == 17 # End index
    assert genes[0][2] == "ATGAAACCCGGGTAG" # Raw DNA
    assert genes[0][3] is None # Promoter
    # Gene 2
    assert genes[1][0] == 21 # Start index
    assert genes[1][1] == 33 # End index
    assert genes[1][2] == "ATGTTTAAATAA" # Raw DNA
    assert genes[1][3] is None # Promoter

def test_find_genes_overlapping_starts():
    # Overlapping start codons, only the first valid gene should be found
    #             START(START) GENE 1 STOP
    dna = DNASequence("XXATGATGAAACCCTAGYYY")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    assert genes[0][0] == 2  # Start index of the outer ATG
    assert genes[0][1] == 17 # End index
    assert genes[0][2] == "ATGATGAAACCCTAG"

def test_find_genes_with_promoter():
    promoter_seq = "TATAAT"
    #             PROMOTER  START GENE 1 STOP
    dna = DNASequence(f"XX{promoter_seq}YYYATGAAACCCGGGTAGZZZ")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    assert genes[0][0] == 11 # Start index of ATG
    assert genes[0][1] == 26 # End index
    assert genes[0][2] == "ATGAAACCCGGGTAG"
    assert genes[0][3] == promoter_seq # Promoter

def test_find_genes_promoter_too_far():
    promoter_seq = "TATAAT"
    #             PROMOTER             START GENE 1 STOP (Promoter > 15bp away)
    dna = DNASequence(f"XX{promoter_seq}YYYYYYYYYYYYYYYATGAAACCCGGGTAGZZZ")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    assert genes[0][0] == 26 # Start index of ATG
    assert genes[0][1] == 41 # End index
    assert genes[0][2] == "ATGAAACCCGGGTAG"
    assert genes[0][3] is None # Promoter should not be found

# --- Tests for Gene Decoding --- 
# (More complex, start with basic type/param identification)

def test_decode_gene_basic_types():
    # Entry rule type (codon 5)
    gene_entry = decode_gene(START_CODON + int_to_dna(5) + "AAAAAA" + STOP_CODONS[0])
    assert gene_entry is not None
    assert gene_entry.gene_type == "entry_rule"
    assert gene_entry.raw_dna.sequence == START_CODON + int_to_dna(5) + "AAAAAA" + STOP_CODONS[0]
    assert gene_entry.expression_level == 1.0 # Default

    # Exit rule type (codon 20)
    gene_exit = decode_gene(START_CODON + int_to_dna(20) + "CCCCCC" + STOP_CODONS[1], promoter="CATCAT")
    assert gene_exit is not None
    assert gene_exit.gene_type == "exit_rule"
    assert gene_exit.expression_level == 0.5 # Weak promoter
    assert gene_exit.promoter_sequence == "CATCAT"

    # Risk management type (codon 40)
    gene_risk = decode_gene(START_CODON + int_to_dna(40) + "GGGGGG" + STOP_CODONS[2])
    assert gene_risk is not None
    assert gene_risk.gene_type == "risk_management"

    # Indicator type (codon 55)
    gene_indic = decode_gene(START_CODON + int_to_dna(55) + "TTTTTT" + STOP_CODONS[0])
    assert gene_indic is not None
    assert gene_indic.gene_type == "indicator"

    # Time filter type (codon 58)
    gene_time = decode_gene(START_CODON + int_to_dna(58) + "ACGTAC" + STOP_CODONS[1])
    assert gene_time is not None
    assert gene_time.gene_type == "time_filter"

    # Order management type (codon 61)
    gene_order = decode_gene(START_CODON + int_to_dna(61) + "TGCACT" + STOP_CODONS[2])
    assert gene_order is not None
    assert gene_order.gene_type == "order_management"

    # Regime filter type (codon 62)
    gene_regime = decode_gene(START_CODON + int_to_dna(62) + "GGTACC" + STOP_CODONS[0])
    assert gene_regime is not None
    assert gene_regime.gene_type == "regime_filter"

    # Meta type (codon 63)
    gene_meta = decode_gene(START_CODON + int_to_dna(63) + "CCAAGG" + STOP_CODONS[1])
    assert gene_meta is not None
    assert gene_meta.gene_type == "meta"

def test_decode_gene_invalid():
    # Too short
    assert decode_gene(START_CODON + STOP_CODONS[0]) is None
    # No start codon
    assert decode_gene("AAACCCGGGTTT") is None
    # Invalid type codon
    gene_invalid_type = decode_gene(START_CODON + "XYZ" + "AAACCC" + STOP_CODONS[0])
    # It decodes, but type is 'unknown'
    assert gene_invalid_type is not None
    assert gene_invalid_type.gene_type == "unknown"

def test_decode_gene_indicator_params():
    # Indicator Gene: Type=48, Name=SMA(codon 0), Period=14(codon 15), Source=close(codon 5)
    rsi_dna_str = START_CODON + int_to_dna(48) + int_to_dna(0) + int_to_dna(15) + int_to_dna(5) + STOP_CODONS[0]
    gene_rsi = decode_gene(rsi_dna_str)
    assert gene_rsi is not None
    assert gene_rsi.gene_type == "indicator"
    assert gene_rsi.parameters.get('indicator_name') == "SMA" # First choice
    assert gene_rsi.parameters.get('period') == 14 # scale_value_int(15, 5, 100)
    assert gene_rsi.parameters.get('source') == "close" # First choice

    # BBANDS Gene: Type=50, Name=BBANDS(codon 50), Period=21(codon 21), StdDev=2.0(codon 31)
    bb_dna_str = START_CODON + int_to_dna(50) + int_to_dna(50) + int_to_dna(21) + int_to_dna(31) + STOP_CODONS[1]
    gene_bb = decode_gene(bb_dna_str)
    assert gene_bb is not None
    assert gene_bb.gene_type == "indicator"
    assert gene_bb.parameters.get('indicator_name') == "BBANDS"
    assert gene_bb.parameters.get('period') == 21 # scale_value_int(21, 5, 100)
    assert gene_bb.parameters.get('std_dev') == pytest.approx(2.0) # scale_value(31, 1.0, 4.0)

# TODO: Add more detailed tests for parameter decoding of each gene type (Risk, Rules, Time, Order, Regime, Meta)

# Helper function to create gene DNA strings for tests
def make_gene_str(type_codon_val: int, param_codons_vals: List[int], stop_codon: str = STOP_CODONS[0]) -> str:
    s = START_CODON + int_to_dna(type_codon_val)
    for val in param_codons_vals:
        s += int_to_dna(val)
    s += stop_codon
    return s

def test_decode_gene_risk_params():
    # Risk Gene: Type=35, SL Mode=ticks(0), SL Val=63(100), TP Mode=atr(20), TP Val=40(scaled ~3.33)
    dna_str = make_gene_str(type_codon_val=35, param_codons_vals=[0, 63, 20, 40])
    gene = decode_gene(dna_str)
    assert gene is not None
    assert gene.gene_type == "risk_management"
    assert gene.parameters.get('stop_loss_mode') == "ticks"
    assert gene.parameters.get('stop_loss_value') == 100 # scale_value_int(63, 5, 100)
    assert gene.parameters.get('take_profit_mode') == "atr" # choices[1] for val 20
    assert gene.parameters.get('take_profit_value') == pytest.approx(scale_value(40, 1.0, 10.0)) # ~3.333

    # Risk Gene: SL Mode=none(60), TP Mode=none(60)
    dna_str_none = make_gene_str(type_codon_val=35, param_codons_vals=[60, 10, 60, 20])
    gene_none = decode_gene(dna_str_none)
    assert gene_none is not None
    assert gene_none.parameters.get('stop_loss_mode') == "none"
    assert 'stop_loss_value' not in gene_none.parameters # Value ignored if mode is none
    assert gene_none.parameters.get('take_profit_mode') == "none"
    assert 'take_profit_value' not in gene_none.parameters

    # Risk Gene: SL % (45), TP RR (50)
    dna_str_pct_rr = make_gene_str(type_codon_val=40, param_codons_vals=[45, 31, 50, 63])
    gene_pct_rr = decode_gene(dna_str_pct_rr)
    assert gene_pct_rr.gene_type == "risk_management"
    assert gene_pct_rr.parameters.get('stop_loss_mode') == "percentage"
    assert gene_pct_rr.parameters.get('stop_loss_value') == pytest.approx(scale_value(31, 0.1, 2.0)) # ~1.08%
    assert gene_pct_rr.parameters.get('take_profit_mode') == "rr_ratio"
    assert gene_pct_rr.parameters.get('take_profit_value') == pytest.approx(5.0) # scale_value(63, 1.0, 5.0)

def test_decode_gene_rule_params():
    # Entry Rule: Type=10, Logic=OR(35), NumCond=1(5->1), Cond1(SMA(0) P=10(10) Op=<(20) ThType=Const(5) ThVal=50->20)
    dna_str = make_gene_str(type_codon_val=10, param_codons_vals=[35, 5, 0, 10, 20, 5, 50])
    gene = decode_gene(dna_str)
    assert gene is not None
    assert gene.gene_type == "entry_rule"
    assert gene.parameters.get('logic') == "OR"
    assert gene.parameters.get('num_conditions_attempted') == 1
    assert len(gene.parameters.get('conditions', [])) == 1
    cond1 = gene.parameters['conditions'][0]
    assert cond1.get('indicator_name') == "SMA"
    assert cond1.get('period') == 10 # scale_value_int(10, 5, 100)
    assert cond1.get('operator') == "<"
    assert cond1.get('threshold_type') == "constant"
    assert cond1.get('threshold_value') == 20 # scale_value_int(50, -50, 50)

    # Exit Rule with Indicator Threshold: Type=25, Logic=AND(10), NumCond=1(5->1), 
    # Cond1(RSI(30) P=14(15) >(5) ThType=Indic(40) ThIndic=SMA(0) ThPer=50(40) ThSrc=close(0))
    dna_str_ind = make_gene_str(type_codon_val=25, param_codons_vals=[10, 5, 30, 15, 5, 40, 0, 40, 0])
    gene_ind = decode_gene(dna_str_ind)
    assert gene_ind is not None
    assert gene_ind.gene_type == "exit_rule"
    assert gene_ind.parameters.get('logic') == "AND"
    assert gene_ind.parameters.get('num_conditions_attempted') == 1
    assert len(gene_ind.parameters.get('conditions', [])) == 1
    cond1_ind = gene_ind.parameters['conditions'][0]
    assert cond1_ind.get('indicator_name') == "RSI"
    assert cond1_ind.get('period') == 14
    assert cond1_ind.get('operator') == ">"
    assert cond1_ind.get('threshold_type') == "indicator"
    assert cond1_ind.get('threshold_indicator_name') == "SMA"
    assert cond1_ind.get('threshold_period') == 50
    assert cond1_ind.get('threshold_source') == "close"
    assert 'threshold_value' not in cond1_ind

    # Rule with multiple conditions (NumCond=30->2)
    dna_str_multi = make_gene_str(type_codon_val=12, param_codons_vals=[
        10, 30, # Logic=AND, NumCond=2
        0, 10, 20, 5, 50, # Cond1 (SMA < Const 20)
        30, 15, 5, 5, 10 # Cond2 (RSI > Const -30)
    ])
    gene_multi = decode_gene(dna_str_multi)
    assert gene_multi.parameters.get('num_conditions_attempted') == 2
    assert len(gene_multi.parameters.get('conditions', [])) == 2
    assert gene_multi.parameters['conditions'][0]['indicator_name'] == "SMA"
    assert gene_multi.parameters['conditions'][1]['indicator_name'] == "RSI"
    assert gene_multi.parameters['conditions'][1]['threshold_value'] == -30 # scale(-50, 50)

def test_decode_gene_time_filter_params():
    # Time Filter: Type=56, Start=8(20), End=16(40), Days=M/W/F(20)
    dna_str = make_gene_str(type_codon_val=56, param_codons_vals=[20, 40, 20])
    gene = decode_gene(dna_str)
    assert gene is not None
    assert gene.gene_type == "time_filter"
    assert gene.parameters.get('start_hour') == 8 # scale_value_int(20, 0, 23)
    assert gene.parameters.get('end_hour') == 16 # scale_value_int(40, 0, 23)
    assert gene.parameters.get('active_days') == ["Mon", "Wed", "Fri"] # map_value_to_choices(20, ...)

def test_decode_gene_order_management_params():
    # Order Management: Type=60, Order=Limit(30), TIF=DAY(20), Offset=10(31)
    dna_str = make_gene_str(type_codon_val=60, param_codons_vals=[30, 20, 31])
    gene = decode_gene(dna_str)
    assert gene is not None
    assert gene.gene_type == "order_management"
    assert gene.parameters.get('order_type') == "Limit"
    assert gene.parameters.get('time_in_force') == "DAY"
    assert gene.parameters.get('offset_ticks') == 10 # scale_value_int(31, 0, 20)

    # Order Management: Type=60, Order=Market(5), TIF=GTC(5), Offset=ignored (scaled to 0)
    dna_str_mkt = make_gene_str(type_codon_val=60, param_codons_vals=[5, 5, 50])
    gene_mkt = decode_gene(dna_str_mkt)
    assert gene_mkt.parameters.get('order_type') == "Market"
    assert gene_mkt.parameters.get('time_in_force') == "GTC"
    assert gene_mkt.parameters.get('offset_ticks') == 0 # Offset is 0 for Market orders

def test_decode_gene_regime_filter_params():
    # Regime Filter: Type=62, Indic=VIX(5), Period=ignored(10), Op=>(40), Thresh=25(15), Action=reduce_size(30)
    dna_str = make_gene_str(type_codon_val=62, param_codons_vals=[5, 10, 40, 15, 30])
    gene = decode_gene(dna_str)
    assert gene is not None
    assert gene.gene_type == "regime_filter"
    assert gene.parameters.get('indicator_name') == "VIX"
    assert gene.parameters.get('period') == 10 # scale_value_int(10, 5, 100)
    assert gene.parameters.get('operator') == ">"
    assert gene.parameters.get('threshold') == 25 # scale_value_int(15, 0, 100)
    assert gene.parameters.get('action') == "reduce_size"

def test_decode_gene_meta_params():
    # Meta Gene: Type=63, Param=max_trades(10), Value=5(15)
    dna_str = make_gene_str(type_codon_val=63, param_codons_vals=[10, 15])
    gene = decode_gene(dna_str)
    assert gene is not None
    assert gene.gene_type == "meta"
    assert gene.parameters.get('meta_param_name') == "max_trades_per_day"
    assert gene.parameters.get('meta_param_value') == 5 # scale_value_int(15, 1, 20)

    # Meta Gene: Type=63, Param=max_concurrent(40), Value=3(35)
    dna_str_2 = make_gene_str(type_codon_val=63, param_codons_vals=[40, 35])
    gene2 = decode_gene(dna_str_2)
    assert gene2.parameters.get('meta_param_name') == "max_concurrent_positions"
    assert gene2.parameters.get('meta_param_value') == 3 # scale_value_int(35, 1, 5)

# --- Tests for Chromosome Decoding ---

def test_decode_chromosome():
    promoter_seq = "TATAAT"
    gene1_dna_str = START_CODON + int_to_dna(5) + "AAAAAA" + STOP_CODONS[0]
    gene2_dna_str = START_CODON + int_to_dna(40) + "GGGGGG" + STOP_CODONS[2]

    # Chromosome with two genes and a promoter before the first one
    full_dna_str = f"XXX{promoter_seq}YYY{gene1_dna_str}ZZZ{gene2_dna_str}WWW"
    full_dna = DNASequence(full_dna_str)

    chromosome = decode_chromosome(full_dna)
    assert len(chromosome.genes) == 2
    assert chromosome.raw_dna == full_dna

    # Check Gene 1
    assert chromosome.genes[0].gene_type == "entry_rule"
    assert chromosome.genes[0].raw_dna.sequence == gene1_dna_str
    assert chromosome.genes[0].promoter_sequence == promoter_seq
    assert chromosome.genes[0].expression_level == 1.5 # Strong promoter

    # Check Gene 2
    assert chromosome.genes[1].gene_type == "risk_management"
    assert chromosome.genes[1].raw_dna.sequence == gene2_dna_str
    assert chromosome.genes[1].promoter_sequence is None
    assert chromosome.genes[1].expression_level == 1.0 # Default

def test_decode_chromosome_empty():
    chromosome = decode_chromosome(DNASequence("ACGTACGTACGT"))
    assert len(chromosome.genes) == 0
    assert chromosome.raw_dna.sequence == "ACGTACGTACGT"

# --- Tests for Encoding (Basic) ---

def test_encode_gene_basic():
    # Currently just returns raw_dna if present
    raw = DNASequence(START_CODON + "AAACCC" + STOP_CODONS[0])
    gene = Gene(gene_type="entry", raw_dna=raw)
    assert encode_gene(gene) == raw

    gene_no_raw = Gene(gene_type="exit")
    assert encode_gene(gene_no_raw).sequence == "" # Placeholder returns empty

def test_encode_chromosome_basic():
    # Currently concatenates raw_dna from genes
    raw1 = DNASequence(START_CODON + "AAA" + STOP_CODONS[0])
    raw2 = DNASequence(START_CODON + "CCC" + STOP_CODONS[1])
    gene1 = Gene(gene_type="entry", raw_dna=raw1)
    gene2 = Gene(gene_type="exit", raw_dna=raw2)
    chromosome = Chromosome(genes=[gene1, gene2], raw_dna=DNASequence("IGNORED_FOR_NOW"))

    expected_dna = DNASequence(raw1.sequence + raw2.sequence)
    assert encode_chromosome(chromosome) == expected_dna

    empty_chromosome = Chromosome()
    assert encode_chromosome(empty_chromosome).sequence == ""

# --- Tests for Encoding (Phase 2+) ---
# TODO: Add tests for encode_gene and encode_chromosome (Phase 2+) 