"""Mutation operators for DNA sequences."""

import random
from typing import Dict, Any, List, Tuple, Optional

from nucleotide_strategy_evolution.core.structures import DNASequence, Nucleotide
from nucleotide_strategy_evolution.encoding import NUCLEOTIDES, find_genes_in_dna, int_to_dna

# Define constant for codon length
CODON_LENGTH = 3

# --- Mutation Operator Implementations ---

def point_mutation(dna: DNASequence, mutation_rate: float) -> DNASequence:
    """Performs point mutations on a DNA sequence.

    Each nucleotide in the sequence has a `mutation_rate` chance of being
    replaced by a different random nucleotide.

    Args:
        dna: The DNA sequence to mutate.
        mutation_rate: The probability (0.0 to 1.0) of mutating each nucleotide.

    Returns:
        A new DNASequence object with potential mutations.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("Mutation rate must be between 0.0 and 1.0")

    sequence_list = list(dna.sequence)
    mutated = False
    for i in range(len(sequence_list)):
        if random.random() < mutation_rate:
            original_nucleotide = sequence_list[i]
            # Choose a *different* random nucleotide
            possible_mutations = [n for n in NUCLEOTIDES if n != original_nucleotide]
            if possible_mutations: # Ensure there are other choices
                 sequence_list[i] = random.choice(possible_mutations)
                 mutated = True
            # else: If only one type of nucleotide exists, can't mutate to something different
    
    if mutated:
        return DNASequence(sequence="".join(sequence_list))
    else:
        # Return the original object if no mutations occurred
        # Could also return a copy: return DNASequence(sequence=dna.sequence)
        return dna 

def insertion_mutation(dna: DNASequence, insertion_rate: float) -> DNASequence:
    """Performs single-nucleotide insertion mutation.

    Iterates through the sequence and has `insertion_rate` chance *per position*
    to insert a random nucleotide *before* that position.
    Note: This increases the sequence length.

    Args:
        dna: The DNA sequence to mutate.
        insertion_rate: Probability of inserting a nucleotide before each position.

    Returns:
        A new DNASequence, potentially longer.
    """
    if not 0.0 <= insertion_rate <= 1.0:
        raise ValueError("Insertion rate must be between 0.0 and 1.0")

    sequence_list = list(dna.sequence)
    inserted = False
    # Iterate backwards to avoid index issues after insertion
    for i in range(len(sequence_list) - 1, -1, -1):
        if random.random() < insertion_rate:
            new_nucleotide = random.choice(NUCLEOTIDES)
            sequence_list.insert(i, new_nucleotide)
            inserted = True
            
    # Chance to insert at the very beginning
    if random.random() < insertion_rate:
         new_nucleotide = random.choice(NUCLEOTIDES)
         sequence_list.insert(0, new_nucleotide)
         inserted = True

    if inserted:
        return DNASequence(sequence="".join(sequence_list))
    else:
        return dna

def deletion_mutation(dna: DNASequence, deletion_rate: float) -> DNASequence:
    """Performs single-nucleotide deletion mutation.
    
    Each nucleotide has a `deletion_rate` chance of being deleted.
    Ensures the sequence doesn't become empty.
    Note: This decreases the sequence length.

    Args:
        dna: The DNA sequence to mutate.
        deletion_rate: Probability of deleting each nucleotide.

    Returns:
        A new DNASequence, potentially shorter.
    """
    if not 0.0 <= deletion_rate <= 1.0:
        raise ValueError("Deletion rate must be between 0.0 and 1.0")

    sequence_list = list(dna.sequence)
    original_length = len(sequence_list)
    # Use list comprehension for efficient filtering based on deletion chance
    new_sequence_list = [nuc for nuc in sequence_list if random.random() >= deletion_rate]
    
    # Ensure sequence doesn't become empty
    if not new_sequence_list and original_length > 0:
        # If everything got deleted, restore one random nucleotide from original
        new_sequence_list = [random.choice(sequence_list)]
        
    deleted = len(new_sequence_list) < original_length
    
    if deleted:
        return DNASequence(sequence="".join(new_sequence_list))
    else:
        return dna

def gene_duplication_mutation(dna: DNASequence, duplication_rate: float) -> DNASequence:
    """Performs gene duplication mutation.

    Finds existing genes (using start/stop codons) and has a chance 
    (`duplication_rate` per gene) to duplicate a gene and insert the copy 
    immediately after the original.
    Note: Increases sequence length.

    Args:
        dna: The DNA sequence to mutate.
        duplication_rate: Probability of duplicating each found gene.

    Returns:
        A new DNASequence, potentially longer.
    """
    if not 0.0 <= duplication_rate <= 1.0:
        raise ValueError("Duplication rate must be between 0.0 and 1.0")

    # Find potential genes first
    # Returns list of (start, end, raw_dna, promoter) tuples
    found_genes = find_genes_in_dna(dna) 
    
    if not found_genes:
        return dna # No genes to duplicate

    # Sort genes by start index to process insertions correctly
    found_genes.sort(key=lambda x: x[0])

    new_sequence_parts = []
    last_end = 0
    duplicated = False

    for gene_start, gene_end, raw_gene_dna, _ in found_genes:
        # Add the non-gene segment before this gene
        new_sequence_parts.append(dna.sequence[last_end:gene_start])
        # Add the original gene
        new_sequence_parts.append(raw_gene_dna)
        
        # Check for duplication
        if random.random() < duplication_rate:
            # Duplicate the gene immediately after
            new_sequence_parts.append(raw_gene_dna) 
            duplicated = True
            # print(f"Duplicating gene: {raw_gene_dna}") # Debugging
            
        last_end = gene_end

    # Add any remaining part of the sequence after the last gene
    new_sequence_parts.append(dna.sequence[last_end:])

    if duplicated:
        return DNASequence(sequence="".join(new_sequence_parts))
    else:
        return dna

def inversion_mutation(dna: DNASequence, inversion_rate: float) -> DNASequence:
    """Performs inversion mutation.
    
    With probability `inversion_rate`, selects a random segment of the DNA sequence
    and reverses the order of nucleotides within that segment.

    Args:
        dna: The DNA sequence to mutate.
        inversion_rate: Probability of performing an inversion.

    Returns:
        A new DNASequence, potentially with an inverted segment.
    """
    if not 0.0 <= inversion_rate <= 1.0:
        raise ValueError("Inversion rate must be between 0.0 and 1.0")

    if random.random() < inversion_rate:
        n = len(dna.sequence)
        if n < 2:
            return dna # Cannot invert segments smaller than 2

        # Choose two distinct indices
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, n - 1)

        # Ensure start_index < end_index
        start_index = min(idx1, idx2)
        end_index = max(idx1, idx2) + 1 # Slice is exclusive at the end

        segment_to_invert = dna.sequence[start_index:end_index]
        inverted_segment = segment_to_invert[::-1]
        
        new_sequence = dna.sequence[:start_index] + inverted_segment + dna.sequence[end_index:]
        return DNASequence(sequence=new_sequence)
    else:
        return dna

def codon_substitution_mutation(dna: DNASequence, substitution_rate: float) -> DNASequence:
    """Performs codon substitution mutation.
    
    Iterates through the sequence codon by codon (non-overlapping triplets)
    and has `substitution_rate` chance to replace the entire codon with a 
    different, randomly generated codon.

    Args:
        dna: The DNA sequence to mutate.
        substitution_rate: Probability of substituting each codon.

    Returns:
        A new DNASequence with potential codon substitutions.
    """
    if not 0.0 <= substitution_rate <= 1.0:
        raise ValueError("Substitution rate must be between 0.0 and 1.0")
        
    sequence_list = list(dna.sequence)
    n = len(sequence_list)
    mutated = False
    
    # Iterate codon by codon
    for i in range(0, n - CODON_LENGTH + 1, CODON_LENGTH):
        if random.random() < substitution_rate:
            original_codon = "".join(sequence_list[i : i + CODON_LENGTH])
            # Generate a *different* random codon
            new_codon = int_to_dna(random.randint(0, 63))
            while new_codon == original_codon:
                 new_codon = int_to_dna(random.randint(0, 63))
                 
            # Replace the codon in the list
            sequence_list[i : i + CODON_LENGTH] = list(new_codon)
            mutated = True
            
    if mutated:
        return DNASequence(sequence="".join(sequence_list))
    else:
        return dna

def translocation_mutation(dna: DNASequence, translocation_rate: float) -> DNASequence:
    """Performs translocation mutation.
    
    With probability `translocation_rate`, selects two non-overlapping segments
    and swaps their positions.
    Note: Length remains the same, but gene structure can be significantly altered.
    """
    if not 0.0 <= translocation_rate <= 1.0:
        raise ValueError("Translocation rate must be between 0.0 and 1.0")

    if random.random() < translocation_rate:
        n = len(dna.sequence)
        if n < 4: # Need at least 4 nucleotides for two segments of length 1+ and space between
            return dna 

        # Determine lengths for the two segments (e.g., 1 to n/3 ?)
        max_len = max(1, n // 3)
        len1 = random.randint(1, max_len)
        len2 = random.randint(1, max_len)
        
        # Choose start positions ensuring segments + gap fit
        if n < len1 + len2 + 1:
            return dna # Cannot fit two segments and a gap
            
        # Try a few times to find non-overlapping segments
        for _ in range(10): # Avoid infinite loop if finding points is hard
            p1 = random.randint(0, n - (len1 + len2 + 1))
            p2 = random.randint(p1 + len1 + 1, n - len2) # Ensure p2 starts after p1 ends + gap
            
            start1, end1 = p1, p1 + len1
            start2, end2 = p2, p2 + len2
            
            # Extract segments and the part between them
            seg1 = dna.sequence[start1:end1]
            seg2 = dna.sequence[start2:end2]
            middle_part = dna.sequence[end1:start2]
            
            # Reconstruct: Start + Seg2 + Middle + Seg1 + End
            new_sequence = (
                dna.sequence[:start1] 
                + seg2 
                + middle_part 
                + seg1 
                + dna.sequence[end2:]
            )
            
            if len(new_sequence) == n: # Sanity check length
                return DNASequence(sequence=new_sequence)
            else:
                # This indicates an issue with slicing logic, fallback
                print(f"Warning: Translocation resulted in incorrect length ({len(new_sequence)} vs {n}). Returning original.")
                return dna 
                
        # If failed to find valid points after attempts, return original
        return dna 
    else:
        return dna

# --- Operator Registry ---

MUTATION_REGISTRY: Dict[str, Any] = {
    "point_mutation": point_mutation,
    "insertion": insertion_mutation,
    "deletion": deletion_mutation,
    "gene_duplication": gene_duplication_mutation,
    "inversion": inversion_mutation,
    "codon_substitution": codon_substitution_mutation,
    "translocation": translocation_mutation,
    # Add other mutation types here as implemented (e.g., codon_substitution, insertion, deletion)
}

def get_mutation_operator(config: Dict[str, Any]) -> Any:
    """Gets the mutation function based on configuration.

    Args:
        config: Dictionary typically containing {'type': 'operator_name', 'rate': float, ...}

    Returns:
        The mutation function.
    """
    operator_type = config.get("type", "point_mutation") # Default to point_mutation
    if operator_type not in MUTATION_REGISTRY:
        raise ValueError(f"Unknown mutation operator type: {operator_type}")
    
    # Here we assume the mutation function might need the rate passed separately
    # An alternative is to use functools.partial to pre-bind the rate if the function signature allows
    # Example: return functools.partial(MUTATION_REGISTRY[operator_type], mutation_rate=config.get('rate', 0.01))
    # For now, just return the function; rates are passed separately in the evolution loop
    return MUTATION_REGISTRY[operator_type]

# --- Example Usage ---
if __name__ == '__main__':
    original_dna = DNASequence("ATGCGTACGT" * 5)
    mutation_rate = 0.1
    print(f"Original DNA ({len(original_dna)}nt): {original_dna}")
    print(f"Mutation Rate: {mutation_rate}")

    mutated_dna = point_mutation(original_dna, mutation_rate)
    print(f"Mutated DNA  ({len(mutated_dna)}nt): {mutated_dna}")

    # Count differences
    diffs = sum(1 for i in range(len(original_dna)) if original_dna.sequence[i] != mutated_dna.sequence[i])
    print(f"Differences: {diffs}")

    # Test getting operator from config
    config = {"type": "point_mutation", "rate": 0.2}
    op = get_mutation_operator(config)
    # Note: The get function just returns the operator; rate needs to be passed separately here
    mutated_dna_2 = op(original_dna, mutation_rate=config["rate"]) 
    print(f"\nUsing get_mutation_operator('{config['type']}', rate={config['rate']}):")
    print(f"Mutated DNA 2({len(mutated_dna_2)}nt): {mutated_dna_2}")
    diffs_2 = sum(1 for i in range(len(original_dna)) if original_dna.sequence[i] != mutated_dna_2.sequence[i])
    print(f"Differences: {diffs_2}")
    
    # Test Insertion/Deletion
    print("\n--- Testing Insertion/Deletion ---")
    dna_indel = DNASequence("ABCDEFGHIJ")
    ins_rate = 0.2
    del_rate = 0.2
    print(f"Original: {dna_indel} (Length: {len(dna_indel)})")
    
    inserted_dna = insertion_mutation(dna_indel, ins_rate)
    print(f"After Insertion (rate={ins_rate}): {inserted_dna} (Length: {len(inserted_dna)})")
    
    deleted_dna = deletion_mutation(dna_indel, del_rate)
    print(f"After Deletion (rate={del_rate}): {deleted_dna} (Length: {len(deleted_dna)})")
    
    # Combined
    mutated_indel = deletion_mutation(insertion_mutation(dna_indel, ins_rate), del_rate)
    print(f"After Insertion then Deletion: {mutated_indel} (Length: {len(mutated_indel)})")
    
    # Test Gene Duplication
    print("\n--- Testing Gene Duplication ---")
    # Need encoding functions here for START/STOP
    from nucleotide_strategy_evolution.encoding import START_CODON, STOP_CODONS
    gene1 = START_CODON + "AAA" + STOP_CODONS[0] # Gene 1: ATGAAATAA
    gene2 = START_CODON + "CCC" + STOP_CODONS[1] # Gene 2: ATGCCCTAG
    dna_dup_test = DNASequence("XXX" + gene1 + "YYY" + gene2 + "ZZZ")
    dup_rate = 1.0 # Force duplication for testing
    
    print(f"Original: {dna_dup_test} (Length: {len(dna_dup_test)})")
    duplicated_dna = gene_duplication_mutation(dna_dup_test, dup_rate)
    print(f"After Duplication (rate={dup_rate}): {duplicated_dna} (Length: {len(duplicated_dna)})")
    # Expect: XXXATGAAATAAATGAAATAA YYY ATGCCCTAGATGCCCTAG ZZZ
    
    dup_rate_low = 0.1
    duplicated_dna_low = gene_duplication_mutation(dna_dup_test, dup_rate_low)
    print(f"After Duplication (rate={dup_rate_low}): {duplicated_dna_low} (Length: {len(duplicated_dna_low)})")
    
    # Test Inversion
    print("\n--- Testing Inversion ---")
    dna_inv_test = DNASequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    inv_rate = 1.0 # Force inversion
    print(f"Original: {dna_inv_test}")
    inverted_dna = inversion_mutation(dna_inv_test, inv_rate)
    print(f"After Inversion (rate={inv_rate}): {inverted_dna}")
    # Check if length is the same and content is different (unless full sequence inverted)
    print(f"  Length same: {len(dna_inv_test) == len(inverted_dna)}")
    print(f"  Content different: {dna_inv_test.sequence != inverted_dna.sequence}")
    
    # Test Codon Substitution
    print("\n--- Testing Codon Substitution ---")
    dna_codon_test = DNASequence("ATGAAACCCTTTGGGTAA") # Multiple codons
    sub_rate = 0.5
    print(f"Original: {dna_codon_test}")
    subst_dna = codon_substitution_mutation(dna_codon_test, sub_rate)
    print(f"After Substitution (rate={sub_rate}): {subst_dna}")
    print(f"  Length same: {len(dna_codon_test) == len(subst_dna)}")
    diffs_codon = sum(1 for i in range(len(dna_codon_test)) if dna_codon_test.sequence[i] != subst_dna.sequence[i])
    print(f"  Nucleotide Diffs: {diffs_codon}")

    # Test Translocation
    print("\n--- Testing Translocation ---")
    dna_trans_test = DNASequence("ABCDEFGHIJ" + "KLMNOPQRST" + "UVWXYZ") # 26 chars
    trans_rate = 1.0 # Force translocation
    print(f"Original: {dna_trans_test}")
    # Seed to get predictable segments (may need adjustment based on random calls)
    random.seed(48)
    trans_dna = translocation_mutation(dna_trans_test, trans_rate)
    print(f"After Translocation (rate={trans_rate}): {trans_dna}")
    print(f"  Length same: {len(dna_trans_test) == len(trans_dna)}")
    # Example with seed 48 (approx): seg1='IJK', seg2='STU'. Start=8, Start2=18
    # Expected: ABCDEFGH + STU + LMNOPQR + IJK + VWXYZ
    # assert trans_dna.sequence == "ABCDEFGHSTULMNOPQRIJKVWXYZ" 