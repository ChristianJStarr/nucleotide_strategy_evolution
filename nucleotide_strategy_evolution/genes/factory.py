"""Factory function for creating Gene objects."""

from typing import Dict, Any, Optional, Type

from nucleotide_strategy_evolution.core.structures import Gene, DNASequence
# Import specific parameter dataclasses if needed for validation later
# from .types import GENE_TYPE_PARAM_MAP

# --- Gene Registry (Optional but recommended for scalability) ---
# Maps gene type strings to functions or classes responsible for creating them.
GENE_CREATION_REGISTRY: Dict[str, Any] = {}

def register_gene_type(gene_type_name: str, creation_func: Any):
    """Registers a function or class to handle creation of a specific gene type."""
    if gene_type_name in GENE_CREATION_REGISTRY:
        print(f"Warning: Overwriting registration for gene type '{gene_type_name}'")
    GENE_CREATION_REGISTRY[gene_type_name] = creation_func

# --- Factory Function ---

def create_gene(
    gene_type: str,
    parameters: Optional[Dict[str, Any]] = None,
    raw_dna_str: Optional[str] = None,
    expression_level: float = 1.0
) -> Gene:
    """Creates a Gene object, potentially using registered type-specific logic.

    Args:
        gene_type: The type identifier string for the gene.
        parameters: A dictionary of parameters for the gene.
        raw_dna_str: The raw DNA sequence string for the gene (optional).
        expression_level: The initial expression level (optional).

    Returns:
        A Gene object.
    """
    if parameters is None:
        parameters = {}
    
    # TODO PH1/PH2: Add validation based on gene_type using GENE_TYPE_PARAM_MAP or similar
    # if gene_type in GENE_TYPE_PARAM_MAP:
    #     ParamClass = GENE_TYPE_PARAM_MAP[gene_type]
    #     try:
    #         # Attempt to instantiate the dataclass to validate params
    #         # This assumes parameters dict keys match dataclass fields
    #         ParamClass(**parameters)
    #     except TypeError as e:
    #         raise ValueError(f"Invalid parameters for gene type '{gene_type}': {e}") from e
    
    # TODO PH2: Use GENE_CREATION_REGISTRY if populated to call specific constructors
    # if gene_type in GENE_CREATION_REGISTRY:
    #     return GENE_CREATION_REGISTRY[gene_type](parameters=parameters, ...)

    # Default basic Gene creation for now
    return Gene(
        gene_type=gene_type,
        parameters=parameters,
        raw_dna=DNASequence(sequence=raw_dna_str if raw_dna_str else ""),
        expression_level=expression_level
    )

# --- Example Registration (Illustrative - remove or move later) ---
# def create_risk_gene(parameters: Dict[str, Any], **kwargs) -> Gene:
#     # Add specific logic for risk gene creation/validation
#     print(f"Creating specific Risk Gene with params: {parameters}")
#     # Validate using RiskManagementGeneParams(***parameters)
#     return create_gene(gene_type="risk_management", parameters=parameters, **kwargs)
# 
# register_gene_type("risk_management", create_risk_gene) 