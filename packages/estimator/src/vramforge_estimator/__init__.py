"""VRAMForge calculation core.

Estimates fine-tuning peak memory from a full, untruncated analysis of the dataset with the
real tokenizer and the trainer's preprocessing. See plan.md and docs/architecture.md.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
