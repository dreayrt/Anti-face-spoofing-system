"""
__main__.py – Allow running the pipeline as ``python -m preprocessing``.
"""
from .pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
