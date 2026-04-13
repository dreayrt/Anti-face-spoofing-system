# Face Anti-Spoofing Preprocessing Pipeline
"""
Modular preprocessing pipeline for face anti-spoofing datasets.

Hỗ trợ 2 dataset:
  - CelebA Spoof:   python -m preprocessing.pipeline
  - FF-C23:         python -m preprocessing.pipeline_ffc23

Modules chung: augmentation.py, visualization.py
Modules CelebA:  config.py, splitting.py, cleaning.py, dataset.py, pipeline.py
Modules FF-C23:  config_ffc23.py, splitting_ffc23.py, frame_extraction.py,
                 cleaning_ffc23.py, dataset_ffc23.py, pipeline_ffc23.py
"""
