# Face Anti-Spoofing Preprocessing Pipeline
"""
Modular preprocessing pipeline for face anti-spoofing datasets.

Hỗ trợ 3 dataset:
  - CelebA Spoof:   python -m preprocessing.pipeline
  - FF-C23:         python -m preprocessing.pipeline_ffc23
  - SiW:            python -m preprocessing.pipeline_siw

Modules chung: augmentation.py, visualization.py, face_alignment.py
Modules CelebA:  config.py, splitting.py, cleaning.py, dataset.py, pipeline.py
Modules FF-C23:  config_ffc23.py, splitting_ffc23.py, frame_extraction.py,
                 cleaning_ffc23.py, dataset_ffc23.py, pipeline_ffc23.py
Modules SiW:     config_siw.py, augmentation_siw.py, dataset_siw.py, pipeline_siw.py
"""
