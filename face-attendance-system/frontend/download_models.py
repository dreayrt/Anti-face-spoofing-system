import urllib.request
import os

os.makedirs('public/models', exist_ok=True)
base_url = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model/'

files = [
    'tiny_face_detector_model-weights_manifest.json',
    'tiny_face_detector_model.weights.bin',
    'face_landmark_68_model-weights_manifest.json',
    'face_landmark_68_model.weights.bin'
]

for f in files:
    print(f"Downloading {f}...")
    urllib.request.urlretrieve(base_url + f, 'public/models/' + f)

print("Done.")
