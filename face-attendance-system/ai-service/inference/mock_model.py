import numpy as np

class MockAntiSpoofModel:
    """Mock Deep Learning Model for Liveness Detection."""
    
    def __init__(self, weights_path=None):
        self.is_loaded = True
        print(f"Loaded Mock Anti-Spoofing Model from {weights_path}")
        
    def predict(self, face_crop):
        """
        Simulate predicting if a face crop is real or spoof (print/replay attack).
        Returns a float between 0.0 (fake) and 1.0 (real).
        """
        # Imagine extracting features through a CNN here.
        # For our mock, return a random score indicating probability of being a real face.
        # High probability means it's real.
        score = np.random.uniform(0.75, 0.99)
        
        # If the input was completely black or noisy, we might return a low score.
        if np.mean(face_crop) < 10:
            score = 0.1
            
        return float(score)

class MockFaceRecognitionModel:
    """Mock Deep Learning Model for Face Matching (e.g., ArcFace/FaceNet)."""
    
    def __init__(self, weights_path=None):
        self.is_loaded = True
        print(f"Loaded Mock Face Recognition Model from {weights_path}")
        
    def get_embedding(self, face_crop):
        """
        Simulate extracting a 128D or 512D feature vector (embedding) from a face crop.
        """
        # In reality, this would be a forward pass through a ResNet/MobileNet.
        embedding = np.random.rand(128).astype(np.float32)
        # L2 normalization of the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def match(self, embedding1, embedding2, threshold=0.5):
        """
        Simulate comparing two embeddings using cosine similarity.
        """
        similarity = np.dot(embedding1, embedding2)
        is_match = similarity > threshold
        return bool(is_match), float(similarity)
