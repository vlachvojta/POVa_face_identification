import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetectionEngine:
	def __init__(self, device: str = 'cuda', keep_all=True):
		self.device = device
		self.model = MTCNN(keep_all=keep_all, device=self.device)
  
	def __call__ (self, image):
     
		boxes, probs, landmarks = self.model.detect(image, landmarks=True)
		results = []

		if boxes is not None:
			for box, landmark in zip(boxes, landmarks):
				results.append({
					"box": box.tolist(),
					"landmarks": landmark.tolist()
				})
        
		return results

if __name__ == "__main__":
    face_detector = FaceDetectionEngine('cpu', True)
    image = Image.open('lorelai.webp')
    result = face_detector(image)
    print("Detection results:", result)
