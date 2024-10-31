import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetectionEngine:
	def __init__(self, device: str = 'cuda', keep_all=True):
		self.device = device
		self.model = MTCNN(keep_all=keep_all, device=self.device)
  
	def __call__ (self, image_path, output_path=None):
		image = Image.open(image_path)
		boxes, _ = self.model.detect(image)
        
		image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

		if boxes is not None:
			for box in boxes:
				x1, y1, x2, y2 = map(int, box)
				cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
		if output_path:
			cv2.imwrite(output_path, image_cv)
        
		return boxes if boxes is not None else []
