import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetectionEngine:
	def __init__(self, device: str = 'cuda', keep_all=True):
		self.device = device
		self.model = MTCNN(keep_all=keep_all, device=self.device)
  
	def __call__ (self, image):
		assert isinstance(image, np.ndarray), f"Image should be a numpy array, got {type(image)}"
		assert image.ndim == 3, f"Image should have 3 dimensions, got {image.ndim}"
		assert image.shape[2] == 3, f"Image should have 3 channels (RGB) at the last dimension, got {image.shape[2]}"
     
		boxes, probs, landmarks = self.model.detect(image, landmarks=True)
		results = []

		if boxes is not None:
			for box, landmark in zip(boxes, landmarks):
				results.append({
					"box": box.tolist(),  # return box as left, top, right, bottom (not x, y, width, height)
					"landmarks": landmark.tolist()
				})
        
		return results


def show_image_with_boxes(image, result):
	# show image with boxes
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	from matplotlib.patches import Circle

	plt.imshow(image)
	ax = plt.gca()

	for face in result:
		# x, y, width, height = face['box']
		# rect = Rectangle((x, y), width, height, fill=False, color='red')
		# ax.add_patch(rect)
		l, t, r, b = face['box']
		rect = Rectangle((l, t), r-l, b-t, fill=False, color='green')
		ax.add_patch(rect)

		for point in face['landmarks']:
			dot = Circle((point[0], point[1]), radius=2, color='red')
			ax.add_patch(dot)

	plt.show()


def example_usage():
	face_detector = FaceDetectionEngine('cpu', True)
	image = np.array(Image.open('example.webp'))
	# image = image.transpose(2, 0, 1)
	print(f'Image shape: {image.shape}')
	result = face_detector(image)
	print("Detection results:", result)
	show_image_with_boxes(image, result)


if __name__ == "__main__":
    example_usage()
