import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


class FaceDetectionEngine:
	def __init__(self, device: str = 'cuda', keep_all=True):
		self.device = device
		self.model = MTCNN(keep_all=keep_all, device=self.device)

	def __call__ (self, image):
		if isinstance(image, np.ndarray):
			assert image.ndim == 3, f"Image should have 3 dimensions, got {image.ndim}"
			assert image.shape[2] == 3, f"Image should have 3 channels (RGB) at the last dimension, got {image.shape[2]}"

		boxes, probs, landmarks = self.model.detect(image, landmarks=True)
		results = []

		if boxes is not None:
			for box, prob, landmark in zip(boxes, probs, landmarks):
				results.append({
					"box": box.tolist(),  # return box as left, top, right, bottom (not x, y, width, height)
					"prob": prob,
					"landmarks": landmark.tolist()
				})

		return results

	def strict_detection(self, image, image_src: str = None):
		# detect one face
		face = self.model(image)

		# check if face was detected
		if face is None:
			print(f"Skipping squarify around face on image {image_src} as no face was detected.")
			face = cv.resize(image, (160, 160))

		# check face has shape [C, H, W]
		if not face.ndim == 3:
			raise ValueError(f"Face should have 3 dimensions (C, H, W) or (H, W, C), got {face.ndim} dimensions with shape {face.shape}")

		if face.shape[0] != 3 and face.shape[2] == 3:
			face = face.transpose(2, 0, 1)

		if face.shape[1] != 160 or face.shape[2] != 160:
			raise ValueError(f"Face should have shape (3, 160, 160), got {face.shape}")

		return face

	def crop_faces(self, image, threshold=0.8) -> tuple[list[np.ndarray], list[float]]:
		results = self(image)

		# filter out boxes with lower probability than threshold
		if threshold is not None:
			results = [result for result in results if result['prob'] > threshold]

		# crop faces
		faces = []
		probs = []
		for result in results:
			l, t, r, b = result['box']
			l, t, r, b = int(l), int(t), int(r), int(b)

			# crop face
			face = image[t:b, l:r]
			faces.append(face)
			probs.append(result['prob'])

		return faces, probs

def show_image_with_boxes(image, result):
	# show image with boxes
	from matplotlib.patches import Rectangle
	from matplotlib.patches import Circle

	plt.imshow(image)
	ax = plt.gca()

	for face in result:
		# add box
		l, t, r, b = face['box']
		rect = Rectangle((l, t), r-l, b-t, fill=False, color='green')
		ax.add_patch(rect)

		# add probability
		ax.text(l, t, f"{face['prob']:.2f}", color='red')

		# add landmarks
		for point in face['landmarks']:
			dot = Circle((point[0], point[1]), radius=2, color='red')
			ax.add_patch(dot)

	plt.savefig('image_with_boxes.png')
	plt.show()


def example_usage():
	face_detector = FaceDetectionEngine('cpu', True)
	image = np.array(Image.open('example.webp'))
	# image = image.transpose(2, 0, 1)
	print(f'Image shape: {image.shape}')
	result = face_detector(image)
	print("Detection results:", result)
	show_image_with_boxes(image, result)

	cropped_faces = face_detector.crop_faces(image)
	print(f'Number of cropped faces: {len(cropped_faces)}')
	for i, face in enumerate(cropped_faces):
		plt.imshow(face)
		plt.show()

def testing_detection_with_MTCNN():
	# mtcnn = MTCNN(keep_all=True)
	mtcnn = MTCNN()

	image = Image.open('example.webp')
	face = mtcnn(image)
	print(f'Faces ({face.shape}): {face}')

	# face is in float [-1, 1] range, convert to uint8 [0, 255]
	face = (face + 1) * 127.5
	print(f'\tface ({face.shape}): {face}')
	face = face.numpy().astype(np.uint8)

	# face has shape [C, H, W], convert to [H, W, C]
	face = np.transpose(face, (1, 2, 0))
	print(f'\tface ({face.shape}): {face}')

	# save face to file
	Image.fromarray(face).save('face.png')

if __name__ == "__main__":
	example_usage()
	# testing_detection_with_MTCNN()
	# show_image_with_boxes()
