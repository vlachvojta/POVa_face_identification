import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import cv2


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

	def crop_faces(self, image, threshold=0.8):
		results = self(image)

		# filter out boxes with lower probability than threshold
		boxes = [result for result in results if result['prob'] > threshold]

		# crop faces + align
		faces = []
		for box in boxes:
			l, t, r, b = box['box']
			# crop face
			face = image[t:b, l:r]
			faces.append(face)

		return faces

	# # example from previous code
	# # Load the image
	# img = Image.open(image_path)

	# # Detect faces
	# boxes, _ = self.mtcnn.detect(img)

	# if boxes is not None:
	# 	# Align faces
	# 	aligned = self.mtcnn(img)

	# 	# Compute embeddings
	# 	embedding = self.extract_face_embeddings(aligned)
	# 	return embedding


def show_image_with_boxes(image, result):
	# show image with boxes
	import matplotlib.pyplot as plt
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

	plt.show()


def example_usage():
	face_detector = FaceDetectionEngine('cpu', True)
	image = np.array(Image.open('example.webp'))
	# image = image.transpose(2, 0, 1)
	print(f'Image shape: {image.shape}')
	result = face_detector(image)
	print("Detection results:", result)
	show_image_with_boxes(image, result)


def test_mtcnn():
	import matplotlib.pyplot as plt
	image = np.array(Image.open('example.webp'))
	mtcnn = MTCNN(keep_all=True)

	print(f'Input image ({image.shape}):\n{image}')

	# boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
	# Detect faces

	# if boxes is not None:
	# Align faces
	aligned = mtcnn(image)
	boxes, _, landmarks = mtcnn.detect(image, landmarks=True)

	# show aligned faces
	for box, landmark in zip(boxes, landmarks):
		l, t, r, b = box.astype(int).tolist()
		print(f'box: {box}')
		print(f'l: {l}, t: {t}, r: {r}, b: {b}')
		print(f'landmark: {landmark}')

		face = image[t:b, l:r, :]
		print(f'face: {face.shape}\n{face}')
		# face_to_show = (face + 1) / 2 * 255
		# face_to_show = face_to_show.permute(1, 2, 0).numpy().astype(np.uint8)
		face_to_show = face
		print(f'face_to_show: {face_to_show.shape}\n{face_to_show}')

		# align face
		aligned_face = align_face(face_to_show, landmark)


		plt.imshow(face_to_show)
		plt.show()

	# # Compute embeddings
	# embedding = mtcnn(aligned)

	print('exiting'); exit()

	results = mtcnn(image)
	print(results.shape)

	for result in results:
		# plt show result
		result_image = result.permute(1, 2, 0).numpy().astype(np.uint8)
		print(f'showing image ({result_image.shape}):\n{result_image}')
		plt.imshow(result_image)
		plt.show()


def align_face(img, landmarks) -> np.ndarray:
    # Define desired left eye position
    desiredLeftEye = (0.35, 0.35)
    
    # Compute the rotation matrix
    dY = landmarks[1][1] - landmarks[0][1]
    dX = landmarks[1][0] - landmarks[0][0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    
    # Determine the scale of the new resulting image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredLeftEye[1] - desiredLeftEye[0])
    desiredDist *= img.size[1]
    scale = desiredDist / dist
    
    # Compute center (x, y)-coordinates of the image
    eyesCenter = ((landmarks[0][0] + landmarks[1][0]) // 2,
                  (landmarks[0][1] + landmarks[1][1]) // 2)
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    
    # Update the translation component of the matrix
    tX = img.size[0] * 0.5
    tY = img.size[1] * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])
    
    # Apply the affine transformation
    (w, h) = (img.size[0], img.size[1])
    (nw, nh) = (img.size[0], img.size[1])
    aligned_img = cv2.warpAffine(np.array(img), M, (nw, nh))
    
    # Return the result
    return aligned_img

def detect_and_align_faces(image_path):
    mtcnn = MTCNN(keep_all=True)
    
    # Load image
    img = Image.open(image_path)
    
    # Detect faces
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    
    # Align faces
    aligned_faces = []
    for i, box in enumerate(boxes):
        face = img.crop(box.tolist())
        aligned_face = align_face(face, landmarks[i])
        aligned_faces.append(aligned_face.resize((160, 160)))  # Resize to standard size
        
    return aligned_faces



if __name__ == "__main__":
	# example_usage()

	test_mtcnn()