from __future__ import annotations
import os
import csv
from enum import Enum
from facenet_pytorch import MTCNN

from PIL import Image
import cv2
import numpy as np

from datasets.data_structure import ImageData
from datasets.data_parser import DataParser
from face_detection.face_detection_engine import FaceDetectionEngine


class Normalization(Enum):
    MEAN_STD = "mean_std"
    MIN_MAX = "min_max"
    _0_1 = "0_1"
    _1_1 = "1_1"


class Squarify(Enum):
    CROP = "crop"
    AROUND_FACE = "around_face"
    AROUND_FACE_STRICT = "around_face_strict"


class ImagePreProcessor:
    def __init__(self, face_detection_engine=None, squarify: Squarify = None,
                 normalize: Normalization = None, resize: int = None):
        self.face_detection_engine = face_detection_engine
        self.squarify = squarify
        self.normalize = normalize
        self.resize = resize

    def __call__(self, image: np.ndarray, save_image: bool = False, image_src: str = None) -> np.ndarray:
        assert image.ndim == 3, f"Image should have 3 dimensions (H, W, C) or (C, H, W), got {image.ndim} dimensions with shape {image.shape}"
        assert image.dtype == np.uint8, f"Image should be of type np.uint8, got {image.dtype}"
        assert image.shape[2] == 3, f"Image should have 3 color channels (H, W, C), got {image.shape} channels"

        orig_image = image.copy()

        image = self.squarify_image(image, self.squarify, image_src)

        if self.resize:
            image = cv2.resize(image, (self.resize, self.resize))

        if save_image:
            image_with_background = np.zeros_like(orig_image)
            image_with_background[:image.shape[0], :image.shape[1]] = image
            image_render = np.hstack([orig_image, image_with_background])
            tmp_path = "tmp-renders/"
            os.makedirs(tmp_path, exist_ok=True)
            existing_names = os.listdir(tmp_path)
            img_name = f'squarify_{self.squarify}_norm_{self.normalize}_resize_{self.resize}.png'
            img_path = os.path.join(tmp_path, img_name)
            # print('Saving debug image:', img_path)
            cv2.imwrite(img_path, image_render)

        if self.normalize:
            if self.normalize == Normalization.MEAN_STD:
                image = (image - 127.5) / 128.0
            elif self.normalize == Normalization.MIN_MAX:
                image = (image - image.min()) / (image.max() - image.min())
            elif self.normalize == Normalization._0_1:
                image = image / 255.0
                # rescale to [0, 1]
            elif self.normalize == Normalization._1_1:
                image = image / 127.5 - 1.0
                # rescale to [-1, 1]
            else:
                raise ValueError(f"Unknown normalization method: {self.normalize}")

        return image

    def squarify_image(self, image, squarify: Squarify, image_src: str):
        if self.squarify is None:
            return image

        if self.squarify == Squarify.CROP:
            return self.squarify_crop(image)
        elif self.squarify == Squarify.AROUND_FACE:
            # detect face box and crop the image around the face
            assert self.face_detection_engine, "Face detection engine must be provided to squarify around face."
            results = self.face_detection_engine(image)
            if results and len(results) > 0:
                result = results[0]
                padding = 20
                face = result['box']
                l, t, r, b = face
                l, t, r, b = int(l), int(t), int(r), int(b)
                l = max(0, l - padding)
                t = max(0, t - padding)
                r = min(image.shape[1], r + padding)
                b = min(image.shape[0], b + padding)
                # crop face and pad to make it square
                if b - t > r - l:
                    pad = (b - t - (r - l)) // 2
                    l = max(0, l - pad)
                    r = min(image.shape[1], r + pad)
                elif r - l > b - t:
                    pad = (r - l - (b - t)) // 2
                    t = max(0, t - pad)
                    b = min(image.shape[0], b + pad)

                t = max(0, t)
                b = min(image.shape[0], b)
                l = max(0, l)
                r = min(image.shape[1], r)
                # complete squarify if needed
                h, w = b - t, r - l
                if h > w:
                    pad = (h - w) // 2
                    l = max(0, l - pad)
                    r = min(image.shape[1], r + pad)
                elif w > h:
                    pad = (w - h) // 2
                    t = max(0, t - pad)
                    b = min(image.shape[0], b + pad)
                image = image[t:b, l:r]
                # print(f'Squarified around face: {face} -> {l, t, r, b} (w: {r-l}, h: {t-b}), shape: {image.shape} orig shape: {orig_image.shape}')
                image = cv2.resize(image, (160, 160))
                return image
            else:
                print(f"Skipping squarify around face on image {image_src} as no face was detected.")
                return self.squarify_crop(image)
        elif self.squarify == Squarify.AROUND_FACE_STRICT:
            assert self.face_detection_engine, "Face detection engine must be provided to squarify around face."
            return self.face_detection_engine.strict_detection(image, image_src)
        else:
            raise ValueError(f"Unknown squarify method: {self.squarify}")

    def squarify_crop(self, image):
        # crop the image to make it square uniformly from the center
        h, w = image.shape[:2]
        if h > w:
            image = image[(h - w) // 2:(h + w) // 2, :]
        elif w > h:
            image = image[:, (w - h) // 2:(w + h) // 2]
        return image


class ImagePreProcessorMTCNN(ImagePreProcessor):
    def __init__(self, device: str = 'cpu'):
        # if keep_all is False, only the first face is returned
        # self.model = MTCNN(keep_all=False, device=device)
        self.face_detection_engine = FaceDetectionEngine(device=device, keep_all=False)
        super().__init__(face_detection_engine=self.face_detection_engine, 
                         squarify=Squarify.AROUND_FACE_STRICT, normalize=None, resize=None)
