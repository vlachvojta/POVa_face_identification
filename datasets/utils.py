import numpy as np

def grayscale_to_color(images: list[np.ndarray], permute: bool = False) -> np.ndarray:
    # (H, W) grayscale images to to (H, W, C)
    images = [np.stack([image, image, image], axis=2) for image in images]
    if permute:
        # (H, W, C) to (C, H, W)
        images = [np.transpose(image, (2, 0, 1)) for image in images]

    return np.array(images)
