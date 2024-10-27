import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# add current working directory + parent to path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from datasets.utils import grayscale_to_color


class ORLDataset:
    """
    Olivetti Research Laboratory dataset.
    =================   =====================
    Classes                                40
    Samples total                         400
    Dimensionality      64 x 64 pixels = 4096
    Features            real, between 0 and 1
    =================   =====================
    """
    def __init__(self):
        self.dataset = fetch_olivetti_faces()
        self.images = self.dataset.images # (400, 64, 64)
        self.images = grayscale_to_color(self.images, permute=True) # (400, 3, 64, 64)
        self.targets = self.dataset.target  # (400,)
        self.targets = [str(target) for target in self.targets]  # for testing purposes
        self.DESCR = self.dataset.DESCR

        # data = images in 1D
        self.data = self.dataset.data      # (400, 4096) 64*64=4096

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> (np.ndarray, int):
        return self.images[idx], self.targets[idx]

    def save_images(self, folder_path: str):
        """for every target, create one figure with 10 subplots and save it to a file in folder path."""

        import os
        os.makedirs(folder_path, exist_ok=True)

        for i in range(40):
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            # add title
            fig.suptitle(f'Class {i+1}')
            for j, ax in enumerate(axs.flatten()):
                ax.imshow(self.images[i*10 + j], cmap='gray')
                ax.axis('off')
            fig.tight_layout()
            print(f'saving image_{i}.png')
            plt.savefig(f'{folder_path}/images_{i}.png')
            plt.close()

    def print_stats(self):
        print('')
        print('------ ORL Dataset Stats ------')
        print(f"len(self.images): {len(self.images)}")
        print(f"self.images[0].shape: {self.images[0].shape}")

        print(f"self.data.shape: {self.data.shape}")

        print(f"self.targets.shape: {self.targets.shape}")

        print(f'type(self.images): {type(self.images)}')
        print(f'self.targets.shape: {self.targets.shape}')

        print(f'type(self.data): {type(self.data)}')

        print(f'{self.targets[:25] = }')
        print('')

    def print_description(self):
        print(self.DESCR)


if __name__ == '__main__':
    orl_dataset = ORLDataset()
    orl_dataset.print_stats()
