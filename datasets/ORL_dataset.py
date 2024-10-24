import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

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
        self.images = self.dataset.images
        self.data = self.dataset.data
        self.target = self.dataset.target
        self.DESCR = self.dataset.DESCR

        print(f"len(self.images): {len(self.images)}")
        print(f"self.images[0].shape: {self.images[0].shape}")
        print(f"self.target.shape: {self.target.shape}")

        print(f"self.data.shape: {self.data.shape}")

        print(f'type(self.images): {type(self.images)}')
        print(f'self.target.shape: {self.target.shape}')

        print(f'type(self.data): {type(self.data)}')

        print(f'{self.target[:10] = }')

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


def main():
    orl_dataset = ORLDataset()

    orl_dataset.save_images('datasets/ORL_dataset_class_wise')

    exit(0)

    # Get the images
    images = orl_dataset.images

    # Display the first 10 images
    display_images(images[:10])

    # Display images from different individuals
    individuals = np.arange(0, len(images), 10)
    display_images(images[individuals])


# Function to display multiple images
def display_images(images, rows=2, cols=5):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 6))
    
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f"Image {i}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
