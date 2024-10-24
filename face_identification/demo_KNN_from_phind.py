import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the ORL dataset
orl_dataset = fetch_olivetti_faces()

# Get the images and labels
images = orl_dataset.images
labels = orl_dataset.target

# Flatten the images
X = images.reshape(images.shape[0], -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

# Make predictions
y_pred = knn.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Display confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Function to display misclassified faces
def show_misclassified():
    misclassified = np.where(y_test != y_pred)[0]
    
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for i, ax in enumerate(axs.flatten()):
        if i < len(misclassified):
            idx = misclassified[i]
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()

show_misclassified()
