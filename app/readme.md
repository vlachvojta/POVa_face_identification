# Demo application

Basic applications for demonstration purposes.

## demo_webcam.py

Application used to identify person in an uploaded image (from a file or a webcam) - shows name, class, similarity and a reference photo of the identified person. 

**run:** python3 demo_webcam.py -s ../datasets/Identities   

## demo_dataset.py

Application used to test DataLoader and CelebA dataset. Shows annotations of each image filename, class and every attribute of the photo. It is possible to filter data based on class ID or based on selected attributes

**run:** python3 demo_dataset.py -s ../datasets/CelebA   

