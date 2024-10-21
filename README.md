# POVa_face_identification
POVa project (Computer Vision) at FIT (B|V)UT. 2024/2025 winter semestr

## Proposal
- [ ] prezentace
 - outline your experiments including datasets and evaluation metrics

## TODOs
- [ ] demo app
  - camera input, take a photo, run engine, find closest embeddings in a database, show result
- [ ] research existing datasets [Martin]
- [ ] facial detection [Zuzka]
  - [ ] find and test a few existing approaches
  - [ ] OpenCV, MTCNN
- [ ] facial recognition / identification  [Vojta]
  - [ ] evaluation of existing models
  - [ ] fine-tuning (training on a new task)

- [ ] create our dataset


## Assignement
Prepare a demo application demonstrating facial recognition in good lighting conditions. Evaluate accuracy on you own data and on a existing datset.

Ideal approach is:
- Detect faces using existing detector. Good choices are OpenCV, Dlib or MTCNN https://github.com/DCGM/mtcnn.
- Align the face based detected facial features (map to avarage face).
- Extract face fingerprint using a convolutional neural network. You can start with some pretrained network or train or fine-tune your own - search Model Zoo for suitable network https://github.com/BVLC/caffe/wiki/Model-Zoo.
- Search database of faces.

Sources:
- https://github.com/betars/Face-Resources
- OpenFace https://cmusatyalab.github.io/openface/
- https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
- https://kpzhang93.github.io/MTCNN_face_detection_alignment/
- http://www.openu.ac.il/home/hassner/projects/augmented_faces/
- http://www.robots.ox.ac.uk/~vgg/software/vgg_face/


