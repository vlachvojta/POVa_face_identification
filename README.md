# POVa_face_identification
POVa project (Computer Vision) at FIT (B|V)UT. 2024/2025 winter semestr

## TODOs
- [ ] demo app
  - camera input, take a photo, run engine, show result
- [ ] research existing datasets
- [ ] create our dataset
- [ ] facial recognition engine (do I know you?)
  - [ ] evaluation of existing models
  - [ ] fine-tuning



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


