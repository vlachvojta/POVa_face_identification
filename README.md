# POVa_face_identification
POVa project (Computer Vision) at FIT (B|V)UT. 2024/2025 winter semestr


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

## Proposal Feedback
**TLDR**
- create your own dataset with 10 people with a phone. Fine-tune model using similar augmentations.
- pytorch-metric-learning + ArcFaceLoss
- pretrained models CLIP (OpenAI), hugging face, ...

Dear students,

in general, the proposal makes sense. However I have some questions and remarks.

- Face detection and alignment - This can be only two lines of code using facenet-pytorch. This part may not be worth mentioning.
- What will the "demo application" do?
- The hard part is probaly "evaluation and possibly fine-tuning".
- I would suggest you collect small dataset in "challenging lighting conditions" and use it as a test set (e.g. only 10 different people). It can be mobile photos in the dark. You can try to finetune a model on the large dataset with augmentations simulating the lighting conditions.
- For finetuning, I would personally suggest pytorch-metric-learning. Good and stable loss functions are mostly central losses (e.g. ArcFaceLoss). Pair and tripplet losses do not work that well (and easily). You can use any pretrained models you like, but CLIP models work well (from OpenAI, models on hugging face, sentence_transformer library, ...). You can try some very small models like resnet18. Resolution 128x128 is usually enough. If you use dataset like CelebA, the pretrained model does not have to be pretrained for facial identification.

Regards,
Michal Hradiš

## DONE
### Proposal
- [x] paper (článek)
 - outline your experiments including datasets and evaluation metrics
- [x] evaluation style and results with baseline

## Questions for 27.11. meeting
- [x] dataset:
      - specific subset - celebA, TODO find a good descriptor
      - (phone dataset, historical?)
- [x] facial detection upgrade? NOP. It's good.
- [x] demo app? - Martin will create a skeleton
- [x] who does what?
      - Vojta: training
      - Martin: data
      - Zuzka: detection engine

## Other TODOs
- [x] research existing datasets [Martin]
  - [x] labeled faces in the wild (13 000 images, 5749 lidí (1680 lidí s dvěma a více fotkama))
  - [x] (CASIA web-Face)
  - chosen CelebA
- [x] facial detection [Zuzka]
  - [x] find and test a few existing approaches
  - chosen DLib
  - [x] ~~OpenCV, MTCNN~~
  - [x] testing on existing dataset
- [x] facial recognition / identification  [Vojta]
  - [x] evaluation of existing models (code is ready)
  - [x] fine-tuning (training on a new task)

- demo app:
  - [x] load images, filters + view attributes
  - camera input, take a photo, run engine, find closest embeddings in a database, show result
- [ ] create our dataset

## Last parts
- [x] [Zuzka] render výsledků trénování - jak jsme se bavili, vyrenderovat nejhorší obrázky z batche, nebo prostě něco vyrenderovat, aby se dala udělat nějaká kvalitativní analýza trénování. (např. vyrenderovat všechny obrázky v batchi do jenoho obrázku a do názvu souboru dát train loss, příklad renderování více obličejů do obrázku) )
- [x] [Vojta] integrace face detection do trénování - to pokud vím, ještě není, takže tam zatím cpeme neupravený data... 😅
- [x] Změřit, jak si vede náš Facenet model na našem datasetu CelebA  - jak si vůbec vede předtrénovaný model bez samotného trénování? (viz face_identification_evaluation.py)
- [x] vybírání atributů, které matou model
