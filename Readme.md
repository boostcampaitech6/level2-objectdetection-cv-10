# **재활용 품목 분류를 위한 Object Detection**


## 1 .프로젝트 개요


이번 프로젝트는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결하는 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다. 대회 지표는 mAP 50으로 진행되었으며 우리 팀원은 다양한 라이브러리와 모델들을 각각 맡아 실험을 진행하였고 최종적으로 앙상블 기법을 통해 Public 0.6352, Private 0.6178 점수를 냈습니다. 

우리팀은 점수를 위해 단순 실험을 진행하기보다는 이론 및 학습에 중점을 두고 Competition에 임했습니다. MMdetection, Detectron2 등 라이브러리의 사용법을 학습하고, EDA를 통해 Class imbalance 문제를 발견하고 data Augmentation과 전처리를 통해 문제를 하고자 했습니다. 또한 여러 모델의 특성을 이해하며 실험을 진행하고 앙상블을 진행하였습니다. 

### 1.1 맡은 역할


은성 : YOLO v7, v8 실험, 앙상블 실험

유민 :  cascade RCNN, Swin b, Swin L, down sampling, UniverseNet 실험, multi_scale실험

종진 : detectron2, NoisyAnchor 적용

진하 : EDA, RetinaNet, Detr 실험

현영 : detectron, efficient det 학습

### 1.2 모델


- YOLO
    - YOLOv7 - E6E
    - YOLOv7 - v8x
- efficientDet
- Faster-RCNN
    - Backbone - ResNext
- Cascade - RCNN
    - Backbone - ResNet50
    - Backbone - ResNet101
    - Backbone - Swin -b
    - Backbone - Swin-l
- Detr
- RetinaNet
- FocalNet
- UniverseNet

### 1.3 Loss 함수


- Cross entropy - base
- focal loss
    - 2-Stage Model 에서 사용

## 2. 문제 해결을 위한 기술적 시도


### 2.1 Data


- flip, rotate, Brightness, mosaic, mixup

저희는 데이터의 다양성을 높이기 위해 각자 실험에 맞게 다양한 Augmentation 실험을 했습니다.   flip, rotate 등 기법을 random하게 사용하였고 그늘진 영역이 존재하거나, 조명 때문에 특정 영역만 밝은 경우를 가정하여 데이터를 위해 Brighness를 조절하는 기법들을 사용하였습니다. 

각 모델마다 특성에 맞게 여러 augmentation을 실험하여 성능향상에 도움이 되는 기법들을 찾았습니다. YOLO 같은경우 작은 물체의 객체를 탐지하지 못하는 문제가 발생하여 mosaic 기법을 사용하였더니 작은 물체를 더 탐지를 잘하여 성능이 향상되는 것을 확인 할 수 있었습니다.

![image](https://github.com/boostcampaitech6/level2-objectdetection-cv-10/assets/98599867/e495ab61-52c9-417c-a874-35a4bf01c233)


                                      

- 데이터 Imbalance 문제를 해결을 위한 DownSampling

또한 EDA결과 하나의 이미지 내에 등장하는 Object의 개수가 너무 많은 경우 train하는과정에서 방해적인 요소가 된다 판단하여 Z score 에서 6sigma를 벗어난 데이터들을 DownSampling 하여 학습을 진행하였습니다.

![image](https://github.com/boostcampaitech6/level2-objectdetection-cv-10/assets/98599867/079cb074-cf72-4d5c-9048-4b8c965619ba)


### 2.2 Noisy-Anchors

---

- ‘**Learning from Noisy Anchors for One-stage Object Detection’ 논문을 보고 프로젝트에 적용**
- IoU(Intersection over Union)을 통해 positive와 negative anchor를 나누게 되면 noise anchor가 발생해 학습이 잘 되지 않기 때문에 Anchor의 좋고(positive) 나쁨(negative)을 2분할하여 표현하지 않고 동적으로 연속적인 값 soft label로 표현해 성능향상

### 2.3 앙상블

---

- WBF(Weighted Boxes Fusion) 와 NMS

저희는 다양한 모델을 사용하여 일정 성능이 나온 모델들을 기준으로 다양성을 높이기 위해 다양한 앙상블을 진행하였습니다. NMS의 경우 높은 confidence score의 bounding box를 선택하기 때문에 평균적인 localization을 구할 수 없었고 그에 반해 WBF bonding box의 평균적인 box의 위치를 구하기 때문에 단일모델에서는 NMS를 사용하였고 여러 모델에서는 다양성을 살리기 위한 WBF를 사용하였습니다. 

### 배운점

---

- object detection task의 전체 흐름도
- image - label(annotation)형태의 데이터에 대한 이해

### 아쉬운 점

---

- 좋은 Validation Set 을 잘 만들지 못해 판단하는데 어려움.
- EDA를 통한 분석이 늦었고 체계적으로 분석하지 못함
- mmcv 3.0 이상이 필요로 하는 최신 모델을 사용하지 못함
- data cleaning
- 실험을 제대로 기록하지 못하였고 그 결과 공유되지 못한 것들이 많음
