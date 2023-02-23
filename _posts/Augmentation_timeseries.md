---
layout: single
title: "Augmentation of Time series data"
excerpt: ""

date: 2023-02-23

categories:
    - Machine learning
tags:
    - Machine learning
    - Time series

toc: true
toc_label: "content"
toc_icon: "bars"
toc_sticky: true
---

## 시작하며
<!-- 도입부 -->
시계열 데이터는 일정한 시간 간격으로 발생하는 시간에 따라 정렬된 데이터를 의미하며, 오늘날 제조, 금융, 의료 등 다양한 분야에서 발생하고 있습니다. 현업에서 머신러닝(회귀,분류)을 위한 시계열 데이터는 다른 데이터(이미지, 텍스트 등)와 동일한 목적으로 활용됩니다. 예를 들면, 주식 가격 예측, 운동 동장 분류 등 시계열 데이터로 회귀, 분류 모델을 개발 및 적용하는 사례가 있습니다. 다른 데이터(이미지, 텍스트)는 웹상에 공유된 데이터도 많고 바로 서비스 적용 가능한 수준의 모델도 많아 모델 개발에 어려움이 없지만, 시계열 데이터는 공유된 데이터도 별로 없을 뿐더러 시계열 분류, 예측 목적의 모델도 찾아보기 어렵습니다. 앞서 언급한 문제에서 데이터 부족 문제를 해결하기 위한 방법 중 하나인 데이터 증강(Data augmentation)에 대해 알아보고자 합니다.

## 데이터 증강(Data augmentation)
### 데이터 증강이란?
데이터 증강이란 데이터 양을 늘리기 위한 기법 중 하나로, 보유한 데이터에 여러 변환을 적용하여 데이터 수를 증가시킵니다. Computer vision 분야에서 필수로 사용되며 아래 이미지와 같이 원본 이미지를 다양한 방식으로 변환해 원본 데이터에서 학습하지 못했던 부분을 모델이 학습할 수 있도록 합니다.   


데이터 증강의 궁극적인 목적은 한정적인 훈련 데이터를 증강시킴으로 모델 성능을 높이고 과적합 문제를 해결하고자 합니다.

### 시계열 데이터 증강의 중요성


## 데이터 증강 기법
<!-- 기법, 구현, 효과 -->
### Time Shifting
1. Forward and backward shift
2. Random shift

### Scaling
1. Uniform scaling
2. Non-uniform scaling

### Noise Addition
1. White noise addition
2. Colored noise addition

### Time warping


### Interpolation



## 데이터 증강 적용
<!-- Use Cases of Time Series Data Augmentation -->
### Improving performance
### Small datasets
### Imbalanced datasets
### Anomaly detection
### Prediction and forecasting




## 데이터 증강 한계 및 과제
1. Data quality issues
여전히 햄스터인가?? -> 증강한 데이터가 원본 데이터의 특성을 유지하고 있는가??
2. Augmentation method selection
3. Computational complexity


## 마치며
<!-- 결론 -->
1. Summary of key points
2. Future research directions
3. Augmetation of time series data에 대한 개인의견

