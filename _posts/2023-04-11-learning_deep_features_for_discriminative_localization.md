---
layout: single
title: "[Paper review] Learning Deep Features for Discriminative Localization"
# excerpt: ""

date: 2023-04-11

sitemap :
  changefreq : daily
  priority : 1.0

categories:
    - Paper review
tags:
    - Machine learning
    - Class activation map

use_math: true

toc: true
toc_label: "content"
toc_icon: "bars"
toc_sticky: true
---

## 시작하며
딥러닝 모델은 심층 신경망을 활용한 모델로 이미지, 텍스트 등 비정형 데이터에서 뛰어난 성능을 보여주고 있습니다. 하지만 딥러닝 모델의 한계점으로 내부 구조가 어떻게 동작하는지 확인할 수 없다는 단점이 있습니다. 즉, 신경망의 어떤 부분이 중요하고, 데이터의 어떤 부분을 보고 모델이 판단하는지 모른다는 것입니다. 이와 같이 딥러닝 모델의 해석 불가능하다는 한계점을 해결하기 위해 XAI(eXplainable AI, 설명가능한 AI) 분야의 연구도 계속해서 진행되고 있습니다.

이번 글에서는 XAI와 관련된 CAM(class activation map)에 다루고자 합니다. 지난번 리뷰한 논문에서 딥러닝 모델을 CAM으로 설명한 부분이 매우 흥미로웠습니다. CAM에 대해서 더 알아보고 싶어져 관련 논문인 'Learning Deep Features for Discriminative Localization'을 리뷰하면서 자세히 파악해보도록 하겠습니다.


<!-- Class activation map은 직역해보면 클래스별 활성 구간으로 해석할 수 있습니다. 모델이 학습한 클래스별 활성 구간으로 입력 데이터의 어떤 부분이 클래스 판별을 위해 사용되었는지 확인할 수 있습니다. -->


## 마치며

### Reference
[Learning Deep Features for Discriminative Localization, 14 Dec 2015](https://arxiv.org/abs/1512.04150)