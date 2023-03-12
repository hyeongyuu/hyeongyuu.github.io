---
layout: single
title: "Unsupervised learning"

date: 2023-02-04

categories:
    - Machine learning
tags:
    - Machine learning
    - Backlog

# toc: true
# toc_label: "content"
# toc_icon: "bars"
# toc_sticky: true
---
<br>
머신러닝은 크게 지도학습(Supervised Learning), 비지도학습(Unsupervised Learning), 강화학습(Reinforcement Learning)으로 나뉜다. 지도학습과 강화학습은 입력(Input)에 대한 결과(Output)를 주고 학습하는 방식이지만 비지도학습은 입력에 대한 결과가 없이 입력만으로 학습을 진행한다. 비지도학습을 시험에 빗대면 문제에 정답이 없는 논술형 시험과 비슷하다고 생각하는데 정답이 없기 때문에 정답을 판단하는 기준이 모호하고 결과는 해석하기 나름이라는 공통점이 있다.  

![unsupervised learning1](https://user-images.githubusercontent.com/82218035/119490037-5218cb00-bd97-11eb-8b2b-7e373249f9bd.PNG)

<br>

비지도학습의 가장 대표적인 기법에는 군집화(Clustering)가 있다. 군집화는 논술형 시험과 유사하다. 논술형 시험은 정해진 결론이 없고 유사한 결론이 있기 때문에 비슷한 결론끼리 묶음 처리가 가능할 것이다. 이처럼 정해진 답은 없지만 유사한 답을 묶어 처리하는 방식이 군집화의 원리이다. 그리고 군집화를 시행하고 각각의 군집에 속한 사람들의 특징을 파악하면 군집 내부에서 새로운 특징 도출이 가능한데 만약 답안으로 긍정적인 결론을 지은 사람들을 모아놓고 다른 특징을 파악해보면 대부분 사람들은 밝은 성격을 갖고 있다는 새로운 특징을 찾을 수 있을 것이다.

>군집화는 지도학습의 분류(Classification)와 공통점이 있다. 두 방식 모두 출력값이 분류 가능한 이산형 값으로 나와 데이터의 분류가 가능하다는 것이다. 그렇다면 비지도학습에서 지도학습의 회귀(Regression) 문제와 같이 연속형 값으로 예측은 불가능한가?

<br>

![unsupervised learning2](https://user-images.githubusercontent.com/82218035/119490066-5a710600-bd97-11eb-8861-a19c8c33e406.PNG)

비지도학습은 차원 축소(Dimensionality Reduction)로 데이터의 주요 특징을 추출하고 변환 가능하다. 차원 축소 기법인 주성분 분석(PCA, Principal Component Analysis)을 실시하여 차원을 축소하고 복원하는 과정에서 데이터의 주요 특징만 남는 원리를 활용해 비지도 학습의 이상치 탐지(Unsupervised Anomaly Detection)가 가능하다.

![unsupervised learning3](https://user-images.githubusercontent.com/82218035/119490097-68bf2200-bd97-11eb-842b-0275e107fe2f.PNG)

이때 입력 데이터의 특징을 추출하기 위해 오토인코더(Autoencoder)를 사용하는데 정상 데이터가 오토인코더를 거치면 정상 데이터가 출력되어 Input과 Output에 차이가 없는 반면, 비정상 데이터가 오토인코더를 거치면 정상 데이터가 출력되어 Input과 Output에 차이가 발생해 비정상 데이터가 구분 가능해진다.

<br>

인공지능의 목적은 비지도학습이 가능한 인공지능을 만드는 것이라 생각한다. 기계가 인간처럼 스스로 판단하고 행동하도록 만들기위해 머신러닝의 지도학습은 정답을 보면서 학습하지만, 비지도학습은 정답을 모르고 계속해서 학습을 진행하기 때문에 처음 맞닥뜨린 상황이더라도 스스로 판단을 한다. 그리고 기계가 처음 접하는 상황에 자신만의 판단을 만든다면, 사람처럼 성격과 인격을 갖춘 인공지능이 구현가능할 것이다. 이처럼 인공지능의 최종 목표는 비지도학습을 통해서 완성되고 구현될것이라 본다.  

<br>

##### 결론
- 비지도학습은 입력에 대한 결과가 없이 입력만으로 학습을 진행한다.
- 비지도학습은 차원 축소로 데이터의 주요 특징을 추출하고 변환 가능하다.
- 인공지능의 완전한 모습은 비지도학습으로 구현될것이다.
