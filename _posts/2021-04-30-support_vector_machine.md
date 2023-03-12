---
layout: single
title: "Support vector machine"

date: 2023-02-04

categories:
    - Machine learning
tags:
    - Machine learning
    - Model
    - Backlog

# toc: true
# toc_label: "content"
# toc_icon: "bars"
# toc_sticky: true
---
<br>

머신러닝의 지도학습 기법으로 주로 분류에 자주 사용되는 SVM에 대하여 알아보고자 한다. SupportVectorMachine의 줄일말로 **SupprotVector(서포트 벡터)와 Hyperplane(초평면)을 이용해 DecisionBoundary(결정 경계)를 정의하고 분류를 시행하는 알고리즘**이다.

>DecisionBoundry  
두 개의 계층을 가지고 있는 통계적인 분류 문제에서,
결정 경계는 기본 벡터공간을 각 클래스에 대하여 하나씩 두 개의 집합으로 나누는 초표면이다. 분류기는 결정 경계의 한쪽에 있는 모든 점을 한 클래스, 다른 한쪽에 있는 모든 점을 다른 클래스에 속하는 것으로 분류한다.

**SVM은 최적의 결정 경계를 찾는 것이 목적인데** 여기서 최적의 결정 경계를 찾기 위해서는 Margin(마진)의 최대점을 찾아야한다. 마진은 결정 경계와 서포트 벡터 사이의 거리로 큰 값을 가질수록 분류 경계가 커짐을 뜻하기 때문에 좋은 분류 성능을 보여줄것이다.

![KakaoTalk_Photo_2021-05-01-16-55-49](https://user-images.githubusercontent.com/82218035/116775479-252a0e80-aa9e-11eb-9e02-f1a8d75e6545.png)

>SVM 구성요소  
Hyperplane : 초평면으로 고차원의 결정 경계를 뜻하기도 한다.  
SupportVector : 결정 경계와 가까이 있는 데이터 포인터  
margin : 결정 경계와 서포트 벡터 사이의 거리  

<br>

**SVM에서 결정 경계는 p차원의 데이터라면 p-1차원의 결정 경계가 만들어진다.** 예를 들어, 3차원의 데이터는 2차원의 평면으로 경계가 만들어지고, 2차원의 데이터는 1차원의 벡터(선)으로 경계를 만든다. 이러한 결정 경계의 특징 때문에 초기에 결정 경계를 구하는 알고리즘은 선형 분류만 가능했었고, 비선형 분류가 불가능했었다. 하지만 **데이터의 분포에 커널 트릭을 적용시켜 분포를 고차원으로 변형함으로써 비선형 분류도 가능하도록 만들었다.**

>커널트릭  
저차원 공간을 고차원으로 매핑해주는 작업, 선형 분리가 불가능한 1차원 데이터를 2차원의 공간으로 매핑하여 선형분리가 가능하도록 한다.

![KakaoTalk_Photo_2021-05-01-16-55-57](https://user-images.githubusercontent.com/82218035/116775481-2824ff00-aa9e-11eb-8c6f-dd261c5a964b.png)

위의 그림은 커널트릭을 활용해 선형분리가 불가능한 1차원의 데이터에 2차원의 데이터를 추가함으로써 선형분리가 가능하도록 하는것이다. **그렇다면 커널트릭을 적용해 저차원의 데이터를 고차원으로 매핑하여 시행하는 분류의 성능은 선형분류보다 정확한 성능을 보여주는 것인가?** 하는 의문이 생긴다. 2차원의 데이터를 1차원의 벡터로 선형 분류를 시행하는 방법과 2차원의 데이터를 3차원으로 변형시켜 2차원의 초평면으로 비선형 분류를 시행하는 방법중 어떤 것이 더 높은 정확도를 보여주는지 파악해보았다.

<img width="713" alt="스크린샷 2021-05-01 오후 4 26 48" src="https://user-images.githubusercontent.com/82218035/116775482-2ce9b300-aa9e-11eb-8767-52a9ac4119b6.png">
<br>
<br>
<img width="713" alt="스크린샷 2021-05-01 오후 4 27 03" src="https://user-images.githubusercontent.com/82218035/116775484-2e1ae000-aa9e-11eb-8caa-5ca9bf18d7f1.png">  

<br>
<br>

분류를 시행해 본 결과, **결론적으로 고차원 공간으로 변형시킨 방법이 데이터를 더 잘 분류했다.** 선형분리가 가능한 데이터에서 커널트릭을 활용한 분류가 마진이 더 컸으며, 비선형 분류를 해야하는 데이터도 선형분리가 불가능한 단점을 유연한 결정 경계로 데이터를 분류해 커널트릭을 활용한 분류가 항상 더 좋은 분류라 생각할수도있다. **그러나 커넡트릭을 활용한 분류는 한가지 큰 문제점이 있다. 머신러닝에서 자주 발생하는 문제점인 OverFitting(과적합)을 초래한다는 것인데, 데이터 차원을 증가시켜 고차원에서 학습시킨 모델은 다시 본래의 차원에서 과적합이 발생하기 때문이다.**

>머신러닝에서 모델이 유연할수록 과적합이 쉽게 발생하는 것은 당연하다.
그리고 유연함의 정도에 따라 과적합까지의 임계값은 존재하고, 임계값 내에서의
최대치를 찾는 것이 가장 좋은 모델이다.

<br>

**SVM 모델에서 중요한 HyperParameter(하이퍼파라미터)로 kernel(커널), C(규제), gamma(커널계수)가 있다.** 앞에서 kernel에 대해서는 확인해보았기 때문에 규제와 커널계수인 C, gamma를 조정하여 모델이 데이터를 어떻게 분류하는지 살펴보자.

<img width="699" alt="스크린샷 2021-05-01 오후 4 53 55" src="https://user-images.githubusercontent.com/82218035/116775485-31ae6700-aa9e-11eb-9954-b076d2cc9b6d.png">

<br>
<br>

C는 규제의 의미로 큰 값일수록 강력한 규제를 뜻하고, 규제가 강할수록 과적합이 일어남을 볼 수 있다. gamma는 커널 계수로 클수록 결정 경계와 서포트 벡터 사이의 거리가 멀어지고 과적합이 일어난다. C와 gamma는 클수록 과적합이 발생하기 때문에 **조절을 통해 최적의 값을 찾는다면 좋은 분류모델을 만들 수 있을 것이다.**

<br>

###### 결론
- SVM은 SupprotVector(서포트 벡터)와 Hyperplane(초평면)을 이용해 DecisionBoundary(결정 경계)를 정의하고 분류를 시행하는 알고리즘이다.
- SVM에서 선형분리는 p개의 변수, p차원을 p-1 차원으로 분리하는 것이고, 비선형분리는 p개의 변수, p차원을 p+1 차원으로 분리하는 것이다.
- kernel, C, gamma의 조절을 통해 과적합이 일어나지 않는 최적의 값을 찾아 우수한 성능의 모델을 만들 수 있다.


<!--

SVM에서 중요한 하이퍼 파라미터로는
kernel(kernel type) -
C(규제) - 클수록 강력한 규제, 클수록 과적합
gamma(커널 계수, 가우시안 커널 폭을 제어, 데이터 포인트 사이의 거리)
클수록 데이터 포인트 사이의 거리가 , 클수록 과적합


다시 본래의 차원에서 과적합을 초래할수있다는 것이다.

고차원 공간에서의 분류는 기존 공간에서의 분류 일반화의 오차를 상승시킨다는 것이 증명되었고,
그 오차의 상승 폭에 대한 임계값이 있다.
과적합까지의 임계값이 있다.



비선형분리

Hyperplane
p차원에서 p-1차원의 Hyperplane 생성

분류기는 고차원 특징 공간에서 선형 초평면이지만
기존 차원 공간에서는 비선형 초평면이다
초평면의 차원증가???
고차원 공간을 이용한 분류는 과적합을 초래하여
기존 공간에서 분류 일반화의 오차를 상승시킨다.
				(기존 공간에서 분류 오차를 증가시킨다.)
				(기존 공간에서의 오분류를 발생시킨다.)



선형분리
p개의 변수, p차원을 p-1 차원으로 분리
3차원을 2차원으로 분리, 2차원을 1차원으로 분리

비선형분리
p개의 변수, p차원을 p+1 차원으로 분리
2차원을 3차원으로 분리, 1차원을 2차원으로 분리

소프트벡터머신에 가장 근접한 값들만 활용해 서포트벡터를 그린다.


n개의 속성을 가진 데이터는 최소 n+1개의 서포트 벡터가 존재한다.


from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=40, centers=2, random_state=20)



option
	kernel
	C
	gamma


svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
plot_decision_function(X_train, y_train, X_test, y_test, clf)



하이퍼 파라미터별로 그래프 그려본 후
그리드 서치를 통한 최적의 하이퍼파라미터 설정


파라미터확인
class_weight_ : ndarray of shape (n_classes,)
classes_ : ndarray of shape (n_classes,)
coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
dual_coef_ : ndarray of shape (n_classes -1, n_SV)
fit_status_ : int
intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
support_ : ndarray of shape (n_SV)
support_vectors_ : ndarray of shape (n_SV, n_features)   #서포트벡터 위치 반환 scatter활용
n_support_ : ndarray of shape (n_classes,), dtype=int32  #서포트벡터 갯수


커널 활용한 데이터 선형 분리 시각화


결정경계 시각화
import mglearn

plt.figure(figsize=[10,8])
mglearn.plots.plot_2d_classification(model, X_train, cm='spring')
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
-->
