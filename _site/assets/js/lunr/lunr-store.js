var store = [{
        "title": "Logistic regression",
        "excerpt":"    머신러닝은 크게 지도학습과 비지도학습 그리고 강화학습으로 구분된다. 그중에서 지도학습의 기법인 로지스틱 회귀(LogisticRegression)에 대해 알아보고자 한다. 지도학습은 크게 분류(Classification)와 회귀(Regression)로 구분짓는데 로지스틱 회귀는 이름에 회귀(Regression)을 포함하고 있지만 대부분 분류를 목적으로 활용된다. 어떤 이유로 이름에 회귀를 포함하고 있지만 분류를 목적으로 사용하고 있는지 알아보고 로지스틱 회귀분석이란 어떤 것인지 정리를 통해 구체적으로 알아보겠다.       우선, 로지스틱 회귀를 알아보기 전에 회귀란 무엇인지 구체적인 개념을 잡고 가보자. 회귀의 사전적 의미는 ‘한 바퀴 돌아서 본디의 자리나 상태로 돌아오는 것’이라고 한다. 분석에서 회귀는 평균으로 돌아가는 것을 뜻하며, 예를 들어 부모와 자녀의 키는 선형적인 관계가 있어서 부모의 키가 크면 자녀의 키도 크겠지만 부모의 키가 아무리 커도 결국 자녀의 키는 자녀 세대의 평균으로 돌아가고 있는 것을 회귀한다고 표현한다.      로지스틱 회귀는 선형 회귀의 목표와 동일하게 종속 변수와 독립 변수의 관계를 하나의 모델로 정의하고 예측에 활용하는 것이다. 이렇듯 독립 변수의 선형 결합으로 종속 변수를 설명하고 있기에 선형 회귀 분석과 유사하지만 로지스틱 회귀는 종속 변수가 연속형이 아닌 범주형이라는 점에서 분류 기법으로 활용된다. 즉, 독립 변수의 선형적 특징을 활용한다는 점이 이름에 회귀를 포함하고 있다고 보았고, 종속 변수가 범주형이기에 분류 기법으로 활용한다고 보여진다.          위의 그래프는 선형 회귀 분석을 통한 분류는 문제가 있음을 알수있다. 첫번째 그래프를 보면 22시간 이상 공부한 사람은 시험에 통과하였고 22시간 보다 적게 공부한 사람은 탈락하였다. 이 결과를 바탕으로 선형적인 모델을 만들면 경계선을 기준으로 22시간일때의 합격률은 0.6~0.7 정도가 나온다. 만약 여기서 새로운 데이터로 많은 시간을 공부해서 합격한 사람이 추가된다고 가정해보자. 결과값인  종속 변수가 최대값이 1이기 때문에 이때 만들어진 모델은 그 전 모델보다 기울기가 감소하게된다. 이때 문제가 생기는데 기울기가 감소하여 합격의 기준이 높아짐으로 그전에 합격이라고 예측했던 값들이 불합격으로 잘못 판단하게 되는 경우가 발생하게된다. 이러한 문제점을 해결하기 위해 발견한 함수가 시그모이드 함수로 로지스틱 회귀에 사용된다.          시그모이드 함수는 무한한 범위의 x값이 들어오면 y값의 범위는 한상 [0, 1] 사이의 값으로 반환한다. 이러한 함수의 특징때문에 높은 값의 x가 들어와도 합격률의 변동이 없어 선형 회귀를 통한 분류의 문제를 해결가능하도록 만들어준다. 반대로 선형 회귀는 y값이 [0, 1]사이를 벗어나는 값이 나오기 때문에 분류로 활용시에는 예측의 정확도만 떨어뜨리게 된다. 그래서 로지스틱 회귀에서 시그모이드 함수를 사용하고 시그모이드 함수는 확률값을 반환해 0에 가까우면 0으로 1에 가까우면 1로 분류를 시행한다.          결론     선형회귀로 분류를 시행 시 발생하는 문제를 해결하기 위해 로지스틱 회귀를 사용한다.   로지스틱 회귀는 시그모이드 함수를 사용한 분류 기법이다.   시그모이드 함수는 확률값을 반환하고 확률에 따라 [0, 1]로 분류를 시행한다.  ","categories": ["Machine learning"],
        "tags": ["Machine learning","Model"],
        "url": "/machine%20learning/Logistic-regression/",
        "teaser": null
      },{
        "title": "Support vector machine",
        "excerpt":"    머신러닝의 지도학습 기법으로 주로 분류에 자주 사용되는 SVM에 대하여 알아보고자 한다. SupportVectorMachine의 줄일말로 SupprotVector(서포트 벡터)와 Hyperplane(초평면)을 이용해 DecisionBoundary(결정 경계)를 정의하고 분류를 시행하는 알고리즘이다.      DecisionBoundry  두 개의 계층을 가지고 있는 통계적인 분류 문제에서, 결정 경계는 기본 벡터공간을 각 클래스에 대하여 하나씩 두 개의 집합으로 나누는 초표면이다. 분류기는 결정 경계의 한쪽에 있는 모든 점을 한 클래스, 다른 한쪽에 있는 모든 점을 다른 클래스에 속하는 것으로 분류한다.    SVM은 최적의 결정 경계를 찾는 것이 목적인데 여기서 최적의 결정 경계를 찾기 위해서는 Margin(마진)의 최대점을 찾아야한다. 마진은 결정 경계와 서포트 벡터 사이의 거리로 큰 값을 가질수록 분류 경계가 커짐을 뜻하기 때문에 좋은 분류 성능을 보여줄것이다.         SVM 구성요소  Hyperplane : 초평면으로 고차원의 결정 경계를 뜻하기도 한다.  SupportVector : 결정 경계와 가까이 있는 데이터 포인터  margin : 결정 경계와 서포트 벡터 사이의 거리        SVM에서 결정 경계는 p차원의 데이터라면 p-1차원의 결정 경계가 만들어진다. 예를 들어, 3차원의 데이터는 2차원의 평면으로 경계가 만들어지고, 2차원의 데이터는 1차원의 벡터(선)으로 경계를 만든다. 이러한 결정 경계의 특징 때문에 초기에 결정 경계를 구하는 알고리즘은 선형 분류만 가능했었고, 비선형 분류가 불가능했었다. 하지만 데이터의 분포에 커널 트릭을 적용시켜 분포를 고차원으로 변형함으로써 비선형 분류도 가능하도록 만들었다.      커널트릭  저차원 공간을 고차원으로 매핑해주는 작업, 선형 분리가 불가능한 1차원 데이터를 2차원의 공간으로 매핑하여 선형분리가 가능하도록 한다.       위의 그림은 커널트릭을 활용해 선형분리가 불가능한 1차원의 데이터에 2차원의 데이터를 추가함으로써 선형분리가 가능하도록 하는것이다. 그렇다면 커널트릭을 적용해 저차원의 데이터를 고차원으로 매핑하여 시행하는 분류의 성능은 선형분류보다 정확한 성능을 보여주는 것인가? 하는 의문이 생긴다. 2차원의 데이터를 1차원의 벡터로 선형 분류를 시행하는 방법과 2차원의 데이터를 3차원으로 변형시켜 2차원의 초평면으로 비선형 분류를 시행하는 방법중 어떤 것이 더 높은 정확도를 보여주는지 파악해보았다.                 분류를 시행해 본 결과, 결론적으로 고차원 공간으로 변형시킨 방법이 데이터를 더 잘 분류했다. 선형분리가 가능한 데이터에서 커널트릭을 활용한 분류가 마진이 더 컸으며, 비선형 분류를 해야하는 데이터도 선형분리가 불가능한 단점을 유연한 결정 경계로 데이터를 분류해 커널트릭을 활용한 분류가 항상 더 좋은 분류라 생각할수도있다. 그러나 커넡트릭을 활용한 분류는 한가지 큰 문제점이 있다. 머신러닝에서 자주 발생하는 문제점인 OverFitting(과적합)을 초래한다는 것인데, 데이터 차원을 증가시켜 고차원에서 학습시킨 모델은 다시 본래의 차원에서 과적합이 발생하기 때문이다.      머신러닝에서 모델이 유연할수록 과적합이 쉽게 발생하는 것은 당연하다. 그리고 유연함의 정도에 따라 과적합까지의 임계값은 존재하고, 임계값 내에서의 최대치를 찾는 것이 가장 좋은 모델이다.        SVM 모델에서 중요한 HyperParameter(하이퍼파라미터)로 kernel(커널), C(규제), gamma(커널계수)가 있다. 앞에서 kernel에 대해서는 확인해보았기 때문에 규제와 커널계수인 C, gamma를 조정하여 모델이 데이터를 어떻게 분류하는지 살펴보자.            C는 규제의 의미로 큰 값일수록 강력한 규제를 뜻하고, 규제가 강할수록 과적합이 일어남을 볼 수 있다. gamma는 커널 계수로 클수록 결정 경계와 서포트 벡터 사이의 거리가 멀어지고 과적합이 일어난다. C와 gamma는 클수록 과적합이 발생하기 때문에 조절을 통해 최적의 값을 찾는다면 좋은 분류모델을 만들 수 있을 것이다.       결론     SVM은 SupprotVector(서포트 벡터)와 Hyperplane(초평면)을 이용해 DecisionBoundary(결정 경계)를 정의하고 분류를 시행하는 알고리즘이다.   SVM에서 선형분리는 p개의 변수, p차원을 p-1 차원으로 분리하는 것이고, 비선형분리는 p개의 변수, p차원을 p+1 차원으로 분리하는 것이다.   kernel, C, gamma의 조절을 통해 과적합이 일어나지 않는 최적의 값을 찾아 우수한 성능의 모델을 만들 수 있다.    ","categories": ["Machine learning"],
        "tags": ["Machine learning","Model"],
        "url": "/machine%20learning/Support-vector-machine/",
        "teaser": null
      },{
        "title": "Decision tree",
        "excerpt":"    머신러닝의 기법으로 결정 트리(Decision Tree)를 사용하여 예측 모델을 만드는 결정 트리 학습법(Decision Tree Learning)이 있다. 결정 트리는 의사 결정 규칙과 그에 따른 결과를 보기 쉽게 트리 구조로 나타내어 결과에 대한 원인 파악이 가능하다는 장점이 있다. 즉, 결정 트리의 규칙들이 어떤 기준으로 설정되었는지 살펴보고 분류 결과 확인이 같이 가능하기 때문에 분류 기준을 파악하기에 유용할것이다.          결정 트리의 종류로 분류트리(Classification Tree)와 회귀트리(Regression Tree)가 있는데, 두 방식 모두 특정한 기준을 정하고 그 순간의 최적의 결정을 하는 greedy 방식을 따른다. 분류트리는 Target변수(Y)가 범주형 변수인 경우 사용하는데 정보이득(InformationGain)이 최대가 되는 특성으로 데이터를 나누고 정보이득은 불순도(Impurity)를 통해 계산한다. 여기서 불순도는 타겟변수의 분류가 잘 되어 있을수록 0에 가까운 값이 나온다.         불순도 불순도를 쉽게 표현하자면 방의 어질러진 정도를 나타낸다고 표현한다. 불순도는 높게 나올수록 방이 어지럽다는 뜻이고, 분류가 이뤄지지 않았다고 이해할수있으며, 지표로는 지니계수(Gini Index)와, 엔트로피(Entropy)가 있다.    분류에서 정보이득이 최대가 된다면 부모 노드의 불순도와 자식 노드의 불순도의 차이가 최대임을 뜻하는데, 노드간의 불순도의 차이가 최대가 되어야 분류가 잘 이뤄졌음을 뜻하기 때문이다.      정보이득과 불순도  정보이득은 분류의 기준을 찾기 위한 노드 간의 불순도 차이이고,   불순도는 분류의 기준을 설정하기 위한 노드 내부의 수치이다.    회귀트리는 Target변수(Y)가 연속형 변수인 경우 사용하며, 잔차 제곱합(RSS)을 최소화 시킬 수 있는 방향으로 분류 기준을 만드는 방식이다. 분류 트리와 크게 다르지 않으며 예측값이 범주형이 아니기 때문에 예측한 구간내의 평균값으로 예측값을 계산한다. 회귀트리를 회귀분석과 혼동 할 수 있는데, 회귀트리는 회귀 함수를 기반으로 하지 않고 트리 기반으로 예측을 진행하기 때문에 트리구조로 데이터를 구간화 시킨 후 나뉜 구간별로 회귀분석을 시행하는 방식이라 생각한다.          위의 그림은 동일한 데이터를 각각 결정 트리와 서포트 벡터 머신(SVM)으로 분류를 진행한 결과이다. 두 분류 모델의 가장 큰 차이점을 결정 경계(Decision Boundary)로 확인 가능하다. 결정 트리는 재귀적 분기 방식으로 한번에 하나의 특성(Feature)만 고려하여 분류를 진행하기 때문에 결정 경계가 모두 수직을 이루고 있으며, 서포트 벡터 머신은 모든 특성을 고려하여 초평면을 찾고 분류를 진행하기 때문에 결정 경계가 곡선형태로 나타난다. 이러한 결정 트리의 분류는 일차원 방식이라 과적합이 발생하지 않을 것이라 생각 할 수도 있지만, 일차원 분류가 여러 번 시행되어 분류 횟수의 마지노선을 정해주지 않으면 과적합이 쉽게 발생하는 단점이 있다. (그림의 분류 경계만 보더라도 하나의 값을 분류하기 위해 모델이 과적합된 모습을 볼 수 있다) 그렇다면 결정 트리는 어떤 데이터의 분류에 적용하는 것이 효과적일까?  결정 트리는 독립변수와 종속변수 간의 관계가 비선형적인 관계를 나타낼때 좋은 모델을 만들 것이다. 독립변수의 값이 커질수록 결정 경계가 같이 변한다면 선형 형태의 분류가 적합하겠지만, 하나의 특성이 여러 구간으로 분리 된다면 단순한 비선형 형태의 분류인 결정 트리가 효과적일 것이다.   결론     머신러닝의 기법인 결정 트리(Decision Tree)는 분리 규칙을 트리 구조로 만들어 모델 이해가 쉽다.   분류트리는 정보이득(InformationGain)으로 데이터를 나누고 정보이득은 불순도(Impurity)를 통해 계산한다.   결정 트리는 비선형적인 관계를 분류할때 효과적인 모델을 만든다.    ","categories": ["Machine learning"],
        "tags": ["Machine learning","Model"],
        "url": "/machine%20learning/Decision-tree/",
        "teaser": null
      },{
        "title": "K-nearest neighbor",
        "excerpt":"    KNN(K-최근접 이웃, K-Nearest Neighbor)은 직관적이고 간단한 방법에 비해 좋은 성능을 보여주어 종종 사용되는 머신러닝 알고리즘이다. 대부분의 머신러닝 알고리즘은 훈련데이터를 통해 모델을 생성하는 방식이라면, KNN은 하나하나의 데이터 값을 통해 학습을 시행하고 따로 모델을 생성하지 않는 비모수 방식이라는 특징이 있다. KNN 알고리즘의 작동방식에 대해 알아보기전 KNN의 특징인 비모수 방식에 대해 짚고 넘어가보자.          모수 방식과 비모수 방식의 가장 큰 차이는 확률분포의 개념을 활용하느냐의 차이이다. 예를 들어, 모수 방식인 선형 회귀는 두 변수 사이의 관계를 표현하는 것인데 이때 생기는 오차는 정규분포, 즉 확률분포로 정규분포를 따르기 때문에 확률분포의 개념을 활용한다. 반대로 비모수 방식은 두 변수 사이의 관계를 표현할때 생기는 오차는 정규분포를 따르지 않는다고 할 수 있다. 즉, 비모수 방식은 오차가 특정 분포를 이루고 있는게 아니라 의미없이 오차들이 골고루 분포하고 있는 것이기 때문에 확률분포의 개념을 활용하지 않는 것이다.     모수 방식과 비모수 방식은 선형, 비선형과 관련있다.  모수 방식은 변수가 선형 관계일때 활용하고, 비모수 방식은 변수가 비선형 관계일때 활용하는 것이 적합한 방식이다.           KNN은 대표적인 비모수 방식으로 데이터 간의 거리를 활용해 알고리즘이 작동한다. 만약 임의로 지정한 k=3이라면 (일반적으로 k는 홀수를 사용, 짝수의 k는 동점을 초래하기 때문) 새로 들어온 데이터에서 거리가 가까운 3개의 데이터를 통해 분류를 시행한다. 모델을 별도로 생성하지 않는 알고리즘의 특성상 결정 경계(Decision Boundary)가 존재하지 않으며, 새로운 데이터가 주어지면 학습을 통해 새로 모델을 생성할 필요없이 새로운 데이터만 추가하여 분류한다. 그리고 거리를 기반으로 분류하여 이상치, 노이즈에 크게 영향을 받지 않는다는 장점도 있다.          KNN은 유클리드 거리와 마할라노비스 거리 등 다양한 계산방식을 통해 거리 측정이 가능하다. 유클리드 거리는 흔히 알고있는 피타고라스 정리에 기초해서 거리를 구하는 방식이고, 마할라노비스 거리는 각 특성의 분포를 파악하고 실제로 같은 거리라도 특성마다 다르게 측정하는 방식이다. 거리를 측정하기 위한 특성으로 x, y가 있다고 가정해보자. 특성 x의 분산이 특성 y보다 크다면 x의 분포가 더욱 퍼져있을것이고, 이때 분산을 거리의 기준으로 보면 특성 x에서의 거리는 동일한 거리라도 특성 y에서 보다 더 짧다고 볼 수 있다. 이처럼 특성마다 분포가 다르기 때문에 거리의 스케일이 다를것이고 이를 동일하게 하기 위해 마할라노비스 거리를 활용한다.      차원의 저주 : 유클리드 거리가 고차원에서 도움이 되지 않음, 차원의 증가로 인한 데이터 간 거리 증가, PCA활용        결론     KNN은 특정한 모델을 생성하지 않고 데이터를 통해 학습을 시행하는 비모수 방식이다.   거리를 기반 알고리즘이기 때문에 노이즈에 크게 영향을 받지 않는다.   거리 측정을 위한 방법으로 크게 유클리드 거리와 마할라노비스 거리가 있다.  ","categories": ["Machine learning"],
        "tags": ["Machine learning","Model"],
        "url": "/machine%20learning/K-nearest-neighbor/",
        "teaser": null
      },{
        "title": "Overfitting",
        "excerpt":"  머신러닝에서 훈련 데이터로 모델을 생성하고 테스트 데이터에 적용해보면 훈련 데이터에서 좋았던 성능이 테스트 데이터에서 현저히 떨어지는 경우가 자주 발생한다. 테스트 데이터에 문제가 있거나 훈련 데이터를 통해 생성한 모델에 문제가 있어 성능에 차이가 생기는 것이다. 테스트 데이터에 문제가 있는 경우는 학습시 발생하는 것이 아니라 모델링과 관계없는 순수한 데이터의 문제이지만, 훈련 데이터로 생성한 모델은 모델링과 직접적인 관련이 있다. 훈련 데이터를 과하게 학습하여 모델이 훈련 데이터에 최적화되어 테스트 데이터에서의 성능이 떨어지는 것인데 이를 과적합(Overfitting)이라 한다. 과적합은 머신러닝 과정에서 발생하는 대표적인 문제인데, 과적합을 발생시키지 않으면서 가능한 최대의 학습을 시키는 것이 머신러닝의 궁극적 목표이다.        회귀(Regression)에서 과적합을 보면, 회귀식이 모든 데이터에 피팅되어 구불해진 회귀식의 모습을 볼 수 있다. 만약 다른 이상치가 더 있다면 회귀식은 더욱 휘어진 형태로 과적합 될것이다. 그리고 분류(Classification)에서는 결정경계(Decision Boundary)가 오분류를 발생시키지 않기 위해 모든 데이터에 피팅 된 모습이다. 회귀에서는 회귀식이 유연한 형태였다면, 분류에서는 결정경계가 유연한 형태로 과적합이 발생한다.          과적합은 분산(Variance), 편향(Bias)과도 연관성이 있다. 위의 이미지를 보면 분산이 크고 편향이 작을 때 전형적인 과적합의 형태를 띄고있는데, 분산이 크면 예측 분포가 큰 복잡한 모델을 만들어 과적합이 발생하고 편향이 크면 예측이 계속해서 틀리는 단순한 모델을 만들어 과소적합이 발생한다.     분산과 편향은 트레이드 오프(Trade-off) 관계로 분산이 커지면 편향이 작아지고 분산이 작아지면 편향이 커지게 된다.        그렇다면 어떤 방법으로 과적합을 해결하고 모델의 성능을 높일 수 있을까? 과적합을 해결하는 가장 간단한 방법은 훈련 데이터를 추가하는 것이다. 새로운 훈련 데이터가 생기면 기존의 데이터들의 영향력이 조금 줄어들어 과적합 해결이 가능할것이다. 즉, 모델의 전체 영향력을 기존 데이터가 나눠 갖고 새로운 데이터의 추가로 전체적인 영향력이 줄어들어 과적합이 사라지는 것이다. 다음의 방법으로 데이터 차원을 감소시키는 방법이 있다. SVM 모델을 예로, 커널 트릭을 활용해 데이터의 차원을 증가시켜 2차원에서 분류 불가능한 데이터를 3차원에서 분류할 수 있는데 이때 3차원에서 생성한 모델을 2차원으로 변형시키면 결정 경계가 과적합된 경우와 비슷한 형태를 보이게 된다. 반대의 경우를 생각해보면 과적합 된 모델의 데이터 차원을 감소시키면 결정 경계가 완만해져 과적합 해결이 가능하다. 마지막 방법으로 규제(Regularization)를 통해 모델 복잡도를 감소시켜 과적합이 일어나지 않게 하는 것으로, 위 방법들은 모델을 훈련하기 전에 시행하는 방식이라면 규제는 학습을 진행하면서 과적합을 제어하는 방식이다. 규제는 크게 계수의 절대값에 비례하여 패널티를 주는 Lasso(L1)과 계수의 제곱에 비례하여 패널티를 주는 Ridge(L2) 방식이 있다. 두 방식 모두 비용 함수(Cost Function)에 패널티를 추가해 기존의 비용(Cost)이 증가하여 학습을 최소화하게 된다.          수식을 보면, 기존의 Cost Function에 각각의 패널티가 더해진 것을 볼 수 있다. 이때 패널티는 가중치(w)에 학습률(Learning Rate)를 곱한 값으로 Cost Function에 포함되어 학습을 하면서 가중치 스스로 값이 커지는 것을 제어한다. 가중치가 큰 값을 갖지 못하기 때문에 회귀식에서 과적합으로 휘어진 식이 과적합 되지 못하고 직선형태로 변하게 되는 것이다. 모델 생성 시 타원의 중앙에서 Cost Function 최소가 되지만, 규제를 통해 타원의 중앙에 가장 근접한 가중치를 찾게 된다. 이때의 Cost Function은 타원의 중앙값이 아니라 최소가 되는 지점이 아니라 생각할수도 있으나 실제 데이터에서는 모델에 규제가 적용되어 더 좋은 효과를 보여주는 것이다.       결론     과적합(Overfitting)은 훈련 데이터를 과하게 학습하여 모델이 훈련 데이터에 최적화 된 것을 말한다.   과적합은 분산(Variance), 편향(Bias)과도 관련있으며, 분산이 크면 예측 분포가 큰 복잡한 모델을 만들어 과적합이 발생한다.   규제(Regularization)는 비용 함수(Cost Function)에 패널티를 추가해 과적합을 방지한다.    ","categories": ["Machine learning"],
        "tags": ["Machine learning"],
        "url": "/machine%20learning/Overfitting/",
        "teaser": null
      },{
        "title": "Unsupervised learning",
        "excerpt":"    머신러닝은 크게 지도학습(Supervised Learning), 비지도학습(Unsupervised Learning), 강화학습(Reinforcement Learning)으로 나뉜다. 지도학습과 강화학습은 입력(Input)에 대한 결과(Output)를 주고 학습하는 방식이지만 비지도학습은 입력에 대한 결과가 없이 입력만으로 학습을 진행한다. 비지도학습을 시험에 빗대면 문제에 정답이 없는 논술형 시험과 비슷하다고 생각하는데 정답이 없기 때문에 정답을 판단하는 기준이 모호하고 결과는 해석하기 나름이라는 공통점이 있다.          비지도학습의 가장 대표적인 기법에는 군집화(Clustering)가 있다. 군집화는 논술형 시험과 유사하다. 논술형 시험은 정해진 결론이 없고 유사한 결론이 있기 때문에 비슷한 결론끼리 묶음 처리가 가능할 것이다. 이처럼 정해진 답은 없지만 유사한 답을 묶어 처리하는 방식이 군집화의 원리이다. 그리고 군집화를 시행하고 각각의 군집에 속한 사람들의 특징을 파악하면 군집 내부에서 새로운 특징 도출이 가능한데 만약 답안으로 긍정적인 결론을 지은 사람들을 모아놓고 다른 특징을 파악해보면 대부분 사람들은 밝은 성격을 갖고 있다는 새로운 특징을 찾을 수 있을 것이다.      군집화는 지도학습의 분류(Classification)와 공통점이 있다. 두 방식 모두 출력값이 분류 가능한 이산형 값으로 나와 데이터의 분류가 가능하다는 것이다. 그렇다면 비지도학습에서 지도학습의 회귀(Regression) 문제와 같이 연속형 값으로 예측은 불가능한가?           비지도학습은 차원 축소(Dimensionality Reduction)로 데이터의 주요 특징을 추출하고 변환 가능하다. 차원 축소 기법인 주성분 분석(PCA, Principal Component Analysis)을 실시하여 차원을 축소하고 복원하는 과정에서 데이터의 주요 특징만 남는 원리를 활용해 비지도 학습의 이상치 탐지(Unsupervised Anomaly Detection)가 가능하다.      이때 입력 데이터의 특징을 추출하기 위해 오토인코더(Autoencoder)를 사용하는데 정상 데이터가 오토인코더를 거치면 정상 데이터가 출력되어 Input과 Output에 차이가 없는 반면, 비정상 데이터가 오토인코더를 거치면 정상 데이터가 출력되어 Input과 Output에 차이가 발생해 비정상 데이터가 구분 가능해진다.       인공지능의 목적은 비지도학습이 가능한 인공지능을 만드는 것이라 생각한다. 기계가 인간처럼 스스로 판단하고 행동하도록 만들기위해 머신러닝의 지도학습은 정답을 보면서 학습하지만, 비지도학습은 정답을 모르고 계속해서 학습을 진행하기 때문에 처음 맞닥뜨린 상황이더라도 스스로 판단을 한다. 그리고 기계가 처음 접하는 상황에 자신만의 판단을 만든다면, 사람처럼 성격과 인격을 갖춘 인공지능이 구현가능할 것이다. 이처럼 인공지능의 최종 목표는 비지도학습을 통해서 완성되고 구현될것이라 본다.       결론     비지도학습은 입력에 대한 결과가 없이 입력만으로 학습을 진행한다.   비지도학습은 차원 축소로 데이터의 주요 특징을 추출하고 변환 가능하다.   인공지능의 완전한 모습은 비지도학습으로 구현될것이다.  ","categories": ["Machine learning"],
        "tags": ["Machine learning"],
        "url": "/machine%20learning/Unsupervised-learning/",
        "teaser": null
      },{
        "title": "Ensemble",
        "excerpt":"    앙상블(Ensemble)은 프랑스어로 단어 자체의 의미가 조화, 통일을 뜻한다. 머신러닝에서 앙상블은 여러 모델을 조화시켜 하나의 새로운 모델을 만드는 것을 의미하며, 여러 모델의 조합으로 만들어졌기 때문에 일반적으로 더욱 좋은 성능을 보여준다. 앙상블의 기본 원리를 알면 앙상블을 왜 사용하는지 이해가능하다. 데이터의 크기가 작다면, 한가지 기법만 사용해 하나의 모델만 생성하여 적용해도 모델이 좋은 성능을 보여줄것이다. 그러나 현재 대부분의 데이터는 크기가 크고 다양해서 하나의 모델만만으로 데이터를 충분히 설명하기 어렵다. 이러한 문제점을 해결하기 위해 모델 각각의 특징을 결합한 새로운 모델을 생성하는 앙상블 기법을 사용하는 것이다.      실생활을 예로 축구팀에 빗대어보자. 팀에 공격수만 있다면, 공격은 잘되지만 수비수가 없어 골을 잘 먹혀 대회에서 좋은 성적을 거두지 못할것이다. 공격수와 수비수가 둘다 있는 공격과 수비에 밸런스를 맞춘 조화로운 팀을 만든다면 한가지에 특출난 팀보다 우수한 성적을 거둘것이다. 축구선수 한명을 모델로 보고 팀의 성적을 모델의 성능이라 가정해보면, 축구는 혼자 팀을 하는 것보다 선수가 많을수록, 공격수만 있는팀 보다 다양한 포지션의 선수가 있는 팀일수록 좋은 성적을 거둘것이다. 앙상블도 마찬가지로 모델은 많을수록, 다양한 모델이 있을수록 좋은 성능을 보여주는 것이다.     앙상블에는 보팅(Voting), 배깅(Bagging), 부스팅(Boosting) 방식이 있다. 보팅과 배깅은 완성된 모델을 결합하는 방식이라면, 부스팅은 모델을 수정해가면서 계속해서 새로운 모델을 만드는 방식이다. 보팅부터 살펴보면, 보팅은 크게 하드 보팅(Hard Voting)과 소프트 보팅(Soft Voting)으로 나뉜다. 두 방법 모두 모델마다 투표권을 갖고 결과에 대한 투표권을 행사하여 많은 득표를 받은 값을 최종 값으로 선정하는 투표 방식의 앙상블 기법으로 하드 보팅은 각 모델의 결과값을 종합하여 많이 나온 값을 최종 값으로 정하는 방식이고 소프트 보팅은 모델의 결과값이 나올 각 확률을 결합하여 최종 값을 정하는 방식이다.        하드 보팅은 말그대로 투표권을 분산 없이 하나의 값에 부여하여 비슷한 확률 값을 갖는 결과는 무시해버리는 경우가 생겨 소프트 보팅보다 유연성 측면에서 떨어진다. 앙상블 원리 자체가 여러 모델을 결합하고자 하는 것인데, 하드 보팅은 모델을 결합한다기보다 결과값을 취합하는 방식같다는 개인적인 느낌이 들어 앙상블의 취지에 부합하지 않다는 생각이 든다. 그래서 일반적으로 하드 보팅보다는 소프트 보팅을 선호하는 경향이 있고 성능적인 측면에서도 소프트 보팅이 더 좋은 성능을 보여주기도 한다.        배깅은 보팅과 비슷한 방식으로 둘은 자주 비교된다. 보팅은 서로 다른 알고리즘을 활용한 모델을 결합하는 것이고, 배깅은 같은 알고리즘을 활용한 모델에 데이터를 다르게 적용하여 생성된 모델을 결합하는 방식이다. 배깅과 보팅은 같은 알고리즘을 활용하느냐, 다른 데이터를 활용하느냐가 두 방식의 차이점이라 생각한다.  마지막으로 부스팅은 앞의 보팅, 배깅과 다르게 모델을 결합한다기보다 계속적으로 모델을 업데이트하는 방식이다. 생성한 모델을 검증 데이터에 적용해 오분류 데이터를 찾고 오분류 데이터에 가중치를 부여해 오분류 데이터를 잘 분류하도록 다음 모델을 업데이트 한다. 이런 과정을 여러번 반복하다보면 최종적으로 생성한 모델은 에러와 오차가 줄어드는 것이다.   앙상블은 현재시점에서 정형데이터에 좋은 성능을 보여주고있다. 데이터가 이미지나, 텍스트 등의 복잡한 데이터가 아니라면 머신러닝의 단순한 알고리즘을 앙상블시키면 충분한 성능을 보여준다는 것이다. 아직까지 관계형데이터베이스(RDB)의 정형데이터를 자주 접하고 있기 때문에 앙상블을 잘 활용한다면 좋은 모델을 만들 수 있다.     딥러닝 모델도 앙상블을 통해 성능을 높이는 방식도 가능하다.        결론     앙상블은 여러 모델을 조화시켜 하나의 새로운 모델을 만드는 것을 의미한다.   모델은 많을수록, 다양한 모델이 있을수록 좋은 성능을 보여준다.   앙상블 기법은 모델 각각의 특징을 결합한 새로운 모델을 생성한다.  ","categories": ["Machine learning"],
        "tags": ["Machine learning"],
        "url": "/machine%20learning/Ensemble/",
        "teaser": null
      },{
        "title": "2023년 나의 다짐",
        "excerpt":"시작하며   해가 바뀐 지도 벌써 한 달이 지났다. 매년 12월 31일 내년은 더 성장한 사람이 되겠다고 스스로 다짐한다. 일 년간의 단기적인 목표를 세우고 실천했지만, 작년은 그렇지 못했다. 처음 취업한 회사에 집중하겠다는 막연한 계획만으로 보낸 작년을 돌아보니 내게 정량적으로 남은 것이 없었다. 이번에도 작년과 같이 보내지 않기 위해 다시 일 년간의 목표를 세워보려 한다.   나의 다짐(목표)  목표을 세우고 세부적인 실천 계획을 작성해보자❗️   일 년간의 목표  무엇을 최종 목표로 이번 연도를 계획해야 할지 고민했다. 재작년까지는 취업을 최종 목표로 실천 계획을 세웠다면, 취업을 완료한 지금 그다음 목표에 대해 생각해 본 적이 없었다. 그렇기에 너무 멀리 보지 않는 선에서 다시 단기적인 최종 목표를 세워보고 이에 따라 이번 연도를 계획해보자.   우선, 단기적인 최종 목표는 지금보다 한 단계 높은 회사로 이직하는 것이다. 스타트업에 들어가면서 내가 성장하는 속도에 맞춰 회사도 같이 성장하기를 기대했으나, 현실적으로 회사 성장이 나의 기대만큼 이뤄지지 못해 아쉬웠다. 지금은 온전히 나를 위해 회사와 같이 성장하는 것이 아닌 성장 가능한 곳으로 나 스스로 움직여야 한다고 판단했다.   최종 목표를 위한 실행 목표를 정리해보자. 한 단계 높은 회사로 이직하기 위해 내가 어떤 것들을 실행해야 할까?  첫 번째. 개인 역량 및 업무능력을 향상하자  두 번째. 모든 활동을 문서화하고 이를 활용해 나만의 스토리를 만들자  위 실행 목표를 이행함으로 최종 목표를 이루기에 충분한 나를 만들 것이다.   목표 및 실천 계획  앞서 정한 목표에 따른 각각의 세부적인 실천 계획을 작성해보자. (정량적인 목표 설정)     한 단계 높은 회사로 이직            개인 역량 및 업무능력 향상                    다양한 도메인 또는 데이터 분석 경험 쌓기.           직무 관련 서적 최소 6권(2달에 1권) 읽기. 감상평도 기록           영어 공부. 책을 읽거나 회화 학원 등록                       나만의 스토리 생성                    최소 1달 1회 포스팅을 목표로 블로그, 깃허브 운영.           kaggle 또는 데이콘 참가. 진행 과정은 기록하고 최소 2개는 참가                           마치며  이번 연도 나의 다짐을 목표에 기반해 작성해봤다. 최종 목표를 세우고 다음 실행 목표, 마지막으로 정량적인 목표까지 작성하면서 어느 정도 이번 연도 나의 다짐과 목표가 구체화 된 것 같다. 최종 목표를 달성하기 위한 2023년을 보내고 연말 회고록에서 최종 목표의 달성 여부와 달성률을 통해 한 해를 돌아볼 때, 후회 없는 한 해가 됐으면 좋겠다. 지나간 날은 돌아오지 않음을 명심하고 이번 연도를 성실하게 보내자.  ","categories": ["Daily life"],
        "tags": ["Daily life","나의 다짐"],
        "url": "/daily%20life/2023%EB%85%84-%EB%82%98%EC%9D%98-%EB%8B%A4%EC%A7%90/",
        "teaser": null
      },{
        "title": "Augmentation of Time series data",
        "excerpt":"시작하며   시계열 데이터는 일정한 시간 간격으로 발생하는 시간에 따라 정렬된 데이터를 의미하며, 오늘날 제조, 금융, 의료 등 다양한 분야에서 발생하고 있습니다. 현업에서 머신러닝(회귀,분류)을 위한 시계열 데이터는 다른 데이터(이미지, 텍스트 등)와 동일한 목적으로 활용됩니다. 예를 들면, 주식 가격 예측, 운동 동장 분류 등 시계열 데이터로 회귀, 분류 모델을 개발 및 적용하는 사례가 있습니다. 다른 데이터(이미지, 텍스트)는 웹상에 공유된 데이터도 많고 바로 서비스 적용 가능한 수준의 모델도 많아 모델 개발에 어려움이 없지만, 시계열 데이터는 공유된 데이터도 별로 없을 뿐더러 시계열 분류, 예측 목적의 모델도 찾아보기 어렵습니다. 앞서 언급한 문제에서 데이터 부족 문제를 해결하기 위한 방법 중 하나인 데이터 증강(Data augmentation)에 대해 알아보고자 합니다.   데이터 증강(Data augmentation)  데이터 증강이란?  데이터 증강이란 데이터 양을 늘리기 위한 기법 중 하나로, 보유한 데이터에 여러 변환을 적용하여 데이터 수를 증가시킵니다. Computer vision 분야에서 필수로 사용되며 아래 이미지와 같이 원본 이미지를 다양한 방식으로 변환해 원본 데이터에서 학습하지 못했던 부분을 모델이 학습할 수 있도록 합니다.   데이터 증강의 궁극적인 목적은 한정적인 훈련 데이터를 증강시킴으로 모델 성능을 높이고 과적합 문제를 해결하고자 합니다.   시계열 데이터 증강의 중요성   데이터 증강 기법   Time Shifting     Forward and backward shift   Random shift   Scaling     Uniform scaling   Non-uniform scaling   Noise Addition     White noise addition   Colored noise addition   Time warping   Interpolation   데이터 증강 적용   Improving performance  Small datasets  Imbalanced datasets  Anomaly detection  Prediction and forecasting   데이터 증강 한계 및 과제     Data quality issues 여전히 햄스터인가?? -&gt; 증강한 데이터가 원본 데이터의 특성을 유지하고 있는가??   Augmentation method selection   Computational complexity   마치며      Summary of key points   Future research directions   Augmetation of time series data에 대한 개인의견   ","categories": ["Machine learning"],
        "tags": ["Machine learning","Time series"],
        "url": "/machine%20learning/Augmentation_timeseries/",
        "teaser": null
      }]
