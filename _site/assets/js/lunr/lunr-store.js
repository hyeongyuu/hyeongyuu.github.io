var store = [{
        "title": "Overfitting",
        "excerpt":"    머신러닝에서 훈련 데이터로 모델을 생성하고 테스트 데이터에 적용해보면 훈련 데이터에서 좋았던 성능이 테스트 데이터에서 현저히 떨어지는 경우가 자주 발생한다. 테스트 데이터에 문제가 있거나 훈련 데이터를 통해 생성한 모델에 문제가 있어 성능에 차이가 생기는 것이다. 테스트 데이터에 문제가 있는 경우는 학습시 발생하는 것이 아니라 모델링과 관계없는 순수한 데이터의 문제이지만, 훈련 데이터로 생성한 모델은 모델링과 직접적인 관련이 있다. 훈련 데이터를 과하게 학습하여 모델이 훈련 데이터에 최적화되어 테스트 데이터에서의 성능이 떨어지는 것인데 이를 과적합(Overfitting)이라 한다. 과적합은 머신러닝 과정에서 발생하는 대표적인 문제인데, 과적합을 발생시키지 않으면서 가능한 최대의 학습을 시키는 것이 머신러닝의 궁극적 목표이다.          회귀(Regression)에서 과적합을 보면, 회귀식이 모든 데이터에 피팅되어 구불해진 회귀식의 모습을 볼 수 있다. 만약 다른 이상치가 더 있다면 회귀식은 더욱 휘어진 형태로 과적합 될것이다. 그리고 분류(Classification)에서는 결정경계(Decision Boundary)가 오분류를 발생시키지 않기 위해 모든 데이터에 피팅 된 모습이다. 회귀에서는 회귀식이 유연한 형태였다면, 분류에서는 결정경계가 유연한 형태로 과적합이 발생한다.          과적합은 분산(Variance), 편향(Bias)과도 연관성이 있다. 위의 이미지를 보면 분산이 크고 편향이 작을 때 전형적인 과적합의 형태를 띄고있는데, 분산이 크면 예측 분포가 큰 복잡한 모델을 만들어 과적합이 발생하고 편향이 크면 예측이 계속해서 틀리는 단순한 모델을 만들어 과소적합이 발생한다.     분산과 편향은 트레이드 오프(Trade-off) 관계로 분산이 커지면 편향이 작아지고 분산이 작아지면 편향이 커지게 된다.        그렇다면 어떤 방법으로 과적합을 해결하고 모델의 성능을 높일 수 있을까? 과적합을 해결하는 가장 간단한 방법은 훈련 데이터를 추가하는 것이다. 새로운 훈련 데이터가 생기면 기존의 데이터들의 영향력이 조금 줄어들어 과적합 해결이 가능할것이다. 즉, 모델의 전체 영향력을 기존 데이터가 나눠 갖고 새로운 데이터의 추가로 전체적인 영향력이 줄어들어 과적합이 사라지는 것이다. 다음의 방법으로 데이터 차원을 감소시키는 방법이 있다. SVM 모델을 예로, 커널 트릭을 활용해 데이터의 차원을 증가시켜 2차원에서 분류 불가능한 데이터를 3차원에서 분류할 수 있는데 이때 3차원에서 생성한 모델을 2차원으로 변형시키면 결정 경계가 과적합된 경우와 비슷한 형태를 보이게 된다. 반대의 경우를 생각해보면 과적합 된 모델의 데이터 차원을 감소시키면 결정 경계가 완만해져 과적합 해결이 가능하다. 마지막 방법으로 규제(Regularization)를 통해 모델 복잡도를 감소시켜 과적합이 일어나지 않게 하는 것으로, 위 방법들은 모델을 훈련하기 전에 시행하는 방식이라면 규제는 학습을 진행하면서 과적합을 제어하는 방식이다. 규제는 크게 계수의 절대값에 비례하여 패널티를 주는 Lasso(L1)과 계수의 제곱에 비례하여 패널티를 주는 Ridge(L2) 방식이 있다. 두 방식 모두 비용 함수(Cost Function)에 패널티를 추가해 기존의 비용(Cost)이 증가하여 학습을 최소화하게 된다.          수식을 보면, 기존의 Cost Function에 각각의 패널티가 더해진 것을 볼 수 있다. 이때 패널티는 가중치(w)에 학습률(Learning Rate)를 곱한 값으로 Cost Function에 포함되어 학습을 하면서 가중치 스스로 값이 커지는 것을 제어한다. 가중치가 큰 값을 갖지 못하기 때문에 회귀식에서 과적합으로 휘어진 식이 과적합 되지 못하고 직선형태로 변하게 되는 것이다. 모델 생성 시 타원의 중앙에서 Cost Function 최소가 되지만, 규제를 통해 타원의 중앙에 가장 근접한 가중치를 찾게 된다. 이때의 Cost Function은 타원의 중앙값이 아니라 최소가 되는 지점이 아니라 생각할수도 있으나 실제 데이터에서는 모델에 규제가 적용되어 더 좋은 효과를 보여주는 것이다.       결론     과적합(Overfitting)은 훈련 데이터를 과하게 학습하여 모델이 훈련 데이터에 최적화 된 것을 말한다.   과적합은 분산(Variance), 편향(Bias)과도 관련있으며, 분산이 크면 예측 분포가 큰 복잡한 모델을 만들어 과적합이 발생한다.   규제(Regularization)는 비용 함수(Cost Function)에 패널티를 추가해 과적합을 방지한다.    ","categories": ["ml"],
        "tags": ["test","ml"],
        "url": "/ml/ML-Overfitting,-%EA%B3%BC%EC%A0%81%ED%95%A9/",
        "teaser": null
      }]
