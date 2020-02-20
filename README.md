# Content- and Link-Aware Network Embedding

## 1. 도입
## 1.1. 네트워크 임베딩
   이 세상에는 다양한 네트워크 데이터가 존재합니다.
   SNS의 사용자 간 친구 관계, 문헌 간 인용 관계 등을 그 예시입니다.
   네트워크 데이터의 규모가 거대해지면서, 그 안에 내재되어 있는 정보들을 활용하기 위한 시도가 많아지고 있습니다.
   네트워크 데이터는 그래프라는 자료구조를 갖고 있습니다.
   그래프는 정점의 집합과 정점들 사이를 연결하는 간선의 집합으로 구성됩니다.
   그래프는 데이터 간 '관계'라는 비교적 복잡한 정보를 나타내기에 유용하지만, 그 복잡성과 특수성 때문에 그 이상의 의미를 추출하기 위한 머신러닝 기법들을 적용하기 어렵습니다.
   대부분의 머신러닝 알고리즘들은 수치화된 데이터들을 입력받기 때문입니다.
   네트워크 데이터에 머신러닝 기법들을 적용할 수 있도록, '관계'를  수치화하는 것이 네트워크 임베딩 기법입니다.

   네트워크 임베딩은 그래프의 정점들을 임의의 벡터 공간에 할당하는 과정입니다. 비슷한 성격의 정점들은 비슷한 좌표 공간에 위치하도록, 성격이 다른 정점들은 비교적 멀리 위치하도록 할당하는 것이 중요합니다.
   
   <img src="https://1.bp.blogspot.com/-hx5DlfIn7xk/XRJlD47Mv6I/AAAAAAAAEO4/o9ztIaCTz7Ie2eVEczhyGuciQPxV7JKFACLcBGAs/s640/Screenshot%2B2019-06-25%2Bat%2B11.11.05%2BAM.png" class="content-image">
   
   위 그림은 대표적인 네트워크 임베딩 기법인 Deepwalk의 적용 예시입니다.
   왼쪽 그래프의 정점들을 그들이 연결된 관계를 반영하여 2차원 벡터 공간에 할당한 결과입니다.
   정점의 색깔이 각 정점의 성격을 의미하는데, 임베딩 결과에서도 같은 색깔의 정점끼리 모여 있는 것을 알 수 있습니다.
   
   각 정점들이 벡터 공간에 할당된 그 좌표를 그 정점의 임베딩이라고 부릅니다.
   이제 정점들은 그들의 관계 정보를 반영한 수치인 임베딩을 얻었으므로 다양한 머신러닝 기법을 적용할 수 있습니다.

## 1.2. 무작위 행보
   그래프에서 무작위 행보란, 일종의 확률 과정(stochastic process)으로 k개의 확률 변수 X0, X1, ..., Xk로 이루어집니다.
   X0는 무작위 행보를 시작하는 노드이며, Xi+1는 Xi의 이웃노드들 중 임의로 하나를 골라 정해집니다.
   k는 자연수이며 무작위 행보의 길이를 나타냅니다.
   가령, 정점 A를 시작으로 무작위 행보를 한 번 시행해 일련의 정점들을 얻었다고 생각해보겠습니다.
   여기에는 정점 A를 기준으로 그래프의 지역적인 구조 정보의 일부가 들어있다고 생각할 수 있습니다.
   무작위 행보는 확률 과정이기 때문에 한번의 무작위 행보 시행으로는 정점 A주변의 구조 정보를 모두 담아내기 어려울 것입니다.
   만약 정점 A를 기준으로 충분한 횟수의 무작위 행보를 시행한다면, 정점 A 주변의 지역적인 구조 정보를 충분히 담아낼 수 있을 것입니다.
   더 나아가 그래프 내 모든 정점들에 대해서 충분한 횟수의 무작위 행보를 시행한다면, 얻어진 무작위 행보 샘플들에는 그래프의 전체적인 구조 정보가 들어 있다고 볼 수 있습니다.
   
## 1.3. 언어 모델의 활용
   하지만 이렇게 얻어진 무작위행보 샘플들도 머신러닝 알고리즘에 사용할 수 있게 수치화된 것은 아닙니다.
   한편, 자연어 처리 분야에서는 언어 모델중 하나인 Skip-gram을 활용해 단어의 임베딩을 성공적으로 수행하는 Word2Vec이 널리 알려져 있습니다.
   앞서 살펴보았던 무작위 행보 샘플 하나를 문장으로, 샘플을 구성하는 정점들을 단어로 간주한다면, Word2Vec을 이용해 각 정점의 임베딩을 구할 수 있습니다.
   
## 1.4.???
   - 기존 방법의 문제점.
      1. 샘플링 방법에 따라 임베딩의 품질이 결정됨.
      2. 
   - 제안하는 방법론
      
## 2. 알고리즘
## 2.1. 임베딩 최적화
## 2.2. 유사도 척도 학습
## 3. 실험
