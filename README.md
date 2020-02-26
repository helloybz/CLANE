# Content- and Link-Aware Network Embedding

## 1. Introduction

   이 세상에는 다양한 네트워크 데이터가 존재합니다.
   SNS의 사용자 간 친구 관계, 문헌 간 인용 관계, 우리가 매일 접하는 월드 와이드 웹 등이 그 예시입니다.
   네트워크 데이터의 규모가 거대해지면서, 그 안에 내재되어 있는 정보들을 추출해 활용하기 위한 시도가 많아지고 있습니다.
   네트워크 데이터는 그래프라는 자료구조를 갖고 있습니다.
   그래프는 정점(node, vertex)의 집합과 정점들 사이를 연결하는 간선(edge, link)의 집합으로 구성됩니다.
   그래프는 데이터 간 '관계'라는 비교적 복잡한 정보를 나타내기에 유용하지만, 그 복잡성과 특수성 때문에 그 이상의 의미를 추출하기 위한 머신러닝 기법들을 적용하기 어렵습니다.
   대부분의 머신러닝 알고리즘들은 숫자의 형태를 갖는 데이터들을 입력받기 때문입니다.
   네트워크 데이터에 머신러닝 기법들을 적용할 수 있도록, '관계'를  수치화하는 것이 네트워크 임베딩 기법입니다.

   네트워크 임베딩은 그래프의 정점들을 임의의 벡터 공간에 할당하는 과정입니다. 비슷한 성격의 정점들은 비슷한 좌표 공간에 위치하도록, 성격이 다른 정점들은 비교적 멀리 위치하도록 할당하는 것이 중요합니다.

   <img src="https://1.bp.blogspot.com/-hx5DlfIn7xk/XRJlD47Mv6I/AAAAAAAAEO4/o9ztIaCTz7Ie2eVEczhyGuciQPxV7JKFACLcBGAs/s640/Screenshot%2B2019-06-25%2Bat%2B11.11.05%2BAM.png"></img>
   
   위 그림은 대표적인 네트워크 임베딩 기법인 Deepwalk의 적용 예시입니다.
   왼쪽 그림 네트워크 데이터를 시각화 한 것이고, 오른쪽 그림은 이 네트워크 안의 정점들을 임베딩한 결과입니다.
   여기서는 2차원 벡터 공간에 할당한 것을 알 수 있습니다.
   정점의 색깔이 각 정점의 성격을 의미하는데, 임베딩 결과에서도 같은 색깔의 정점끼리 모여 있는 것을 알 수 있습니다.
   이제 정점들은 그들의 관계 정보를 반영한 수치인 임베딩을 얻었으므로 다양한 머신러닝 기법을 적용할 수 있습니다.

## 2. Related Works

   Deepwalk, Node2Vec은 네트워크 임베딩의 효시가 되는 메소드로, 다음과 같은 과정을 거쳐 임베딩을 얻습니다.

   1. 주어진 그래프에서 모든 정점들로부터 다수의 랜덤워크를 시행해 충분한 수의 랜덤워크 샘플을 얻습니다.
   2. 이렇게 얻은 랜덤워크 샘플들을 각각 하나의 문장으로, 각 정점을 단어로 간주합니다.
   3. 문장을 구성하는 단어의 임베딩을 생성하는 언어 모델을 차용해 정점의 임베딩을 생성합니다.

## 3. Methods

### 3.1 Prerequisites

   임의의 그래프 G=(V,E)가 있습니다.
   V는 G 내 존재하는 모든 정점의 집합이고, E는 G 내 존재하는 모든 간선의 집합입니다.
   임의의 정점 v는 그만의 내적 정보를 (어떤 형태로든) 가지고 있습니다.
   그 정보의 임베딩을 정점 v의 내용 임베딩이라고 부르고 x_v라고 표기하겠습니다.

   모든 간선은 가중치를 갖습니다.
   임의의 두 노드 v와 u가 있고 두 노드 사이에 간선이 존재할 때, v로부터 u로 단 한번의 이동으로 갈 수 있는 확률을 M_vu라고 하고 이 확률을 weight로 사용하겠습니다.
   추가로, 이 확률을 두 정점의 유사도로부터 계산해, 유사도가 높을 수록 weight가 높아지도록 하겠습니다.
   위 정의에 따라, 간선의 weight는 다음과 같은 성질을 갖습니다.

   ![Sum of the weights goes 1](./resources/Eq1.png)

   N(v)는 정점 v의 인접 노드의 집합입니다. 임의의 정점 v와 인접 노드u를 잇는 간선의 weight는 위에서 정의했듯이 v에서 u로 이동할 확률이므로 모든 인접 노드에 대한 weight의 합은 1입니다.

   확률 행렬 M=|M_vu|을 생각해보겠습니다. 
   
### 3.2 

### 3.2 Random Walks on Graph

   그래프에서 무작위 행보란, 일종의 확률 과정(stochastic process)으로 k개의 확률 변수 X1, ..., Xk로 이루어집니다.
   X0는 무작위 행보를 시작하는 정점이며, 정점 Xi+1는 Xi의 이웃노드들 중 임의로 하나를 골라 정해집니다.
   k는 자연수이며 무작위 행보의 길이를 나타냅니다.
   이웃노드들 중 하나를 선택하는 확률의 분포로 Uniform distribution을 사용할 수 있겠지만, 우리는 앞서 정의한 M을 활용해 보겠습니다.
   그러면, Xi의 이웃노드들 중 Xi와 가장 

   

   가령, 정점 A를 시작으로 무작위 행보를 한 번 시행해 일련의 정점들을 얻었다고 생각해보겠습니다.
   여기에는 정점 A를 기준으로 그래프의 지역적인 구조 정보의 일부가 들어있다고 생각할 수 있습니다.
   무작위 행보는 확률 과정이기 때문에 한번의 무작위 행보 시행으로는 정점 A주변의 구조 정보를 모두 담아내기 어려울 것입니다.
   만약 정점 A를 기준으로 충분한 횟수의 무작위 행보를 시행한다면, 정점 A 주변의 지역적인 구조 정보를 충분히 담아낼 수 있을 것입니다.
   더 나아가 그래프 내 모든 정점들에 대해서 충분한 횟수의 무작위 행보를 시행한다면, 얻어진 무작위 행보 샘플들에는 그래프의 전체적인 구조 정보가 들어 있다고 볼 수 있습니다.

### 3.2 Node Embedding
### 3.3 Training Similarity Measure


## 4. Experiments
