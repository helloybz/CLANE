# Content- and Link-Aware Network Embedding

## Introduction
    - 네트워크 임베딩
네트워크 임베딩은 노드와 엣지로 표현되는 네트워크 데이터에 내재되어 있는 정보를 벡터 공간에 매핑하는 기법입니다. 
네트워크 구조 안에는 여러가지 형태의 데이터가 있습니다. 
본 연구에서는 그중에서도 노드를 벡터 공간에 할당하는 방법은 소개합니다.
노드 사이의 연결관계만 반영하여 매핑하는 선행 연구들과는 달리, 각 노드에 내재된 고유한 데이터도 함께 반영하여 노드를 매핑합니다. 
더 나아가, 비대칭적인 유사도를 제시하여, 두 노드사이의 연결 방향도 고려할 수 있습니다.
 유사도는 간단한 뉴럴넷으로 구성하였으며, 노드들의 연결관계의 분포를 학습합니다.

    - 마코프 연쇄를 활용한 네트워크 임베딩
    - 학습이 가능한 노드 간 유사도
    - 간선의 방향을 고려한 네트워크 임베딩
## Methods
    - 노드 임베딩 optimization과 유사도 모델 학습의 iterative method
### 유사도 모델
    - 임의의 두 노드 x1, x2. 각 노드의 임베딩을  z1, z2.
    1. 코사인 유사도
      - 두 노드의 간선 방향 고려 불가.
      - 학습 불가
      - 연산이 간단함.
    2. 뉴럴넷 기반 유사도
      - 두 노드의 간선 방향 고려.
      - z1 -> z2일 때, n x n 

   
