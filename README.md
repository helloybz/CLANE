# Content- and Link-Aware Network Embedding

## Introduction
    - 네트워크 임베딩
    - 노드의 내적 정보도 반영한 네트워크 임베딩
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

   
