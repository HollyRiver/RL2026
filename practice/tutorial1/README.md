Basic Reinforcement Learning Tutorial
https://github.com/vmayoral/basic_reinforcement_learning

## 1. Q-learning

* state-action pair에 대한 Value Function Q(s, a)를 학습
> 특정 상태 s가 주어질 때, 행동 a가 얼마나 좋은지에 대한 추정값
* off-policy algorithm for temporal difference learning
> 추정된 value function을 실제로는 시도되지 않을 행동으로 업데이트할 수 있음

```{raw}
Initialize Q(s,a) arbitrarily
Repeat (for each generation):
	Initialize state s
	While (s is not a terminal state):		
		Choose a from s using policy derived from Q
		Take action a, observe r, s'
		Q(s,a) += alpha * (r + gamma * max,Q(s') - Q(s,a))
		s = s'
```

* s: 이전 상태, a: 이전 행동
* Q(): Q-learning 알고리즘
* s': 현재 상태
* alpha: 학습률. [0, 1]
* gamma: discount factor
* max,: s'에서 보상을 최대화하는 행동을 선택했을 때의 보상

1. Q-value table을 초기화
2. 현재 상태를 관측
3. 현재 상태에서 행동을 선택
4. 행동을 취하고 보상 r과 다음 상태 s'을 획듯
5. 현재 상태를 다음 상태로 설정하고 종단 상태에 도달할 때까지 2~5를 반복

### The World and Cat player implementations

* discrete 2D world

`-` Cat Player

* Cat player class는 cellular.Agent 클래스에 선언되고 Mouse player를 추적하도록 세팅됨
```{goTowards Algorithm}
현재 셀이 타겟 셀이면 리턴
현재 셀에서 이웃한 셀 8개에 대해서 해당 셀이 타겟 셀이면 best로 리턴
8개 셀에 대해서 타겟 셀과 거리를 계산하고 가장 짧아지는 곳으로 이동
베스트 셀이 벽이면 리턴 (랜덤 이동)
```
* 타겟을 잡은 상태가 아니면 이동
* 이동 후에 반환한 셀이 기존 셀과 같으면 랜덤한 방향으로 이동

`-` Mouse Player

* Q learning 수행 대상
* 8개의 행동 존재
* Cat에게 잡아먹힌 횟수 / 치즈를 먹은 횟수를 저장하여 퍼포먼스 측정
* 지난 상태 s와 지난 행동 a를 저장하여 Q-value table 업데이트
* directions: `dx, dy = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]`
* epsilon: exploration constant
* 다음 상태 조회
> 해당 셀에 Cat이 있음: 3
>
> 해당 셀에 치즈가 있음: 2
>
> 해당 셀이 벽임: 1
>
> 그 외(빈 공간): 0
* lookdist 값에 따라 인근 n개 칸까지 조회 가능 (여기선 2로 설정하여 2칸까지 확인 가능)

### Q-learning 구현

* update 함수

```{raw}
인근 셀의 상태를 조회 (calcState)
기본적으로 -1의 보상을 제공 (이동에 따른 보상)

## Q-value table 업데이트
1. 현재 셀에 고양이가 있음: eaten ++, reward = -100
> self.ai.learn(s, a, r, s'), mouse는 랜덤 위치로 이동
2. 현재 셀에 치즈가 있음: fed ++, reward = 50
> 잡혀서 랜덤 이동으로 치즈에 도달한 게 아니면 self.ai.learn(s, a, r, s')

## 액션 선택 및 실행
self.ai에서 현재 상태에 대해서 policy 기반 액션 선택 후 실제 행동
```

* 고양이에게 잡히면 -100
* 치즈를 먹으면 +50
* 그 외의 모든 이동 -1