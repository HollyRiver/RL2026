Basic Reinforcement Learning Tutorial
https://github.com/vmayoral/basic_reinforcement_learning

## 2. SARSA

* on-policy algorithm for temporal difference learning
> 추정된 value fuction을 target policy에게서 결정된 행동에 의해서 업데이트
* Q-Learning과 달리 다음 상태에서의 최대 보상이 업데이트에 필수적으로 사용되지 않음

```{raw}
Initialize Q(s, a) arbitrarily
Repeat (for each episode):
	Initialize s
	Choose a from s using policy derived from Q
	While (s is not a terminal state):
		Take action a, observe r, s'
		Choose a' from s' using policy derive from Q
		Q(s,a) += alpha * (r + gamma * Q(s', a') - Q(s,a))
		s = s', a = a'
```

## Implementation of SARSA

* Q-Learning learn method

```Python
def learn(self, state1, action1, reward, state2):
    maxqnew = max([self.getQ(state2, a) for a in self.actions])
    self.learnQ(state1, action1,
                reward, reward + self.gamma*maxqnew)
```

* SARSA learn method

```Python
def learn(self, state1, action1, reward, state2, action2):
    qnext = self.getQ(state2, action2)
    self.learnQ(state1, action1,
                reward, reward + self.gamma * qnext)
```

## The cliff example

* 2D world. 플레이어가 시작지점에서 초록색 영역에 도달하되, 붉은 색 영역을 피해 이동