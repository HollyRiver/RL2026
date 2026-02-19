Basic Reinforcement Learning Tutorial
https://github.com/vmayoral/basic_reinforcement_learning

## 4. Q-Learning in OpenAI Gym

* Gymnasium 라이브러리를 활용하여 CartPole problem을 풀이

### The CartPole problem

* pole attached by an un-actuated joint to a cart.
* 막대가 카트에 수동 관절 형태로 연결된 상태, 카트는 마찰 없이 좌우로만 움직일 수 있음.
* 막대가 바닥에 닿지 않도록 해야 함
* 막대가 지표면과 수직인 상태에서 좌우로 12도 이상 회전하거나 카트가 화면을 벗어나면(중앙에서 2.4만큼 이동) 에피소드 종료. 500 time-step 이후 truncated
* 매 time-step마다 1씩 보상 제공


### Adapting Q-learning

`-` Q-Learning으로 풀이

```Python
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
```

* `__init__`: actions, epsilon, alpha, gamma를 입력
> action_space, exploration, learning_rate, discount_rate
* `learnQ`: state, action, reward, value를 입력
> Q(s, a)를 행동에 대한 보상과, 다음 시간의 할인된 최고 가치로 갱신
* `chooseAction`: state, return_q를 입력
> 학습된 Q-values table과 exploration factor epsilon에 따라 다음 행동을 결정
* `learn`: state, action, reward, nextstate를 입력
> 행동 이후 최대 가치를 계산하고, `learnQ` 함수를 호출해서 Q-value table 업데이트


`-` 구간으로 변환

* 해당 문제는 연속적이므로 구현에 어려움이 있음. 따라서 실제 구현에서는 공간을 이산적으로 추상화하는 경향이 있음

```Python
def build_state(features):    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

cart_position_bins = pandas.cut([-4.8, 4.8], bins=n_bins, retbins=True)[1][1:-1]
pole_angle_bins = pandas.cut([-0.418, 0.418], bins=n_bins_angle, retbins=True)[1][1:-1]
cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]
```

* [cartpole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)을 참고하여 `env.observation_space`에서 얻은 값을 입력
* cart_velocity_bins, angle_rate_bins는 물리학적 지식을 활용해서 경험적으로 기입한 듯...