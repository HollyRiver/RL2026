Basic Reinforcement Learning Tutorial
https://github.com/vmayoral/basic_reinforcement_learning

## 5. Deep Q-Learning

* 많은 시나리오에서 단순 Q-values table은 제대로 확장되지 않음
* Pacman과 같은 그래픽 기반 게임에서 state는 픽셀 데이터가 되며, tabular method에서는 단 하나의 픽셀만 변경되어도 테이블에 완전히 별도의 항목으로 상태가 저장되어야 함
> 매우 비효율적. 상태 간 일반화 및 패턴 매칭의 방법이 필요. 상태 X의 정확한 가치를 산출하는 것보다 상태 X와 같은 종류의 가치를 추정하는 알고리즘이 요구됨
* neural networks, simple linear model 등 다양한 형태의 근사 함수를 사용할 수 있음
* state-action 쌍에 대해서 가치를 뱉어낼 수 있어야 함

```
## traditional Q-learning:
Q(s, a) += alpha * (reward(s, a) + gamma * max(Q(s') - Q(s, a))

## DQN (y label이 되는 값)
target = reward(s, a) + gamma * max(Q(s'))
```

### Playing with the hyperparameters in a DQN

* cartpole problem의 해결
> input dimension: 4 (위치, 가속도, 봉의 각도, 봉의 각도에 대한 가속도)
>
> output dimension: 2 (Binary Classification, 왼쪽/오른쪽으로 이동)

* 학습 데이터 (`Memory`)
> size만큼 이전의 `[s, a, r, s', t]`를 저장하는 메모리 마련. state, action, reward, nextState, isFinal
>
> dequeue 형태로 구현하여 최대 길이를 초과한 step이 진행되면 가장 나중의 것을 제외 (코드에선 메모리를 신경쓰지 않고 속도를 우선하여 시작 인덱스를 변수로 추가하였음)
>
> `getMiniBatch` 함수를 사용하여 덱에 저장된 데이터를 랜덤 샘플링으로 k개 뽑아내어 학습에 사용함

* DQN
> input_size, output_size, memory, discountFactor, learnStart, learningRate를 인자로 선언
>
> input_size=4, output_size=2인 네트워크를 구성 (createModel)


```Python
    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples        
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)
```

> `updateTargetNetwork=10000`에 따라 그 이전 step까지는 한 개 모델에서 가치 추정(y값 산출) 및 업데이트 전부 수행
>
> 그 이후 step에서는 TargetModel에서 가치 추정, Model만 업데이트
>
> true label y값(target)은 해당 state-action으로 얻은 보상에 새로운 상태에서 얻을 수 있는 최대 가치 추정치로 계산됨

* 그 외 Q-Learning 코드는 동일하게 구성하면 됨