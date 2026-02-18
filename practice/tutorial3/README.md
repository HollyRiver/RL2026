Basic Reinforcement Learning Tutorial
https://github.com/vmayoral/basic_reinforcement_learning

## 3. OpenAI Gym

* OpenAI Gym 기본 사용법: 포크 버전인 Gymnasium으로 이전하는 게 안정적임...

### Installing

* python 3.10에서 버전 유지보수 끝냄...

```console
pip install gymnasium[all]
```

### The environment

`-` main class: env object
* 내부적으로 발생하는 환경을 캡슐화. 환경은 일부 또는 전체가 관측됨
* main API methods는 reset, step, render (render는 레거시인듯). overriding methods는 _step, _reset, _render
* action_space, observation_space attributes가 지정됨
> 유효한 행동, 상태의 공간
* step 함수가 action을 파라미터로 받으면 5개의 값을 리턴
> `observation` (object): 특정 환경을 대표하는 관측 객체(상태라고 보면 됨)
>
> `reward` (float): 이전 action에 대한 보상의 정도. 목표는 언제나 총 보상을 키우는 것
>
> `terminated` (bool): 에피소드 종단점 도달 여부
>
> `truncated` (bool): 에피소드가 너무 길어졌을 때 절사시킴
>
> `info` (dict): 디버깅 등에 활용할 수 있는 정보.


`-` 간단한 사용법

```Python
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = "human")
for i_episode in range(20):
    observation, info = env.reset()

    for t in range(100):
        print(observation)
        action = env.action_space.sample()    ## 여기서 Q-Learning 등의 정책을 반영. 현재는 그냥 랜덤
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# env.close()
```

* 가능한 환경 리스트 확인

```Python
from gym import envs
print(envs.registry.keys())
```

### Recoding

* `gymnasium.wrappers.RecordVideo`를 사용해서 녹화 가능 (.mp4)

```Python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, video_folder='./tmp/cartpole-experiment-1', episode_trigger=lambda count: count % 10 == 0)

for i_episode in range(20):
    observation, info = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
```

> episode_trigger에는 현재 에피소드 수를 입력으로, bool을 출력으로 하는 함수를 넣어주면 됨. 이건 특정 에피소드에서 영상을 저장할 것인지를 결정함