import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make('CartPole-v1', render_mode="rgb_array")  ## 영상을 파일로 저장 시 사용하는 렌더링 모드
## wrapper로 env 객체 감싸기, 10개 에피소드마다 녹화 (에피소드 단위)
env = RecordVideo(env, video_folder='./cartpole-experiment-1', episode_trigger=lambda count: count % 10 == 0)

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