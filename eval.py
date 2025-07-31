from stable_baselines3 import DQN
from car_game_env import CarGameEnv
from gymnasium.wrappers import RecordVideo

env = CarGameEnv(render_mode='rgb_array')
env = RecordVideo(env, video_folder="eval_videos", episode_trigger=lambda e: True)

model = DQN.load("car_rl_model")

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
