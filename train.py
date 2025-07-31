from stable_baselines3 import DQN
from car_game_env import CarGameEnv
from stable_baselines3.common.callbacks import EvalCallback

env = CarGameEnv()

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,
    buffer_size=50000,
    batch_size=64,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

eval_callback = EvalCallback(env, best_model_save_path="./best_model/",
                             log_path="./logs/", eval_freq=10000,
                             deterministic=True, render=False)

model.learn(total_timesteps=50_000_000, callback=eval_callback, progress_bar=True)
model.save("car_rl_model")
print("Model saved.")





import os
import shutil
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from car_game_env import CarGameEnv
from gymnasium.wrappers import RecordVideo

# ✅ Setup
env = CarGameEnv(render_mode='rgb_array')  # Required for RecordVideo or gym's eval

# Optional: wrap with RecordVideo to capture progress
env = RecordVideo(
    env,
    video_folder="training_videos",
    episode_trigger=lambda ep: ep % 10 == 0,
    name_prefix="car_rl_training"
)

# ✅ EvalCallback to track and save best model
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    verbose=1
)

# ✅ Initialize DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,
    buffer_size=50000,
    batch_size=64,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# ✅ Start training
model.learn(
    total_timesteps=50_000_000,
    callback=eval_callback,
    progress_bar=True
)

# ✅ Save final model (optional)
model.save("car_rl_model_final")

# ✅ Automatically overwrite final model with best model
best_model_path = "./best_model/best_model.zip"
final_model_path = "car_rl_model.zip"

if os.path.exists(best_model_path):
    shutil.copyfile(best_model_path, final_model_path)
    print("✅ Best model copied and saved as car_rl_model.zip")
else:
    print("⚠️ No best model found. Check if training ran long enough.")

env.close()
