
from __future__ import annotations

import glob
import os
import time
import tensorboard

import supersuit as ss
from stable_baselines3 import PPO, DQN
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.sisl import multiwalker_v9
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback,CallbackList

from stable_baselines3.common.vec_env import VecVideoRecorder


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn.parallel_env(**env_kwargs, n_walkers = 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")
    #env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")
    

    #checlpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=max(500000 //8,1 ), save_path='./chekpoint_models/level_0_2w_to_3w',
                                         name_prefix=f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}",
                                         save_replay_buffer=True,save_vecnormalize=True)
    eval_callback = EvalCallback(env)
    callback = CallbackList([checkpoint_callback,eval_callback])

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
        normalize_advantage=True,
        tensorboard_log="logs/level_0_2w_to_3w",
    )
    
    model.load("chekpoint_models/level_0_noFW/multiwalker_v9_20240127-161752_8000000_steps.zip")

    model.set_env(env)
    
    model.learn(total_timesteps=steps, callback=callback)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    
    
    env = env_fn.env(render_mode=render_mode, **env_kwargs, n_walkers = 3)
    
    # Apply the same frame stacking to the evaluation environment
    env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, 3)


    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    #model = PPO.load(latest_policy)chekpoint_models/level_0/multiwalker_v9_20240127-161752_14000000_steps.zip
    model = PPO.load("/home/matteo/projects/RL/RL-Projects/multiwalker/chekpoint_models/level_0_2w_to_3w/multiwalker_v9_20240201-105646_7500000_steps.zip")

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using the same model for every agent
    for i in range(num_games):
                   
        
        env.reset(seed=i)   

        for agent in env.agent_iter():
            
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = multiwalker_v9
    env_kwargs = {}

    # Train a model (takes ~3 minutes on GPU)
    #train_butterfly_supersuit(env_fn, steps=8_000_000, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=3, render_mode="human", **env_kwargs)