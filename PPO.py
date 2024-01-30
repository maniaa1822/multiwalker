
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




def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn.parallel_env(**env_kwargs, n_walkers = 2, terminate_on_fall = True, max_cycles = 500)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")
    #env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")
    

    #checlpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=max(500000 //8,1 ), save_path='./chekpoint_models/level_0',
                                         name_prefix=f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}",
                                         save_replay_buffer=True,save_vecnormalize=True)
    eval_callback = EvalCallback(env)
    callback = CallbackList([checkpoint_callback,eval_callback])

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=0.00025,
        batch_size=512,
        normalize_advantage=True,
        n_steps=4096,
        n_epochs=30,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.3,
        ent_coef=0.001,
        tensorboard_log="logs/level_0",
    )
    
    #model.load("chekpoint_models/multiwalker_v9_20240124-114702_32000_steps.zip")

    model.learn(total_timesteps=steps, callback=callback)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs, n_walkers = 2,terminate_on_fall = False, remove_on_fall = False)
    
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

    #model = PPO.load(latest_policy)
    model = PPO.load("chekpoint_models/level_0/multiwalker_v9_20240127-132409_1000000_steps.zip")

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
    #train_butterfly_supersuit(env_fn, steps=15_000_000, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=3, render_mode="human", **env_kwargs)