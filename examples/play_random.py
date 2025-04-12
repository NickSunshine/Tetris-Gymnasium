from dataclasses import dataclass
import gymnasium as gym
import tyro
import os
import time
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation

from stable_baselines3.common.atari_wrappers import ClipRewardEnv


@dataclass
class Args:
    exp_name: str = "play_random"
    """The name of this experiment"""
    seed: int = 1
    """Seed for reproducibility"""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """If toggled, CUDA will be enabled by default"""
    track: bool = True
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tetris_gymnasium_random"
    """The WandB project name"""
    wandb_entity: str = None
    """The entity (team) of the WandB project"""
    capture_video: bool = True
    """Whether to capture videos of the agent's performance"""
    env_id: str = "tetris_gymnasium/Tetris"
    """The ID of the environment"""
    total_timesteps: int = 10000000
    """Total timesteps to run the random agent"""


def make_env(env_id, capture_video, run_name):
    if capture_video:  # Capture video for the environment
        env = gym.make(env_id, render_mode="rgb_array")
        env = RgbObservation(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
        env = RgbObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ClipRewardEnv(env)
    return env


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Initialize WandB if tracking is enabled
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Initialize the environment
    env = make_env(args.env_id, args.capture_video, run_name)
    next_obs, _ = env.reset(seed=args.seed)

    global_step = 0

    while global_step < args.total_timesteps:
        global_step += 1

        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Log episodic metrics if the episode has ended
        if "episode" in info:
            episodic_return = info["episode"]["r"]
            episodic_length = info["episode"]["l"]
            print(f"global_step={global_step}, episodic_return={episodic_return}")

            # Log to TensorBoard
            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
            writer.add_scalar("charts/episodic_length", episodic_length, global_step)

            # Log to WandB
            if args.track:
                wandb.log(
                    {
                        "charts/episodic_return": episodic_return,
                        "charts/episodic_length": episodic_length,
                        "global_step": global_step,
                    }
                )

        # Reset the environment if the episode has terminated
        if terminated or truncated:
            next_obs, _ = env.reset()

    env.close()
    writer.close()