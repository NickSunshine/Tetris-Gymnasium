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
from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.mappings.actions import ActionsMapping

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
    reward: str = "R0"
    """Reward mapping to use: R0, R1, R2, R3, or R4"""
    action: str = "A0"
    """Action mapping to use: A0 or  A1"""


def make_env(env_id, capture_video, run_name, reward, action):
    R0 = RewardsMapping(
        alife=1.0,
        clear_line=1.0,
        game_over=0.0,
    )

    R1 = RewardsMapping(
        alife=0.0,
        clear_line=1.0,
        game_over=0.0,
    )

    R2 = RewardsMapping(
        alife=1.0,
        clear_line=1.0,
        game_over=-1.0,
    )

    R3 = RewardsMapping(
        alife=0.0,
        clear_line=1.0,
        game_over=-1.0,
    )

    R4 = RewardsMapping(
        alife=0.1,
        clear_line=1.0,
        game_over=-10.0,
    )

    A0 = ActionsMapping(
        move_left = 0,
        move_right = 1,
        move_down = 2,
        rotate_clockwise = 3,
        rotate_counterclockwise = 4,
        hard_drop = 5,
        swap = 6,
        no_op = 7
    )

    A1 = ActionsMapping(
        move_left = 0,
        move_right = 1,
        rotate_clockwise = 2,
        rotate_counterclockwise = 3,
        no_op = 4
    )

    if reward == "R0":
        selected_reward = R0
    elif reward == "R1":
        selected_reward = R1
    elif reward == "R2":
        selected_reward = R2
    elif reward == "R3":
        selected_reward = R3
    elif reward == "R4":
        selected_reward = R4
    else:
        raise ValueError(f"Invalid reward option: {reward}. Choose from R0, R1, R2, or R3")
    
    if action == "A0":
        selected_action = A0
    elif action == "A1":
        selected_action = A1
    else:
        raise ValueError(f"Invalid action option: {action}. Choose from A0 or A1")

    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array", rewards_mapping=selected_reward, actions_mapping=selected_action)
        env = RgbObservation(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id, rewards_mapping=selected_reward, actions_mapping=selected_action)
        env = RgbObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if reward in ["R0", "R1", "R2", "R3"]:
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
    env = make_env(args.env_id, args.capture_video, run_name, args.reward, args.action)
    next_obs, _ = env.reset(seed=args.seed)
    cumulative_lines_cleared = 0  # Initialize cumulative lines cleared
    best_episodic_return = float("-inf")  # Initialize best episodic return
    best_episodic_lines_cleared = float("-inf")  # Initialize best episodic lines cleared

    global_step = 0

    while global_step < args.total_timesteps:
        global_step += 1

        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if "lines_cleared" in info:
            cumulative_lines_cleared += info["lines_cleared"]

        # Log episodic metrics if the episode has ended
        if "episode" in info:
            episodic_return = info["episode"]["r"]
            episodic_length = info["episode"]["l"]
            episodic_time = info["episode"]["t"]

            print(f"global_step={global_step}, episodic_return={episodic_return}")

            # Log to TensorBoard
            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
            writer.add_scalar("charts/episodic_time", episodic_time, global_step)
            writer.add_scalar("charts/episodic_lines_cleared", cumulative_lines_cleared, global_step)  # Log lines cleared

            # Log to WandB
            if args.track:
                wandb.log(
                    {
                        "charts/episodic_return": episodic_return,
                        "charts/episodic_length": episodic_length,
                        "charts/episodic_time": episodic_time,
                        "charts/episodic_lines_cleared": cumulative_lines_cleared,  # Log lines cleared
                        "global_step": global_step,
                    }
                )

            # Update and log best episodic return
            if episodic_return > best_episodic_return:
                best_episodic_return = episodic_return
                print(f"New best episodic return: {best_episodic_return}")

                # Log to TensorBoard
                writer.add_scalar("charts/best_episodic_return", best_episodic_return, global_step)

                # Log to WandB
                if args.track:
                    wandb.log({"charts/best_episodic_return": best_episodic_return, "global_step": global_step})

            # Update and log best episodic lines cleared
            if cumulative_lines_cleared > best_episodic_lines_cleared:
                best_episodic_lines_cleared = cumulative_lines_cleared
                print(f"New best episodic lines cleared: {best_episodic_lines_cleared}")

                # Log to TensorBoard
                writer.add_scalar("charts/best_episodic_lines_cleared", best_episodic_lines_cleared, global_step)

                # Log to WandB
                if args.track:
                    wandb.log({"charts/best_episodic_lines_cleared": best_episodic_lines_cleared, "global_step": global_step})


            # Reset cumulative lines cleared for the next episode
            cumulative_lines_cleared = 0

        # Reset the environment if the episode has terminated
        if terminated or truncated:
            next_obs, _ = env.reset()

    env.close()
    writer.close()