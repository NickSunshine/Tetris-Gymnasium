from dataclasses import dataclass
import cv2
import gymnasium as gym
import tyro

from tetris_gymnasium.envs.tetris import Tetris


@dataclass
class Args:
    render_mode: str = "ansi"
    """Render mode for the environment: 'ansi' or 'human'"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Validate render_mode
    if args.render_mode not in ["ansi", "human"]:
        raise ValueError("Invalid render_mode. Choose 'ansi' or 'human'.")

    env = gym.make("tetris_gymnasium/Tetris", render_mode=args.render_mode)
    env.reset(seed=42)

    terminated = False
    while not terminated:
        if args.render_mode == "ansi":
            print(env.render() + "\n")  # ANSI-specific rendering
        elif args.render_mode == "human":
            env.render()  # Human-specific rendering
            

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if args.render_mode == "human":
            key = cv2.waitKey(100)  # Timeout for human mode

    print("Game Over!")