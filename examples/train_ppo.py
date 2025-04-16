# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import matplotlib.pyplot as plt
import os

from tetris_gymnasium.mappings.rewards import RewardsMapping

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tetris_gymnasium_ppo"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model into the `runs/{run_name}` folder"""
    load_model_path: str = None
    """Path to a saved model to load and continue training"""
    eval_only: str = None
    """Path to a saved model to evaluate without training"""
    reward: str = "R0"
    """Reward mapping to use: R0, R1, or R2"""

    # Algorithm specific arguments
    env_id: str = "tetris_gymnasium/Tetris"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 6
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, reward):
    def thunk():
        
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

        if reward == "R0":
            selected_reward = R0
        elif reward == "R1":
            selected_reward = R1
        elif reward == "R2":
            selected_reward = R2
        else:
            raise ValueError(f"Invalid reward option: {reward}. Choose from R0, R1, or R2.")

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", rewards_mapping=selected_reward)
            env = RgbObservation(env)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, rewards_mapping=selected_reward)
            env = RgbObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def print_summary_statistics(name, data):
        if len(data) == 0:
            print(f"{name}: No data available.")
            return

        data = np.array(data, dtype=np.float32)

        mean_value = np.mean(data)
        median_value = np.median(data)
        std_dev = np.std(data, ddof=1) if len(data) > 1 else 0.0
        min_value = np.min(data)
        max_value = np.max(data)

        print(
            f"{name} - Mean: {mean_value:.3f}, Median: {median_value:.3f}, "
            f"Std Dev: {std_dev:.3f}, Min: {min_value:.3f}, Max: {max_value:.3f}"
        )

def create_histogram(data, title, xlabel, filename, run_name, plots_dir, color='blue', bins=20):
    """
    Creates and saves a histogram.

    Args:
        data (list or np.array): The data to plot.
        title (str): The title of the histogram.
        xlabel (str): The label for the x-axis.
        filename (str): The name of the file to save the plot as.
        run_name (str): The name of the current run (used in the footnote).
        plots_dir (str): The directory to save the plot in.
        color (str): The color of the bars in the histogram.
        bins (int): The number of bins for the histogram.
    """
    data = np.array(data).flatten()  # Ensure the data is a 1D array
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
    plt.title(title, y=1.02)
    plt.figtext(0.5, 0.01, f"Run Name: {run_name}", ha="center", fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    histogram_path = os.path.join(plots_dir, filename)
    plt.savefig(histogram_path)
    plt.close()
    print(f"{title} saved to {histogram_path}")

def create_violin_plot(data, title, xlabel, filename, run_name, plots_dir, color='blue'):
    """
    Creates and saves a violin plot.

    Args:
        data (list or np.array): The data to plot.
        title (str): The title of the violin plot.
        xlabel (str): The label for the x-axis.
        filename (str): The name of the file to save the plot as.
        run_name (str): The name of the current run (used in the footnote).
        plots_dir (str): The directory to save the plot in.
        color (str): The color of the violin plot.
    """
    data = np.array(data).flatten()  # Ensure the data is a 1D array
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    
    # Customize the color of the violin plot
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')  # Mean line color
    parts['cmedians'].set_color('black')  # Median line color

    plt.title(title, y=1.02)
    plt.figtext(0.5, 0.01, f"Run Name: {run_name}", ha="center", fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    violin_path = os.path.join(plots_dir, filename)
    plt.savefig(violin_path)
    plt.close()
    print(f"{title} saved to {violin_path}")

def create_scatter_plot(x_data, y_data, title, xlabel, ylabel, filename, run_name, plots_dir, color='blue'):
    """
    Creates and saves a scatterplot.

    Args:
        x_data (list or np.array): The data for the X-axis.
        y_data (list or np.array): The data for the Y-axis.
        title (str): The title of the scatterplot.
        xlabel (str): The label for the X-axis.
        ylabel (str): The label for the Y-axis.
        filename (str): The name of the file to save the plot as.
        run_name (str): The name of the current run (used in the footnote).
        plots_dir (str): The directory to save the plot in.
        color (str): The color of the scatter points.
    """
    x_data = np.array(x_data).flatten()
    y_data = np.array(y_data).flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color=color, alpha=0.7, edgecolor='black')
    plt.title(title, y=1.02)
    plt.figtext(0.5, 0.01, f"Run Name: {run_name}", ha="center", fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    scatterplot_path = os.path.join(plots_dir, filename)
    plt.savefig(scatterplot_path)
    plt.close()
    print(f"{title} saved to {scatterplot_path}")

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    writer=None
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, args.reward)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_lengths = []
    episodic_times = []
    episodic_lines_cleared = []

    cumulative_lines_cleared = 0

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            action, _, _, _ = model.get_action_and_value(obs_tensor)
        obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())

        if "lines_cleared" in infos:
            cumulative_lines_cleared += infos["lines_cleared"][0]

        if "final_info" in infos:
            for final_info in infos["final_info"]:
                if final_info is not None and "episode" in final_info:
                    episodic_lines_cleared.append(cumulative_lines_cleared)
                    cumulative_lines_cleared = 0 
                    episodic_returns.append(final_info["episode"]["r"])
                    episodic_lengths.append(final_info["episode"]["l"])
                    episodic_times.append(final_info["episode"]["t"])
                    print(f"Eval Episode Lines: {cumulative_lines_cleared}, Return: {final_info['episode']['r']}, Length: {final_info['episode']['l']}, Time: {final_info['episode']['t']}")

    if writer is not None:
        for idx, episodic_lines_clear in enumerate(episodic_lines_cleared):
            writer.add_scalar("eval/episodic_lines_cleared", episodic_lines_clear, idx)
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        for idx, episodic_length in enumerate(episodic_lengths):
            writer.add_scalar("eval/episodic_length", episodic_length, idx)
        for idx, episodic_time in enumerate(episodic_times):
            writer.add_scalar("eval/episodic_time", episodic_time, idx)

    envs.close()
    
    print_summary_statistics("Lines Cleared", episodic_lines_cleared)
    print_summary_statistics("Return", episodic_returns)
    print_summary_statistics("Length", episodic_lengths)
    print_summary_statistics("Time", episodic_times)

    plots_dir = os.path.join("plots", run_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    create_histogram(episodic_returns,"Histogram of Episodic Returns","Episodic Return","histogram_episodic_returns.png",run_name,plots_dir,'blue')
    create_histogram(episodic_lengths,"Histogram of Episodic Lengths","Episodic Length","histogram_episodic_lengths.png",run_name,plots_dir,'green')
    create_histogram(episodic_times,"Histogram of Episodic Times","Episodic Time (seconds)","histogram_episodic_times.png",run_name,plots_dir,'orange')
    create_histogram(episodic_lines_cleared,"Histogram of Lines Cleared","Lines Cleared","histogram_episodic_lines_cleared.png",run_name,plots_dir,'purple')

    create_violin_plot(episodic_returns,"Violin Plot of Episodic Returns","Episodic Return","violin_episodic_returns.png",run_name,plots_dir,'blue')
    create_violin_plot(episodic_lengths,"Violin Plot of Episodic Lengths","Episodic Length","violin_episodic_lengths.png",run_name,plots_dir,'green')
    create_violin_plot(episodic_times,"Violin Plot of Episodic Times","Episodic Time (seconds)","violin_episodic_times.png",run_name,plots_dir,'orange')
    create_violin_plot(episodic_lines_cleared,"Violin Plot of Lines Cleared","Lines Cleared","violin_episodic_lines_cleared.png",run_name,plots_dir,'purple')

    create_scatter_plot(episodic_lengths,episodic_returns,"Scatterplot of Episodic Lengths vs Returns","Episodic Length","Episodic Return","scatterplot_lengths_vs_returns.png",run_name,plots_dir,'blue')
    create_scatter_plot(episodic_lengths,episodic_lines_cleared,"Scatterplot of Episodic Lengths vs Lines Cleared","Episodic Length","Lines Cleared","scatterplot_lengths_vs_lines_cleared.png",run_name,plots_dir,'green')

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                idx=i,
                capture_video=args.capture_video,
                run_name=run_name,
                reward=args.reward
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    best_episodic_return = float("-inf")
    best_episodic_lines_cleared = float("-inf")
    global_step = 0 
    if args.load_model_path:
        print(f"Loading model from {args.load_model_path}")
        checkpoint = torch.load(args.load_model_path, weights_only=False)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_episodic_return = checkpoint.get("best_episodic_return", float("-inf"))
        best_episodic_lines_cleared = checkpoint.get("best_episodic_lines_cleared", float("-inf"))
        global_step = checkpoint.get("global_step", 0)


    if args.eval_only:
        print(f"Evaluating model from {args.eval_only} without training...")
        evaluate(
            model_path=args.eval_only,
            make_env=make_env,
            env_id=args.env_id,
            eval_episodes=100,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu"),
            capture_video=args.capture_video,
            writer=writer
        )
        exit(0)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    cumulative_lines_cleared = {i: 0 for i in range(args.num_envs)}

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)
            
            for i in range(args.num_envs):
                if "lines_cleared" in infos:
                    cumulative_lines_cleared[i] += infos["lines_cleared"][i]

                if "final_info" in infos and infos["final_info"][i] is not None:
                    final_info = infos["final_info"][i]
                    if isinstance(final_info, dict) and "episode" in final_info:
                        writer.add_scalar("charts/episodic_lines_cleared", cumulative_lines_cleared[i], global_step)
                        
                        if cumulative_lines_cleared[i] > best_episodic_lines_cleared:
                            best_episodic_lines_cleared = cumulative_lines_cleared[i]
                            writer.add_scalar("charts/best_episodic_lines_cleared", best_episodic_lines_cleared, global_step)

                        cumulative_lines_cleared[i] = 0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/episodic_time", info["episode"]["t"], global_step)

                        if episodic_return > best_episodic_return:
                            best_episodic_return = episodic_return
                            if args.save_model:
                                best_model_path = f"runs/{run_name}/{args.exp_name}_best.cleanrl_model"
                                torch.save(
                                    {
                                        "model_state_dict": agent.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "best_episodic_return": best_episodic_return,
                                        "best_episodic_lines_cleared": best_episodic_lines_cleared,
                                        "global_step": global_step,
                                    },
                                    best_model_path,
                                )
                                print(f"New best model saved to {best_model_path} with return {best_episodic_return}")
                            writer.add_scalar("charts/best_episodic_return", best_episodic_return, global_step)
                            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        #writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}_final.cleanrl_model"
        torch.save(
            {
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_episodic_return": best_episodic_return,
                "best_episodic_lines_cleared": best_episodic_lines_cleared,
                "global_step": global_step,
            },
            model_path,
        )
        print(f"Final model and optimizer saved to {model_path}")

        best_model_path = f"runs/{run_name}/{args.exp_name}_best.cleanrl_model"
        if os.path.exists(best_model_path):
            print(f"Evaluating the best model saved at {best_model_path}")
            evaluate(
                model_path=best_model_path,
                make_env=make_env,
                env_id=args.env_id,
                eval_episodes=100,
                run_name=f"{run_name}-best-eval",
                Model=Agent,
                device=device,
                capture_video=args.capture_video,
                writer=writer,
            )
        else:
            print(f"No best model found. Evaluating the final model saved at {model_path}")
            evaluate(
                model_path=model_path,
                make_env=make_env,
                env_id=args.env_id,
                eval_episodes=100,
                run_name=f"{run_name}-final-eval",
                Model=Agent,
                device=device,
                capture_video=args.capture_video,
                writer=writer,
            )

    envs.close()
    writer.close()