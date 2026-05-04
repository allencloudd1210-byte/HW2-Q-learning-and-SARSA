from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

ACTION_SYMBOLS = {
    0: "^",
    1: "v",
    2: "<",
    3: ">",
}


@dataclass
class ExperimentConfig:
    rows: int = 4
    cols: int = 12
    episodes: int = 500
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    max_steps_per_episode: int = 1_000
    runs: int = 30
    seed: int = 2026


class CliffWalkingEnv:
    def __init__(self, rows: int = 4, cols: int = 12) -> None:
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = {(rows - 1, col) for col in range(1, cols - 1)}

    def reset(self) -> tuple[int, int]:
        return self.start

    def step(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], int, bool, bool]:
        row, col = state
        d_row, d_col = ACTIONS[action]
        next_row = min(max(row + d_row, 0), self.rows - 1)
        next_col = min(max(col + d_col, 0), self.cols - 1)
        next_state = (next_row, next_col)

        reward = -1
        done = False
        fell_into_cliff = False

        if next_state in self.cliff:
            reward = -100
            next_state = self.start
            fell_into_cliff = True
        elif next_state == self.goal:
            done = True

        return next_state, reward, done, fell_into_cliff


def epsilon_greedy(
    q_table: np.ndarray,
    state: tuple[int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, 4))

    state_values = q_table[state[0], state[1]]
    best_actions = np.flatnonzero(state_values == np.max(state_values))
    return int(rng.choice(best_actions))


def greedy_action(q_table: np.ndarray, state: tuple[int, int]) -> int:
    state_values = q_table[state[0], state[1]]
    return int(np.argmax(state_values))


def train_q_learning(
    env: CliffWalkingEnv,
    config: ExperimentConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_table = np.zeros((config.rows, config.cols, 4), dtype=np.float64)
    episode_rewards = np.zeros(config.episodes, dtype=np.float64)
    episode_cliff_falls = np.zeros(config.episodes, dtype=np.int32)

    for episode in range(config.episodes):
        state = env.reset()
        total_reward = 0.0
        cliff_falls = 0

        for _ in range(config.max_steps_per_episode):
            action = epsilon_greedy(q_table, state, config.epsilon, rng)
            next_state, reward, done, fell_into_cliff = env.step(state, action)

            best_next_value = 0.0 if done else np.max(q_table[next_state[0], next_state[1]])
            td_target = reward + config.gamma * best_next_value
            td_error = td_target - q_table[state[0], state[1], action]
            q_table[state[0], state[1], action] += config.alpha * td_error

            total_reward += reward
            cliff_falls += int(fell_into_cliff)
            state = next_state

            if done:
                break

        episode_rewards[episode] = total_reward
        episode_cliff_falls[episode] = cliff_falls

    return q_table, episode_rewards, episode_cliff_falls


def train_sarsa(
    env: CliffWalkingEnv,
    config: ExperimentConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_table = np.zeros((config.rows, config.cols, 4), dtype=np.float64)
    episode_rewards = np.zeros(config.episodes, dtype=np.float64)
    episode_cliff_falls = np.zeros(config.episodes, dtype=np.int32)

    for episode in range(config.episodes):
        state = env.reset()
        action = epsilon_greedy(q_table, state, config.epsilon, rng)
        total_reward = 0.0
        cliff_falls = 0

        for _ in range(config.max_steps_per_episode):
            next_state, reward, done, fell_into_cliff = env.step(state, action)

            if done:
                td_target = reward
                td_error = td_target - q_table[state[0], state[1], action]
                q_table[state[0], state[1], action] += config.alpha * td_error
            else:
                next_action = epsilon_greedy(q_table, next_state, config.epsilon, rng)
                td_target = reward + config.gamma * q_table[next_state[0], next_state[1], next_action]
                td_error = td_target - q_table[state[0], state[1], action]
                q_table[state[0], state[1], action] += config.alpha * td_error
                state, action = next_state, next_action

            total_reward += reward
            cliff_falls += int(fell_into_cliff)

            if done:
                break

        episode_rewards[episode] = total_reward
        episode_cliff_falls[episode] = cliff_falls

    return q_table, episode_rewards, episode_cliff_falls


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def estimate_convergence_episode(
    rewards: np.ndarray,
    window: int = 20,
    tolerance: float = 5.0,
    streak: int = 10,
) -> int | None:
    if rewards.size < max(window, 50):
        return None

    reward_ma = moving_average(rewards, window)
    reference_reward = float(np.mean(rewards[-50:]))
    threshold = reference_reward - tolerance

    for start_idx in range(0, reward_ma.size - streak + 1):
        if np.all(reward_ma[start_idx:start_idx + streak] >= threshold):
            return start_idx + window

    return None


def extract_greedy_path(
    env: CliffWalkingEnv,
    q_table: np.ndarray,
    max_steps: int = 100,
) -> tuple[list[tuple[int, int]], bool]:
    state = env.start
    path = [state]
    visited_state_actions: set[tuple[tuple[int, int], int]] = set()

    for _ in range(max_steps):
        if state == env.goal:
            return path, True

        action = greedy_action(q_table, state)
        state_action = (state, action)
        if state_action in visited_state_actions:
            return path, False
        visited_state_actions.add(state_action)

        next_state, _, done, _ = env.step(state, action)
        path.append(next_state)
        state = next_state

        if done:
            return path, True

    return path, False


def policy_grid_text(env: CliffWalkingEnv, q_table: np.ndarray) -> str:
    lines: list[str] = []
    for row in range(env.rows):
        row_symbols: list[str] = []
        for col in range(env.cols):
            state = (row, col)
            if state == env.start:
                row_symbols.append("S")
            elif state == env.goal:
                row_symbols.append("G")
            elif state in env.cliff:
                row_symbols.append("C")
            else:
                row_symbols.append(ACTION_SYMBOLS[greedy_action(q_table, state)])
        lines.append(" ".join(row_symbols))
    return "\n".join(lines)


def path_grid_text(env: CliffWalkingEnv, path: list[tuple[int, int]]) -> str:
    grid = [["." for _ in range(env.cols)] for _ in range(env.rows)]
    for row, col in env.cliff:
        grid[row][col] = "C"
    start_row, start_col = env.start
    goal_row, goal_col = env.goal
    grid[start_row][start_col] = "S"
    grid[goal_row][goal_col] = "G"

    for row, col in path[1:-1]:
        if (row, col) not in env.cliff and (row, col) != env.goal:
            grid[row][col] = "*"

    return "\n".join(" ".join(row) for row in grid)


def draw_grid(ax: plt.Axes, env: CliffWalkingEnv, path: list[tuple[int, int]], title: str) -> None:
    color_map = np.ones((env.rows, env.cols, 3), dtype=np.float64)
    color_map[:] = np.array([0.98, 0.98, 0.98])

    for row, col in env.cliff:
        color_map[row, col] = np.array([0.1, 0.1, 0.1])

    start_row, start_col = env.start
    goal_row, goal_col = env.goal
    color_map[start_row, start_col] = np.array([0.70, 0.86, 0.76])
    color_map[goal_row, goal_col] = np.array([0.97, 0.91, 0.53])

    for row, col in path[1:-1]:
        if (row, col) not in env.cliff:
            color_map[row, col] = np.array([0.73, 0.85, 0.98])

    ax.imshow(color_map, origin="upper")

    for row in range(env.rows + 1):
        ax.axhline(row - 0.5, color="black", linewidth=0.8)
    for col in range(env.cols + 1):
        ax.axvline(col - 0.5, color="black", linewidth=0.8)

    xs = [col for _, col in path]
    ys = [row for row, _ in path]
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.0, marker="o", markersize=4)

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if state == env.start:
                label = "S"
            elif state == env.goal:
                label = "G"
            elif state in env.cliff:
                label = "C"
            else:
                label = ""
            ax.text(col, row, label, ha="center", va="center", fontsize=10, color="white" if label == "C" else "black")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title)
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)


def plot_reward_curves(
    output_path: Path,
    q_learning_rewards: np.ndarray,
    sarsa_rewards: np.ndarray,
    window: int = 20,
) -> None:
    episodes = np.arange(1, q_learning_rewards.shape[1] + 1)

    q_mean = q_learning_rewards.mean(axis=0)
    q_std = q_learning_rewards.std(axis=0)
    s_mean = sarsa_rewards.mean(axis=0)
    s_std = sarsa_rewards.std(axis=0)

    q_ma = moving_average(q_mean, window)
    s_ma = moving_average(s_mean, window)
    ma_episodes = np.arange(window, q_learning_rewards.shape[1] + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(episodes, q_mean, color="#1f77b4", alpha=0.25, label="Q-learning mean reward")
    ax.fill_between(episodes, q_mean - q_std, q_mean + q_std, color="#1f77b4", alpha=0.10)
    ax.plot(episodes, s_mean, color="#d62728", alpha=0.25, label="SARSA mean reward")
    ax.fill_between(episodes, s_mean - s_std, s_mean + s_std, color="#d62728", alpha=0.10)
    ax.plot(ma_episodes, q_ma, color="#1f77b4", linewidth=2.5, label=f"Q-learning {window}-episode moving average")
    ax.plot(ma_episodes, s_ma, color="#d62728", linewidth=2.5, label=f"SARSA {window}-episode moving average")
    ax.set_title("Cliff Walking: Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_paths(
    output_path: Path,
    env: CliffWalkingEnv,
    q_learning_path: list[tuple[int, int]],
    sarsa_path: list[tuple[int, int]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    draw_grid(axes[0], env, q_learning_path, "Q-learning Greedy Path")
    draw_grid(axes[1], env, sarsa_path, "SARSA Greedy Path")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_algorithm_summary(
    name: str,
    rewards: np.ndarray,
    cliff_falls: np.ndarray,
    q_table: np.ndarray,
    env: CliffWalkingEnv,
) -> dict[str, object]:
    reward_mean = rewards.mean(axis=0)
    path, reached_goal = extract_greedy_path(env, q_table)

    return {
        "name": name,
        "mean_reward_last_100": float(np.mean(reward_mean[-100:])),
        "std_reward_last_100": float(np.std(reward_mean[-100:])),
        "mean_cliff_falls_last_100": float(np.mean(cliff_falls.mean(axis=0)[-100:])),
        "overall_mean_reward": float(np.mean(reward_mean)),
        "best_episode_reward": float(np.max(reward_mean)),
        "convergence_episode": estimate_convergence_episode(reward_mean),
        "greedy_path_length": len(path) - 1,
        "greedy_path_reaches_goal": reached_goal,
        "policy_grid": policy_grid_text(env, q_table),
        "path_grid": path_grid_text(env, path),
        "path_coordinates": [list(point) for point in path],
    }


def run_experiment(config: ExperimentConfig, output_dir: Path) -> dict[str, object]:
    env = CliffWalkingEnv(rows=config.rows, cols=config.cols)

    q_learning_rewards: list[np.ndarray] = []
    q_learning_cliff_falls: list[np.ndarray] = []
    sarsa_rewards: list[np.ndarray] = []
    sarsa_cliff_falls: list[np.ndarray] = []

    for run_idx in range(config.runs):
        run_seed = config.seed + run_idx
        _, q_rewards, q_falls = train_q_learning(env, config, run_seed)
        _, s_rewards, s_falls = train_sarsa(env, config, run_seed)
        q_learning_rewards.append(q_rewards)
        q_learning_cliff_falls.append(q_falls)
        sarsa_rewards.append(s_rewards)
        sarsa_cliff_falls.append(s_falls)

    q_learning_rewards_array = np.stack(q_learning_rewards)
    q_learning_cliff_falls_array = np.stack(q_learning_cliff_falls)
    sarsa_rewards_array = np.stack(sarsa_rewards)
    sarsa_cliff_falls_array = np.stack(sarsa_cliff_falls)

    representative_q_table, _, _ = train_q_learning(env, config, config.seed)
    representative_sarsa_table, _, _ = train_sarsa(env, config, config.seed)

    q_path, _ = extract_greedy_path(env, representative_q_table)
    s_path, _ = extract_greedy_path(env, representative_sarsa_table)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_reward_curves(output_dir / "reward_curves.png", q_learning_rewards_array, sarsa_rewards_array)
    plot_paths(output_dir / "final_paths.png", env, q_path, s_path)

    q_summary = build_algorithm_summary(
        "Q-learning",
        q_learning_rewards_array,
        q_learning_cliff_falls_array,
        representative_q_table,
        env,
    )
    s_summary = build_algorithm_summary(
        "SARSA",
        sarsa_rewards_array,
        sarsa_cliff_falls_array,
        representative_sarsa_table,
        env,
    )

    faster_convergence = "Q-learning" if (
        q_summary["convergence_episode"] is not None and
        s_summary["convergence_episode"] is not None and
        q_summary["convergence_episode"] < s_summary["convergence_episode"]
    ) else "SARSA"

    more_stable = "Q-learning" if q_summary["std_reward_last_100"] < s_summary["std_reward_last_100"] else "SARSA"

    summary = {
        "config": asdict(config),
        "q_learning": q_summary,
        "sarsa": s_summary,
        "comparison": {
            "faster_convergence": faster_convergence,
            "more_stable": more_stable,
            "reward_curve": "Higher reward means better performance because rewards are negative step costs.",
            "artifacts": {
                "reward_curve_plot": str(output_dir / "reward_curves.png"),
                "path_plot": str(output_dir / "final_paths.png"),
            },
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Q-learning and SARSA on Cliff Walking.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent runs for averaging.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for epsilon-greedy.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to store plots and summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        runs=args.runs,
        seed=args.seed,
    )
    summary = run_experiment(config, args.output_dir)

    print("Experiment complete.")
    print(f"Reward curve: {summary['comparison']['artifacts']['reward_curve_plot']}")
    print(f"Path plot: {summary['comparison']['artifacts']['path_plot']}")
    print(f"Faster convergence: {summary['comparison']['faster_convergence']}")
    print(f"More stable: {summary['comparison']['more_stable']}")
    print(
        "Final 100-episode mean reward | "
        f"Q-learning: {summary['q_learning']['mean_reward_last_100']:.2f}, "
        f"SARSA: {summary['sarsa']['mean_reward_last_100']:.2f}"
    )


if __name__ == "__main__":
    main()
