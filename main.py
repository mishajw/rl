import abc
import dataclasses
import multiprocessing
import random
from typing import List, Iterable

import streamlit as st
import seaborn as sns

import numpy as np
import pandas as pd

NUM_ITERATIONS = 200
NUM_STEPS = 1000
NUM_ARMS = 10
NUM_PROCESSES = 64


class Environment(abc.ABC):
    @abc.abstractmethod
    def get_reward(self, action: int) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_optimal_action(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self):
        raise NotImplementedError()


@dataclasses.dataclass
class StaticEnvironment(Environment):
    arm_means: np.array
    optimal_action: int

    @classmethod
    def random(cls) -> "Environment":
        arm_means = np.random.normal(size=NUM_ARMS)
        return StaticEnvironment(
            arm_means=arm_means, optimal_action=np.argmax(arm_means)
        )

    def get_reward(self, action: int) -> float:
        return random.gauss(self.arm_means[action], 1)

    def get_optimal_action(self) -> int:
        return self.optimal_action

    def update(self):
        pass


@dataclasses.dataclass
class NonStationaryEnvironment(Environment):
    arm_means: np.array
    arm_trajectories: np.array

    @classmethod
    def random(cls) -> "NonStationaryEnvironment":
        return NonStationaryEnvironment(
            arm_means=np.random.normal(size=NUM_ARMS),
            arm_trajectories=np.random.normal(scale=0.01, size=NUM_ARMS),
        )

    def get_reward(self, action: int) -> float:
        return random.gauss(self.arm_means[action], 1)

    def get_optimal_action(self) -> float:
        return np.argmax(self.arm_means)

    def update(self):
        self.arm_means += self.arm_trajectories


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_action(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, action: int, reward: float):
        raise NotImplementedError()

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass()
class Greedy(Agent):
    initial_value: float = 0
    estimates: np.array = None
    num_actions: np.array = None

    def __post_init__(self):
        self.estimates = np.ones(NUM_ARMS) * self.initial_value
        self.num_actions = np.zeros(NUM_ARMS)

    def get_action(self) -> int:
        return np.argmax(self.estimates)

    def update(self, action: int, reward: float):
        if self.num_actions[action] == 0:
            self.estimates[action] = reward
        else:
            error = reward - self.estimates[action]
            self.estimates[action] += (
                self.get_update_coefficient(action) * error
            )
        self.num_actions[action] += 1

    def get_update_coefficient(self, action) -> float:
        return 1 / self.num_actions[action]

    def name(self) -> str:
        return f"greedy, i={self.initial_value}"


@dataclasses.dataclass()
class ConstStepSizeGreedy(Greedy):
    coefficient: float = 0.1

    def get_update_coefficient(self, _) -> float:
        return self.coefficient

    def name(self) -> str:
        return f"const {super().name()}, c={self.coefficient}"


@dataclasses.dataclass()
class EpsilonGreedy(Agent):
    greedy: Greedy
    epsilon: float

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ARMS - 1)
        return self.greedy.get_action()

    def update(self, action: int, reward: float):
        self.greedy.update(action, reward)

    def name(self) -> str:
        return f"epsilon {self.greedy.name()}, e={self.epsilon}"


@dataclasses.dataclass()
class UpperConfidenceBound(Greedy):
    coefficient: float = 1

    def get_action(self) -> int:
        if np.sum(self.num_actions) == 0:
            return 0
        # TODO: Remove +0.01 hack.
        return np.argmax(
            self.estimates
            + self.coefficient
            * np.sqrt(
                np.log(np.sum(self.num_actions)) / (self.num_actions + 0.01)
            )
        )

    def name(self) -> str:
        return f"ucb {super().name()}, c={self.coefficient}"


@dataclasses.dataclass()
class Gradient(Agent):
    learning_rate: float = 0.1
    preferences: np.array = np.empty(0)
    reward_sums: float = 0
    num_rewards: int = 0

    def __post_init__(self):
        self.preferences = np.ones(NUM_ARMS)

    def get_action(self) -> int:
        return np.random.choice(NUM_ARMS, p=self._get_probabilities())

    def update(self, action: int, reward: float):
        reward_mean = (
            self.reward_sums / self.num_rewards if self.num_rewards > 0 else 1
        )

        self.preferences += (
            self.learning_rate
            * (reward - reward_mean)
            * ((np.arange(0, NUM_ARMS) == action) - self._get_probabilities())
        )
        self.reward_sums += reward
        self.num_rewards += 1

    def _get_probabilities(self) -> np.array:
        preferences_exp = np.exp(self.preferences)
        return preferences_exp / np.sum(preferences_exp)

    def name(self) -> str:
        return f"gradient, lr={self.learning_rate}"


@dataclasses.dataclass(frozen=True)
class StepResult:
    agent_name: str
    action: int
    reward: float
    is_action_optimal: bool
    step: int
    iteration: int


def main():
    df = run_simulations()
    st.write(f"Found {len(df)} results")
    df["step"] = (df["step"] // 10) * 10
    st.write(f"Joined to {len(df)} rows")

    sns.lineplot(
        data=df,
        x="step",
        y="reward",
        hue="agent_name",
    )
    st.pyplot()
    sns.lineplot(
        data=df,
        x="step",
        y="is_action_optimal",
        hue="agent_name",
    )
    st.pyplot()
    st.write("Done")


@st.cache(suppress_st_warning=True)
def run_simulations() -> pd.DataFrame:
    results = []
    bar = st.progress(0.0)

    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        iteration_results = pool.imap_unordered(
            run_random_simulation, range(NUM_ITERATIONS)
        )
        for i, step_results in enumerate(iteration_results):
            results.extend(step_results)
            bar.progress(i / NUM_ITERATIONS)
    bar.progress(1.0)
    st.write("Finished, caching results")
    return pd.DataFrame(results)


def run_random_simulation(iteration: int) -> List[StepResult]:
    # environment = NonStationaryEnvironment.random()
    environment = StaticEnvironment.random()
    agents = [
        Greedy(),
        Greedy(initial_value=3),
        EpsilonGreedy(Greedy(), epsilon=0.1),
        EpsilonGreedy(Greedy(), epsilon=0.01),
        UpperConfidenceBound(coefficient=0.8),
        EpsilonGreedy(ConstStepSizeGreedy(coefficient=0.1), epsilon=0.1),
        Gradient(learning_rate=0.1),
        Gradient(learning_rate=0.05),
        Gradient(learning_rate=0.01),
    ]
    return list(run_simulation(environment, agents, iteration))


def run_simulation(
    environment: Environment, agents: List[Agent], iteration: int
) -> Iterable[StepResult]:
    for step in range(NUM_STEPS):
        optimal_action = environment.get_optimal_action()
        for agent in agents:
            action = agent.get_action()
            reward = environment.get_reward(action)
            agent.update(action, reward)
            yield StepResult(
                agent_name=agent.name(),
                action=action,
                reward=reward,
                is_action_optimal=action == optimal_action,
                step=step,
                iteration=iteration,
            )
        environment.update()


if __name__ == "__main__":
    st.set_option("deprecation.showPyplotGlobalUse", False)
    main()
