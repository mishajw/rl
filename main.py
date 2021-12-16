import abc
import dataclasses
import multiprocessing
import random
from typing import List, Iterable

import streamlit as st
import seaborn as sns

import numpy as np
import pandas as pd

NUM_ITERATIONS = 2000
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
    estimates: np.array = None
    num_actions: np.array = None

    def __post_init__(self):
        if self.estimates is None:
            self.estimates = np.ones(NUM_ARMS)
        if self.num_actions is None:
            self.num_actions = np.ones(NUM_ARMS)

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
        return "greedy"


@dataclasses.dataclass()
class ConstStepSizeGreedy(Greedy):
    coefficient: float = 0.1

    def get_update_coefficient(self, _) -> float:
        return self.coefficient

    def name(self) -> str:
        return f"const greedy, c={self.coefficient}"


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


@dataclasses.dataclass(frozen=True)
class StepResult:
    agent_name: str
    action: int
    reward: float
    is_action_optimal: bool
    step: int
    iteration: int


def main():
    results = run_simulations()
    st.write(f"Found {len(results)} results")
    df = pd.DataFrame(results)
    df = (
        df.groupby(["agent_name", "step"])[["reward", "is_action_optimal"]]
        .mean()
        .reset_index()
    )

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


def run_simulations() -> List[StepResult]:
    results = []
    bar = st.progress(0.0)

    def run_random_simulation(iteration: int) -> List[StepResult]:
        environment = NonStationaryEnvironment.random()
        agents = [
            Greedy(),
            EpsilonGreedy(Greedy(), epsilon=0.1),
            EpsilonGreedy(Greedy(), epsilon=0.01),
            EpsilonGreedy(ConstStepSizeGreedy(coefficient=0.1), epsilon=0.1),
        ]
        return list(run_simulation(environment, agents, iteration))

    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        iteration_results = pool.imap_unordered(
            run_random_simulation, range(NUM_ITERATIONS)
        )
        for i, step_results in enumerate(iteration_results):
            results.extend(step_results)
            bar.progress(i / NUM_ITERATIONS)
    bar.progress(1.0)
    return results


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
