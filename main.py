import abc
import dataclasses
import random
import streamlit as st
import seaborn as sns

import numpy as np
import pandas as pd

NUM_ITERATIONS = 2000
NUM_STEPS = 1000
NUM_ARMS = 10


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
        arm_means = [random.gauss(0, 1) for _ in range(NUM_ARMS)]
        return StaticEnvironment(arm_means=arm_means, optimal_action=np.argmax(arm_means))

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
            arm_means=[random.gauss(0, 1) for _ in range(NUM_ARMS)],
            arm_trajectories=[random.gauss(0, 0.01) for _ in range(NUM_ARMS)],
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
    def title(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass()
class Greedy(Agent):
    estimates: np.array
    num_actions: np.array

    @classmethod
    def create(cls) -> "Greedy":
        return Greedy(estimates=np.ones(NUM_ARMS), num_actions=np.ones(NUM_ARMS))

    def get_action(self) -> int:
        return np.argmax(self.estimates)

    def update(self, action: int, reward: float):
        if self.num_actions[action] == 0:
            self.estimates[action] = reward
        else:
            self.estimates[action] += (1 / self.num_actions[action]) * (
                reward - self.estimates[action]
            )
        self.num_actions[action] += 1

    def title(self) -> str:
        return "greedy"


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

    def title(self) -> str:
        return f"epsilon greedy ({self.epsilon})"


def main():
    df = run_simulations()
    st.write(f"Found {len(df)} results")
    df = df.groupby(["agent", "step"])[["reward", "is_action_optimal"]].mean().reset_index()

    sns.lineplot(
        data=df,
        x="step",
        y="reward",
        hue="agent",
    )
    st.pyplot()
    sns.lineplot(
        data=df,
        x="step",
        y="is_action_optimal",
        hue="agent",
    )
    st.pyplot()
    st.write("Done")


@st.cache(suppress_st_warning=True)
def run_simulations() -> pd.DataFrame:
    results = []
    bar = st.progress(0.0)
    for iteration in range(NUM_ITERATIONS):
        bar.progress(iteration / NUM_ITERATIONS)
        environment = NonStationaryEnvironment.random()
        agents = [
            Greedy.create(),
            EpsilonGreedy(Greedy.create(), epsilon=0.01),
            EpsilonGreedy(Greedy.create(), epsilon=0.1),
        ]

        for step in range(NUM_STEPS):
            optimal_action = environment.get_optimal_action()
            for agent in agents:
                action = agent.get_action()
                reward = environment.get_reward(action)
                agent.update(action, reward)
                results.append(
                    dict(
                        agent=agent.title(),
                        action=action,
                        reward=reward,
                        step=step,
                        iteration=iteration,
                        is_action_optimal=action == optimal_action,
                    )
                )
            environment.update()
    bar.progress(1.0)
    return pd.DataFrame(results)


if __name__ == "__main__":
    st.set_option("deprecation.showPyplotGlobalUse", False)
    main()
