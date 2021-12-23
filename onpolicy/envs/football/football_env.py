import gym
import numpy as np
import gfootball.env as football_env


class FootballEnv(gym.Env):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    """

    def __init__(self, args):
        self._env = football_env.create_environment(
            env_name=args.scenario_name,
            representation=args.representation,
            number_of_left_players_agent_controls=args.num_agents,
            rewards=args.rewards)
        self.num_agents = args.num_agents
        self.action_space = [gym.spaces.Discrete(19) for _ in range(self.num_agents)]
        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(115,),
                dtype=self._env.observation_space.dtype
            ) for _ in range(self.num_agents)]
        self.share_observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(115 * self.num_agents,),
                dtype=self._env.observation_space.dtype
            ) for _ in range(self.num_agents)]

    def reset(self):
        obs_list = self._env.reset()
        return obs_list

    def step(self, onehot_actions):
        actions = [act.index(1.0) for act in onehot_actions.tolist()]
        obs, reward, done, info = self._env.step(actions)
        return obs, [[r] for r in reward], [done] * self.num_agents, info

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        self._env.render()
