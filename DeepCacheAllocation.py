import numpy as np

import gym
from gym import spaces

MAX_CACHE_CAPACITY = 100  # cache capacity - limited to 100


class DeepCacheNetw(gym.Env):
    """Deep QN Environment for Cache Allocation"""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_x1, max_cost_SP1, max_cost_SP2):
        super(DeepCacheNetw, self).__init__()
        # fixed allocation of cache resources to SPs
        self.initial_x1 = initial_x1
        self.x1 = initial_x1  # SP1
        self.x2 = MAX_CACHE_CAPACITY - initial_x1  # SP2
        self.max_cost_SP1 = max_cost_SP1
        self.max_cost_SP2 = max_cost_SP2
        self.reward_range = (-max_cost_SP1 - max_cost_SP2, 0)  # -20 the worst case of all requests
        self.action_space = spaces.Discrete(3, )  # -1 0 1 action space range

        print("x1 ", self.x1)
        print("x2 ", self.x2)

        # The range in which x1 can take value
        self.observation_space = spaces.Box(
            low=self.reward_range[0], high=self.reward_range[1], shape=(1, MAX_CACHE_CAPACITY), dtype=np.float16)

    # first count the cost, and then calculate the reward (that is the opposite of the cost)
    def step(self, action_before_conversion):

        print(" ")
        action = 2 * (action_before_conversion - 1)

        # initialization
        penalty = 0

        print("x1 + action", self.x1 + action)

        # if the value of x1 + *action it takes* is negative or over the max_cache_capacity it gains a penalty
        if self.x1 + action < 0 or self.x2 - action < 0 or self.x1 + action > MAX_CACHE_CAPACITY or self.x2 - action > MAX_CACHE_CAPACITY:
            penalty = (self.max_cost_SP1 + self.max_cost_SP2)
            print("illegal action ", action)
            print("x1 + action: ", self.x1 + action)
            print("x2 - action: ", self.x2 - action)
            self.x1 = self.initial_x1
            self.x2 = MAX_CACHE_CAPACITY - self.x1
        elif self.x1 > 50:
            self.x1 -= action
            self.x2 += action
            print("legal action ", action)

        # Now, let's calculate the reward
        # rand_reqSP1 = np.random.randint(low=1, high=100, size=self.max_cost_SP1)  # random requests of sp1 (numbers from 1 to 100)
        # rand_reqSP2 = np.random.randint(low=1, high=100, size=self.max_cost_SP2)  # random requests of sp2 (numbers from 1 to 200)

        # try lognormal
        np.random.seed(10)
        rand_reqSP1 = np.random.zipf(2.5, self.max_cost_SP1)
        rand_reqSP2 = np.random.zipf(1.2, self.max_cost_SP2)

        # everytime the requests is not in cache we increase the size of the instantaneous cost
        inst_cost = 0

        for req in rand_reqSP1:
            if req > self.x1:
                inst_cost += 1

        for req in rand_reqSP2:
            if req > self.x2:
                inst_cost += 1

        reward = -inst_cost - penalty

        observation = self.x1

        print('x1, reward , x2 = ', observation, reward, self.x2)  # print for every step x1 and the reward

        # necessary parameter to return
        done = False

        return observation, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.x1 = self.initial_x1
        self.x2 = MAX_CACHE_CAPACITY - self.x1

        observation = self.x1

        return observation
