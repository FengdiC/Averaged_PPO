from gym.core import Env
from gym import spaces
import torch
import numpy as np
import itertools

class DotReacher(Env):
    """
    Action space, Discrete(8)
    Observation space, Box(2), positions
    """
    def __init__(self,stepsize=0.4,timeout=100):
        # actions: up down left right
        # actions: upleft upright downleft downright
        self._aval = stepsize * np.array([[0,1], [0,-1], [-1, 0], [1,0], [-1,1], [1,1], [-1, -1], [1,-1]
                                            ], dtype=np.float32)

        states = np.arange(0.0,1.0,stepsize)
        states = np.concatenate((np.arange(-states[-1],0.0,stepsize),states))
        print(states)
        states = states.tolist()
        self.num_pt = len(states)
        self._states = list(itertools.product(states,states))
        self._obs = np.array(self._states,dtype=np.float32)

        self._pos_tol = stepsize/2
        self._LB = np.array([-states[-1], -states[-1]])
        self._UB = np.array([states[-1], states[-1]])
        self._timeout = timeout
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.pos = self._obs[np.random.randint(0,self.num_pt**2)]
        obs = self.pos
        return obs

    def step(self,action):
        self.steps += 1
        self.pos = np.clip(self.pos + self._aval[int(action-1)] ,self._LB, self._UB)
        next_obs = self.pos
        # Reward
        reward = -0.01
        # Done
        done = np.allclose(self.pos, np.zeros(2), atol=self._pos_tol)
        done = done or self.steps == self._timeout
        # Metadata
        info = {}
        return next_obs, reward, done, info

    @property
    def action_space(self):
        return spaces.Discrete(8)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))

    def transition_matrix(self,policy):
        """
        Label states by self._states
        Inputs: policy is a numpy of shape 25 x 8, containing softmax policies for each state
        Outputs: transition matrix 25 x 25
        """

        # find out the next state with the state and the action
        next_state = []
        states = [[round(key,2) for key in item] for item in self._states]
        for i in range(len(self._states)):
            obs = self._obs[i]
            next = np.clip(self._aval + obs, self._LB, self._UB)
            next = np.around(next,2)
            next_idx = [states.index(next[i].tolist()) for i in range(8)]
            next_state.append(next_idx)

        next_state = np.array(next_state).astype(np.int32)
        # this matrix of size 25 x 8 contains the indices of next states under all 8 actions
        P = np.zeros((self.num_pt**2,self.num_pt**2))
        for i in range(len(self._states)):
            for j in range(8):
                P[i,next_state[i,j]] += policy[i,j]

        terminal_idx = states.index([0,0])
        P[terminal_idx, :] = np.ones(self.num_pt ** 2) / float(self.num_pt ** 2)
        return P

    def expected_reward(self):
        """
        Label states by self._states
        Inputs: policy is a numpy of shape 25 x 8, containing softmax policies for each state
        Outputs: expected reward of shape 25,
        """
        states = [[round(key,2) for key in item] for item in self._states]
        r = -0.01*  np.ones(self.num_pt**2)

        terminal_idx = states.index([0,0])
        r[terminal_idx] = 0
        return r

    def get_states(self):
        return self._obs

    def q_values(self,policy,gamma):
        """Variables: v is a vector of size 25
        Q is a matrix of size 25*8
        """
        P = self.transition_matrix(policy)
        r = self.expected_reward()
        v = np.matmul( np.linalg.inv(np.eye(self.num_pt**2)-gamma*P), r)
        next_value = np.matmul(P,v)
        next_value = np.transpose(np.tile(next_value,(8,1)))
        Q = -0.01+ next_value

        states = [[round(key, 2) for key in item] for item in self._states]
        terminal_idx = states.index([0, 0])
        r[terminal_idx] = 0
        Q[terminal_idx,:] +=0.01
        return Q

class DotReacherRepeat(Env):
    """
    Action space, Discrete(8)
    Observation space, Box(2), positions
    """
    def __init__(self,stepsize=0.4,timeout=500):
        # actions: up down left right
        # actions: upleft upright downleft downright
        self._aval = stepsize * np.array([[0,1], [0,-1], [-1, 0], [1,0], [-1,1], [1,1], [-1, -1], [1,-1]
                                            ], dtype=np.float32)

        states = np.arange(0.0,1.0,stepsize)
        states = np.concatenate((np.arange(-states[-1],0.0,stepsize),states))
        print(states)
        states = states.tolist()
        self.num_pt = len(states)
        self._states = list(itertools.product(states,states))
        self._obs = np.array(self._states,dtype=np.float32)

        self._pos_tol = stepsize/2
        self._LB = np.array([-states[-1], -states[-1]])
        self._UB = np.array([states[-1], states[-1]])
        self._timeout = timeout
        self.steps = 0

    def reset(self):
        self.steps = 0
        # self.pos = self._obs[np.random.randint(0,self.num_pt**2)]
        self.pos = self._obs[0]
        obs = self.pos
        return obs

    def _restart(self):
        self.pos = self._obs[0]
        # self.pos = self._obs[np.random.randint(0, self.num_pt ** 2)]
        obs = self.pos
        return obs

    def step(self,action):
        self.steps += 1
        done = np.allclose(self.pos, np.zeros(2), atol=self._pos_tol)
        if done:
            next_obs = self._restart()
            reward = -0.01
        else:
            self.pos = np.clip(self.pos + self._aval[int(action-1)] ,self._LB, self._UB)
            next_obs = self.pos
            # Reward
            reward = -0.01
        # Reach goal
        done = np.allclose(self.pos, np.array([0.8,-0.8]), atol=self._pos_tol)
        if done:
            reward = 1

        # Check termiation
        done = self.steps == self._timeout
        # Metadata
        info = {}
        return next_obs, reward, done, info

    @property
    def action_space(self):
        return spaces.Discrete(8)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))

    def q_values(self,policy,gamma):
        """Variables: v is a vector of size 25
        Q is a matrix of size 25*8
        """
        P = self.transition_matrix(policy)
        r = self.expected_reward()
        v = np.matmul( np.linalg.inv(np.eye(self.num_pt**2)-gamma*P), r)
        next_value = np.matmul(P,v)
        next_value = np.transpose(np.tile(next_value,(8,1)))
        Q = -0.01+ next_value

        states = [[round(key, 2) for key in item] for item in self._states]
        terminal_idx = states.index([0, 0])
        r[terminal_idx] = 0
        Q[terminal_idx,:] +=0.01
        return Q

    def expected_reward(self):
        """
        Label states by self._states
        Inputs: policy is a numpy of shape 25 x 8, containing softmax policies for each state
        Outputs: expected reward of shape 25,
        """
        states = [[round(key,2) for key in item] for item in self._states]
        r = -0.01*  np.ones(self.num_pt**2)

        terminal_idx = states.index([0,0])
        r[terminal_idx] = 0
        return r

    def transition_matrix(self,policy):
        """
        Label states by self._states
        Inputs: policy is a numpy of shape 25 x 8, containing softmax policies for each state
        Outputs: transition matrix 25 x 25
        """

        # find out the next state with the state and the action
        next_state = []
        states = [[round(key,2) for key in item] for item in self._states]
        for i in range(len(self._states)):
            obs = self._obs[i]
            next = np.clip(self._aval + obs, self._LB, self._UB)
            next = np.around(next,2)
            next_idx = [states.index(next[i].tolist()) for i in range(8)]
            next_state.append(next_idx)

        next_state = np.array(next_state).astype(np.int32)
        # this matrix of size 25 x 8 contains the indices of next states under all 8 actions
        P = np.zeros((self.num_pt**2,self.num_pt**2))
        for i in range(len(self._states)):
            for j in range(8):
                P[i,next_state[i,j]] += policy[i,j]

        terminal_idx = states.index([0,0])
        P[terminal_idx,:] = np.ones(self.num_pt**2) / float(self.num_pt**2)
        return P

    def get_states(self):
        return self._obs