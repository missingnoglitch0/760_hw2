import numpy as np

states = ['A', 'B']
actions = ['STAY', 'MOVE']

class QLearner():

    def __init__(self, steps_max=200, epsilon=0, alpha=1, gamma=0):
        self.epsilon = epsilon
        self.reset()
        self.steps_max = steps_max
        self.alpha = alpha
        self.gamma = gamma
        self.reset()
    
    def reset(self):
        self.Q = { s: [0 for a in actions] for s in states}

    def run_episode(self):
        state = 'A'
        for i in range(self.steps_max):
            action = self.epsilon_greedy_action(state)
            state_new = self.step(state, action)
            reward = 0
            if state == state_new:
                reward = 1
            print(f'Started at {state}, chose {actions[action]}, got reward {reward}')
            a_prime_opt = self.greedy_action(state_new)
            self.Q[state][action] = (1-self.alpha)*(self.Q[state][action]) + (self.alpha)*(reward + self.gamma * self.Q[state_new][a_prime_opt])
            state = state_new

    def step(self, state, action):
        if action == 1:
            if state == 'A':
                return 'B'
            else:
                return 'A'
        else:
            return state
            
    def greedy_action(self, state):
        # return np.argmax(self.Q[state])
        q_s = self.Q[state]
        if q_s[0] > q_s[1]:
            return 0
        elif q_s[1] > q_s[0]:
            return 1
        else:
            return 1

    def epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(actions))
        
        return self.greedy_action(state)

    def make_table_from_Q(self):
        return {s:{actions[0]: a[0], actions[1]: a[1]} for s, a in self.Q.items()}



Q_4_1 = QLearner(steps_max=200, epsilon=0, alpha=0.5, gamma=0.8)
Q_4_1.run_episode()

print(Q_4_1.make_table_from_Q())

Q_4_2 = QLearner(steps_max=200, epsilon=0.5, alpha=0.5, gamma=0.8)
Q_4_2.run_episode()

print(Q_4_2.make_table_from_Q())

