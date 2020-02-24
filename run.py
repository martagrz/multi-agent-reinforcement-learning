import numpy as np
from env import NGridworld
from nashq import NashQLearning

n_players = 2
n_rows = 4
n_cols = 4
init_states = np.array(((0,0), (1,1)))
goal_states = np.array(((2,0), (2,2)))

alpha = 0.8
gamma = 0.6
epsilon = 0.2
episodes = 1000

env = NGridworld(n_players, n_rows, n_cols, init_states, goal_states)
algo = NashQLearning(env, alpha, gamma, epsilon)
q_table = algo.train(episodes)
print(algo.q_tables[0][0])
print(algo.q_tables[1][1])

print(np.argmax(q_table[0][0][0]))




action_path, rewards_list = algo.evaluate()

print(action_path)
print(rewards_list)




