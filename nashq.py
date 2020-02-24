import numpy as np
from tqdm import tqdm

class NashQLearning():
    def __init__(self, env, alpha, gamma, epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.q_tables = np.ones((self.env.n_players, self.env.n_players, self.env.grid_size, len(self.env.actions)))

    def get_index(self, state):
        row = state[0]
        col = state[1]
        index = np.int(row * self.env.n_cols + col)
        return index

    def choose_action(self, player, current_state_index, greedy=False):
        if greedy:
            action = np.argmax(self.q_tables[player, player, current_state_index])
        else:
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.choice(self.env.actions, 1)
            else:
                action = np.argmax(self.q_tables[player, player, current_state_index])
        return action

    def get_policies(self, player, state):
        policies = 1
        for other_player in range(self.env.n_players):
            state_index = self.get_index(state[other_player])
            policy = np.argmax(self.q_tables[player, other_player, state_index])
            policies *= policy
        return policies

    def train(self, episodes):
        for _ in tqdm(range(episodes)):
            state = self.env.reset()
            actions = np.zeros(self.env.n_players, dtype=int)
            done = np.zeros(self.env.n_players)
            actions_list = []
            states_list = []
            i = 0
            cumulative_rewards = np.zeros(self.env.n_players)
            while not done.all():
                for player in range(self.env.n_players):
                    current_state_index = self.get_index(state[player])
                    if not done[player]:
                        actions[player] = np.int(self.choose_action(player, current_state_index))
                    else:
                        actions[player] = 4 #stays in the same place

                next_state, rewards, done = self.env.step(state, actions)
                #print('action', actions, 'rewards', rewards, 'state', state)
                actions_list.append(actions)
                states_list.append(state)

                for n in range(self.env.n_players):
                    policies = self.get_policies(n, next_state)
                    if not done[n]:
                        for m in range(self.env.n_players):
                            current_state_index = self.get_index(state[m])
                            next_state_index = self.get_index(next_state[m])
                            next_state_q = np.max(self.q_tables[n, m, next_state_index])
                            #print(policies, next_state_q) #next_state_q explodes to infinity
                            nash_q = policies * next_state_q
                            #print(nash_q)
                            if np.isnan(nash_q):
                                print(policies, next_state_q, nash_q)
                                print(states_list, actions_list)
                            assert not np.isnan(nash_q)
                            value = (1 - self.alpha) * self.q_tables[n, m, current_state_index, actions[m]] + self.alpha * (rewards[m] + self.gamma * nash_q)
                            self.q_tables[n, m, current_state_index, actions[m]] = value
                    else:
                        pass

                state = next_state
                cumulative_rewards += rewards
                i += 1

                #print(np.all(done))

                #if i % 10 == 0:
                #    print('\n', 'Episode: ', _, 'Step: ', i, 'Rewards: ', cumulative_rewards)

            if _ % 1 == 0:
                print('\n', 'Episode', _, 'converged.')
            print(done)

        return self.q_tables


    def evaluate(self):
        state = self.env.reset()
        actions = np.zeros(self.env.n_players, dtype=int)
        action_path = []
        rewards_list = []
        done = np.zeros(self.env.n_players)
        while not np.all(done):
            for player in range(self.env.n_players):
                current_state_index = self.get_index(state[player])
                if not done[player]:
                    actions[player] = self.choose_action(player, current_state_index, greedy=True)
                else:
                    actions[player] = 4  # stays in the same place

            #print('actions', actions, 'state', state)

            next_state, rewards, done = self.env.step(state, actions)
            print(state)
            #print(state, actions, done)

            state = next_state
            action_path.append(actions)
            rewards_list.append(rewards)

        return action_path, rewards_list



