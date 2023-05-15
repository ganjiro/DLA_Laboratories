import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gym_navigation.memory.replay_memory import ReplayMemory, Transition
import numpy as np
import math
import random


class DQLN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_sizes):
        super(DQLN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQL:
    def __init__(self, hidden_sizes, batch_size, eps_decay, path):

        self.eps = 1  # the starting value of epsilon
        self.eps_decay = eps_decay  # controls the rate of exponential decay of epsilon, higher means a slower decay
        self.min_eps = 0.01
        self.gamma = 0.99  # Discount Factor
        self.batch_size = batch_size  # is the number of transitions random sampled from the replay buffer
        self.steps_done = 0
        self.hidden_sizes = hidden_sizes
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
        self.max_distance = 15.0
        self.min_distance = 0.2
        self.writer = SummaryWriter("../dql/runs")
        self.path = path

        self.env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
        self.env.action_space.seed(42)  # 42
        state_observation, info = self.env.reset(seed=42)

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        # Get the number of state observations
        n_observations = len(state_observation)

        # Instantiate policy network.
        self.q_function_1 = DQLN(n_observations, n_actions, hidden_sizes).to(self.device)
        self.q_function_2 = DQLN(n_observations, n_actions, hidden_sizes).to(self.device)

    def _normalize(self, state):
        nornmalized_state = np.ndarray(shape=state.shape, dtype=np.float64)
        for i in range(len(state)):
            if i < 17:
                nornmalized_state[i] = (state[i] - self.min_distance) / (self.max_distance - self.min_distance)
            else:
                nornmalized_state[i] = state[i] / math.pi
        return nornmalized_state

    # Epsilon-greedy action sampling.
    def _select_action_epsilon(self, state):
        sample = random.random()

        if sample > self.eps:
            with torch.no_grad():
                # return index of action with the best Q value in the current state

                if random.random() < 0.5:
                    action = self.q_function_1(state).max(1)[1].view(1, 1)
                else:
                    action = self.q_function_2(state).max(1)[1].view(1, 1)

                return action
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    # Do one batch of gradient descent.
    def _optimize_model(self, ):
        # Make sure we have enough samples in replay buffer.
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample uniformly from replay buffer.
        transitions = self.replay_buffer.sample(self.batch_size)

        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Concatenate into tensors for batch update.
        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.Tensor(batch.done).to(self.device)

        if random.random() < 0.5:
            q_values = self.q_function_1(state_batch)
            state_action_values = q_values.gather(1, action_batch)
            actions = q_values.max(1)[1]
            with torch.no_grad():
                target_q = torch.gather(self.q_function_2(next_state_batch), 1, actions.unsqueeze(1)).squeeze(1)

                expected_state_action_values = reward_batch + self.gamma * (1 - done_batch) * target_q

            loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer_1.zero_grad()
            loss.backward()

            self.optimizer_1.step()
        else:
            q_values = self.q_function_2(state_batch)
            state_action_values = q_values.gather(1, action_batch)
            actions = q_values.max(1)[1]
            with torch.no_grad():
                target_q = torch.gather(self.q_function_1(next_state_batch), 1, actions.unsqueeze(1)).squeeze(1)
                expected_state_action_values = reward_batch + self.gamma * (1 - done_batch) * target_q

            loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer_2.zero_grad()
            loss.backward()

            # writer.add_scalars('loss', {'policy_net': loss}, i_episode)
            self.optimizer_2.step()

    def _run_validation(self, env, policy_net, num=10):
        running_rewards = [0.0] * num
        for i in range(num):
            state_observation, info = env.reset()
            while True:
                state_observation = self._normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32, device=self.device).unsqueeze(
                    0)

                action = policy_net(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, _ = env.step(action.item())
                running_rewards[i] += reward

                if terminated or truncated:
                    break
        return running_rewards

    def train(self, lr, replay_memory_len, num_episodes):

        self.optimizer_1 = optim.Adam(self.q_function_1.parameters(), lr=lr, weight_decay=1e-5)
        self.optimizer_2 = optim.Adam(self.q_function_2.parameters(), lr=lr, weight_decay=1e-5)

        self.replay_buffer = ReplayMemory(replay_memory_len)

        print("START Deep Q-Learning Navigation Goal")

        # Sample experience, save in Replay Buffer.
        for i_episode in range(0, num_episodes, 1):

            state_observation, info = self.env.reset()
            state_observation = self._normalize(state_observation)
            state_observation = torch.tensor(state_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            steps = 0

            # Run one episode.
            while True:
                action = self._select_action_epsilon(state_observation)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())

                observation = self._normalize(observation)  # Normalize in [0,1]

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.replay_buffer.push(state_observation, action, next_state, reward, done)

                # Move to the next state
                state_observation = next_state
                steps += 1

                if done:
                    break

            # Perform one step of the optimization (on the policy network)
            self._optimize_model()

            # Epsilon decay
            EPS = max(self.min_eps, self.eps * self.eps_decay)

            # Every 50 episodes, validate.
            if not i_episode % 100:
                print("Episode: ", i_episode)
                rewards = self._run_validation(self.env, self.q_function_1)

                self.writer.add_scalar("Reward", np.mean(rewards), i_episode)
                self.writer.add_scalar('Epsilon', EPS, i_episode)

        torch.save(self.q_function_1.state_dict(), self.path)
        self.writer.close()
        self.env.close()
        print('COMPLETE')

    def test(self, n_episodies, render_mode=None):
        env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=render_mode, track_id=1)

        env.action_space.seed(42)
        state_observation, info = env.reset(seed=42)

        not_terminated = 0
        success = 0

        for _ in range(n_episodies):
            steps = 0
            while True:
                state_observation = self._normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32, device=self.device).unsqueeze(
                    0)
                action = self.q_function_1(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, info = env.step(action.item())
                steps += 1

                if truncated:
                    not_terminated += 1

                if terminated or truncated:
                    if not truncated and reward == 500:
                        success += 1
                    state_observation, info = env.reset()
                    break

        env.close()
        print("Executed " + str(n_episodies) + " episodes:\n" + str(success) + " successes\n" + str(
            not_terminated) + " episodes not terminated\n" + str(
            n_episodies - (success + not_terminated)) + " failures\n")
