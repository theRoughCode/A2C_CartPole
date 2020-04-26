import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ACNet(nn.Module):
    # Currently, both actor and critic use the same model
    def __init__(self, input_dim, output_dim):
        super(ACNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        return value


class A2CAgent():

    def __init__(self, env, gamma=0.99, lr=7e-3, value_coeff=0.5, entropy_coeff=0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = gamma
        self.lr = lr

        # Coefficients used for loss term
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.value_network = ACNet(self.obs_dim, 1).to(self.device)                 # Critic
        self.policy_network = ACNet(self.obs_dim, self.action_dim).to(self.device)  # Actor

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.lr)

    def get_action(self, state):
        # Get action from Actor network
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy_network.forward(state)
        # Get prob distribution
        dist = F.softmax(logits, dim=0)
        # Create distribution for sampling
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()

    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0]
                                    for sars in trajectory]).to(self.device)
        actions = torch.LongTensor(
            [sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2]
                                     for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor(
            [sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor(
            [sars[4] for sars in trajectory]).view(-1, 1).to(self.device)

        # compute value target
        # Discounted rewards are calculated as discounted sum of future rewards.
        # If done, include only final reward
        discounted_rewards = torch.zeros_like(rewards)
        for j in reversed(range(rewards.size(0) - 1)):
            discounted_rewards[j] = rewards[j] + self.gamma * \
                discounted_rewards[j + 1] * (1 - dones[j])
            # returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        # discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]) \
        #                       * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = discounted_rewards.double()
        value_targets = value_targets.view(-1, 1)
        values = self.value_network.forward(states).double()
        advantages = value_targets - values
        logits = self.policy_network.forward(states)

        # compute losses
        value_loss = self._value_loss(values, value_targets)
        policy_loss = self._policy_loss(logits, actions, advantages)

        return value_loss, policy_loss

    def _value_loss(self, values, value_targets):
        return self.value_coeff * F.mse_loss(values, value_targets.detach())

    def _policy_loss(self, logits, actions, advantages):
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        batched_ce_loss = nn.CrossEntropyLoss(reduction='none')
        actions = actions.type(torch.long).view(-1)
        policy_loss = batched_ce_loss(logits, actions) * advantages.detach()
        policy_loss = policy_loss.mean()

        # Compute entropy bonus
        probs = F.softmax(logits, dim=1)  # (B, C)
        cat_entropy = -torch.sum(probs * torch.log(probs), dim=1)
        entropy = cat_entropy.mean()

        # Compute policy loss with entropy bonus
        policy_loss = policy_loss - self.entropy_coeff * entropy

        return policy_loss

    def update(self, trajectory):
        value_loss, policy_loss = self.compute_loss(trajectory)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
