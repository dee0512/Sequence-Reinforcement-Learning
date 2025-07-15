import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def soft_update(target, source, tau):
    """
    Perform a soft update of the target network parameters.

    Args:
        target: Target network.
        source: Source network.
        tau (float): Update rate.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    Perform a hard update of the target network parameters.

    Args:
        target: Target network.
        source: Source network.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    """
    Initialize network weights.

    Args:
        m: Network layer.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(obs_shape, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
        self.apply(weights_init_)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        out = self.linear3(x)
        return out

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, neurons=[256, 256]):
        super(Model, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], state_dim)
        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class GaussianPolicyGRU(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyGRU, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(num_actions, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state, previous_action):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        x = self.gru(previous_action, x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, previous_action, planning_horizon=1, evaluate=False):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        output_actions, output_log_probs, output_means = [], [], []
        for _ in range(planning_horizon):
            x = self.gru(previous_action, x)
            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            output_actions.append(action)
            previous_action = mean if evaluate else action
            output_log_probs.append(log_prob)
            output_means.append(mean)
        if planning_horizon > 1:
            return torch.stack(output_actions, dim=1), torch.stack(output_log_probs, dim=1), torch.stack(output_means, dim=1)
        else:
            return output_actions[0], output_log_probs[0], output_means[0]

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyGRU, self).to(device)


class GaussianPolicyRNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyRNN, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.rnn = nn.RNNCell(num_actions, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state, previous_action):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        x = self.rnn(previous_action, x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, previous_action, planning_horizon=1, evaluate=False):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        output_actions, output_log_probs, output_means = [], [], []
        for _ in range(planning_horizon):
            x = self.rnn(previous_action, x)
            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            output_actions.append(action)
            previous_action = mean if evaluate else action
            output_log_probs.append(log_prob)
            output_means.append(mean)
        if planning_horizon > 1:
            return torch.stack(output_actions, dim=1), torch.stack(output_log_probs, dim=1), torch.stack(output_means, dim=1)
        else:
            return output_actions[0], output_log_probs[0], output_means[0]

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyRNN, self).to(device)

class GaussianPolicyLatentGRU(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, latent_dim, action_space=None):
        super(GaussianPolicyLatentGRU, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.gru = nn.GRUCell(num_actions, latent_dim)
        self.mean_linear = nn.Linear(latent_dim, num_actions)
        self.log_std_linear = nn.Linear(latent_dim, num_actions)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state, previous_action):
        x = F.relu(self.linear1(state))
        latent = self.linear2(x)
        x = self.gru(previous_action, latent)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, latent

    def sample(self, state, previous_action, planning_horizon=1, evaluate=False):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        output_actions, output_log_probs, output_means = [], [], []
        latent = x
        for _ in range(planning_horizon):
            x = self.gru(previous_action, x)
            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            output_actions.append(action)
            previous_action = mean if evaluate else action
            output_log_probs.append(log_prob)
            output_means.append(mean)
        if planning_horizon > 1:
            return torch.stack(output_actions, dim=1), torch.stack(output_log_probs, dim=1), torch.stack(output_means, dim=1), latent
        else:
            return output_actions[0], output_log_probs[0], output_means[0], latent

    def get_latent(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyLatentGRU, self).to(device)

class SAC:
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning, hidden_size, lr):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save_checkpoint(self, filename):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict()
        }, filename)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        checkpoint = torch.load(ckpt_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()


class SACGRU:
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning, hidden_size, lr, steps, actor_update_frequency=1, actor_type='GRU', bp_through_time=False, time_aware_critic=False):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device

        self.time_aware_critic = time_aware_critic
        if self.time_aware_critic:
            self.critic = QNetwork(num_inputs+1, action_space.shape[0], hidden_size).to(self.device)
            self.critic_target = QNetwork(num_inputs+1, action_space.shape[0], hidden_size).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target, self.critic)
        self.action_dim = action_space.shape[0]
        self.bp_through_time = bp_through_time

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            if actor_type == 'GRU':
                self.policy = GaussianPolicyGRU(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            elif actor_type == 'RNN':
                self.policy = GaussianPolicyRNN(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.steps = steps
        self.actor_update_frequency = actor_update_frequency
        self.model = Model(num_inputs, action_space.shape[0], neurons=[400, 300]).to(device)
        self.model_target = copy.deepcopy(self.model)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model_loss_fn = nn.MSELoss()

    def select_action(self, state, previous_action, steps, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        previous_action = torch.FloatTensor(previous_action).to(self.device).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.policy.sample(state, previous_action, steps, evaluate)
        else:
            _, _, action = self.policy.sample(state, previous_action, steps, evaluate)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):

        if self.time_aware_critic:
            # todo: Add time aware critic update by updating replay buffer sample method
            state_batch, action_batch, previous_action_batch, next_state_batch, reward_batch, mask_batch = memory.sample_trajectory(
                batch_size=batch_size, steps=self.steps)


            qf_loss = 0
            state_batch = state_batch[:, 0, :]
            state_batch_tensor = torch.ones(next_state_batch.shape[0], 1).to(device)
            for i in range(self.steps):
                with torch.no_grad():
                    pi, log_pi, _ = self.policy.sample(next_state_batch[:, i, :], action_batch[:, i, :], 1)
                    qf1_next_target, qf2_next_target = self.critic_target(torch.cat((next_state_batch[:, i, :], state_batch_tensor*0), dim=1), pi)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_pi
                    next_q_value = reward_batch[:, i,:] + mask_batch[:, i, :] * self.gamma * (min_qf_next_target)

                qf1, qf2 = self.critic(torch.cat((state_batch, state_batch_tensor*i), dim=1), action_batch[:, i, :])
                qf1_loss = F.mse_loss(qf1, next_q_value)
                qf2_loss = F.mse_loss(qf2, next_q_value)
                qf_loss += qf1_loss + qf2_loss
                with torch.no_grad():
                    state_batch = self.model_target(state_batch, action_batch[:, i, :])

            qf_loss = qf_loss/self.steps
            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

        else:
            state_batch, action_batch, previous_action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(
                batch_size=batch_size)
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, action_batch, 1)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

        state_batch, action_batch, previous_action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(
            batch_size=batch_size)
        pred = self.model(state_batch, action_batch)
        model_loss = self.model_loss_fn(pred, next_state_batch)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        if updates % self.actor_update_frequency == 0:
            if self.bp_through_time:
                pi, log_pi, _ = self.policy.sample(state_batch, previous_action_batch, self.steps)
                ps = 0
                while ps < self.steps:
                    state_batch = self.model_target(state_batch, pi[:, ps, :])
                    ps += 1

                qf1_pi, qf2_pi = self.critic(state_batch, pi[:, ps-1, :])
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                total_policy_loss = (self.alpha * log_pi[:, ps-1, :] - min_qf_pi).mean()
                self.policy_optim.zero_grad()
                total_policy_loss.backward()
                self.policy_optim.step()
            elif self.time_aware_critic:
                total_policy_loss = 0
                pi, log_pi, _ = self.policy.sample(state_batch, previous_action_batch, self.steps)
                state_batch_tensor = torch.ones(next_state_batch.shape[0], 1).to(device)
                for ps in range(self.steps):
                    qf1_pi, qf2_pi = self.critic(torch.cat((state_batch, state_batch_tensor*ps), dim=1), pi[:, ps, :])
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    policy_loss = (self.alpha * log_pi[:, ps, :] - min_qf_pi).mean()
                    total_policy_loss += policy_loss

                    with torch.no_grad():
                        state_batch = self.model_target(state_batch, pi[:, ps, :])

                self.policy_optim.zero_grad()
                total_policy_loss.backward()
                self.policy_optim.step()

            else:
                total_policy_loss = 0
                pi, log_pi, _ = self.policy.sample(state_batch, previous_action_batch, self.steps)
                for ps in range(self.steps):
                    qf1_pi, qf2_pi = self.critic(state_batch, pi[:, ps, :])
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    policy_loss = (self.alpha * log_pi[:, ps, :] - min_qf_pi).mean()
                    total_policy_loss += policy_loss

                    with torch.no_grad():
                        state_batch = self.model_target(state_batch, pi[:, ps, :])

                self.policy_optim.zero_grad()
                total_policy_loss.backward()
                self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi.mean(dim=1) + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha)

            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.model_target, self.model, self.tau)

            return qf_loss.item(), qf_loss.item(), total_policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), model_loss.item()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.model_target, self.model, self.tau)

        return qf1_loss.item(), qf2_loss.item(), model_loss.item()

    def save_checkpoint(self, filename):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'model_target_state_dict': self.model_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict()
        }, filename)

    def save_model_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_target_state_dict': self.model_target.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict()
        }, filename)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        checkpoint = torch.load(ckpt_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_target.load_state_dict(checkpoint['model_target_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])

        # if evaluate:
        #     self.policy.eval()
        #     self.critic.eval()
        #     self.critic_target.eval()
        # else:
        #     self.policy.train()
        #     self.critic.train()
        #     self.critic_target.train()






class SACGRULatent(SACGRU):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning, hidden_size, lr, steps, actor_update_frequency, model_horizon):
        super(SACGRULatent, self).__init__(num_inputs, action_space, gamma, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning, hidden_size, lr, steps, actor_update_frequency)
        self.model_horizon = model_horizon
        latent_dim = 50
        self.enc = Encoder(num_inputs, hidden_size, latent_dim=latent_dim).to(device)
        self.enc_target = copy.deepcopy(self.enc)
        self.model = Model(latent_dim, action_space.shape[0], neurons=[hidden_size, hidden_size]).to(device)
        self.model_target = copy.deepcopy(self.model)
        self.model_optimizer = torch.optim.Adam(list(self.enc.parameters()) + list(self.model.parameters()), lr=lr)
        self.model_loss_fn = nn.MSELoss()

    def select_action(self, state, previous_action, steps, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        previous_action = torch.FloatTensor(previous_action).to(self.device).unsqueeze(0)
        enc = self.enc(state)
        if not evaluate:
            action, _, _ = self.policy.sample(enc, previous_action, steps)
        else:
            _, _, action = self.policy.sample(enc, previous_action, steps)
        return action.detach().cpu().numpy()[0]

    def update_model(self, memory, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch = memory.sample(batch_size=batch_size)
        loss = 0
        state_enc = self.enc(state_batch[:, 0, :])
        for t in range(self.model_horizon):
            next_pred = self.model(state_enc, action_batch[:, t, :])
            with torch.no_grad():
                target_enc = self.enc_target(next_state_batch[:, t, :])
            rho = self.rho ** t
            loss += rho * torch.mean(self.model_loss_fn(next_pred, target_enc), dim=-1)
            state_enc = next_pred
        loss = loss.mean()
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return loss.item()

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, previous_action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)
        with torch.no_grad():
            state_enc = self.enc(state_batch)
            next_state_enc = self.enc(next_state_batch)

        self.critic_optim.zero_grad()
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_enc, action_batch, 1)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_enc, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_enc, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        qf_loss.backward()
        self.critic_optim.step()

        if updates % self.actor_update_frequency == 0:
            total_policy_loss = 0
            pi, log_pi, _ = self.policy.sample(state_enc, previous_action_batch, self.steps)
            for ps in range(self.steps):
                qf1_pi, qf2_pi = self.critic(state_enc, pi[:, ps, :])
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss = ((self.alpha * log_pi[:, ps, :]) - min_qf_pi).mean()
                total_policy_loss += policy_loss
                with torch.no_grad():
                    state_enc = self.model_target(state_enc, pi[:, ps, :])

            self.policy_optim.zero_grad()
            total_policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi.mean(dim=1) + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha)

            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.enc_target, self.enc, self.tau)
                soft_update(self.model_target, self.model, self.tau)

            return qf1_loss.item(), qf2_loss.item(), total_policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.enc_target, self.enc, self.tau)
            soft_update(self.model_target, self.model, self.tau)

        return qf1_loss.item(), qf2_loss.item()

    def save_checkpoint(self, filename):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'enc_state_dict': self.enc.state_dict(),
            'enc_target_state_dict': self.enc_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict()
        }, filename)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        checkpoint = torch.load(ckpt_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.enc.load_state_dict(checkpoint['enc_state_dict'])
        self.enc_target.load_state_dict(checkpoint['enc_target_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()

def cosine(pred, target, reduce=False):
    x = F.normalize(pred, dim=-1, p=2)
    y = F.normalize(target, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1, keepdim=(not reduce))
