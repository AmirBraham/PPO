import torch
import torch.nn.functional as F
from network import FeedForwardNetwork
from torch.distributions import MultivariateNormal
import gymnasium as gym
import matplotlib.pyplot as plt

class PPO:
    def __init__(self, env):
        self.env = env
        
        # Hyperparameters (added missing ones)
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.clip = 0.2
        self.num_updates_per_iteration = 5
        self.gamma = 0.95
        self.lr = 0.005  # Updated learning rate
        
        # Environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Networks
        self.actor = FeedForwardNetwork(input_size=self.obs_dim, output_size=self.act_dim)
        self.critic = FeedForwardNetwork(input_size=self.obs_dim, output_size=1)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Covariance matrix (fixed initialization)
        self.cov_var = torch.full((self.act_dim,), 0.5)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(0)  # Add batch dimension
    
    def get_action(self, obs):
        obs = torch.from_numpy(obs).float()
        
        mean = self.actor(obs.T)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def rollout(self, render=False):
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        ep_rewards = []  # Total reward per episode
        t = 0

        while t < self.timesteps_per_batch:
            obs, _ = self.env.reset()
            ep_rews = []
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                if render:
                    self.env.render()  # Render if requested

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rews.append(float(rew))

                if terminated or truncated or t >= self.timesteps_per_batch:
                    break

            batch_lens.append(len(ep_rews))
            batch_rews.append(ep_rews)
            ep_rewards.append(sum(ep_rews))  # Store total reward for this episode

        # Compute the reward-to-go for every time step across episodes
        return (torch.tensor(batch_obs, dtype=torch.float),
                torch.tensor(batch_acts, dtype=torch.float),
                torch.tensor(batch_log_probs, dtype=torch.float),
                self.compute_rtgs(batch_rews),
                batch_lens,
                ep_rewards)


    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        return torch.tensor(batch_rtgs, dtype=torch.float)
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        return V, dist.log_prob(batch_acts)
    
    def learn(self, total_timesteps):
        timesteps = 0
        all_episode_rewards = []  # Store rewards for every episode during training

        while timesteps < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, ep_rewards = self.rollout()
            timesteps += len(batch_obs)  # Update the total timesteps seen
            
            # Log average reward for this rollout
            avg_reward = sum(ep_rewards) / len(ep_rewards)
            print(f"Average Reward this iteration: {avg_reward:.2f}")
            all_episode_rewards.extend(ep_rewards)
            
            # Advantage calculation
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = (batch_rtgs - V.detach()).float()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            # Update networks
            for _ in range(self.num_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                self.actor_optimizer.zero_grad()
                (-torch.min(surr1, surr2)).mean().backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                F.mse_loss(V, batch_rtgs).backward()
                self.critic_optimizer.step()

        # After training, plot the reward curve over episodes
        plt.plot(all_episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Reward over Episodes")
        plt.show()

    def test(self, episodes=5):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                self.env.render()  # This will open a window (if supported) to show the environment
                action, _ = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                total_reward += rew
                done = terminated or truncated
            print(f"Test Episode {ep+1} Reward: {total_reward}")
if __name__ == "__main__":
    env = gym.make('Pendulum-v1',render_mode='rgb_array')
    model = PPO(env)
    model.learn(1000000)
    
    # After training, test the trained policy with rendering.
    model.test(episodes=3)