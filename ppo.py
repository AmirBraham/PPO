import torch
from network import FeedForwardNetwork

class PPO:
    def __init__(self, env):
        self.env = env
        
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # max timesteps per episode
        
        # The input size and output size of the FFN depend on the environment used
        
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # Initialize the actor and critic networks
        self.actor = FeedForwardNetwork(input_size=self.obs_dim, output_size=self.act_dim)
        self.critic = FeedForwardNetwork(input_size=self.obs_dim, output_size=1)
    
    def get_action(self, obs):
        """
        By outputting a mean and using a standard deviation to create a covariance matrix, the network effectively defines a probability distribution over actions. Sampling from this distribution allows the agent to explore actions around the mean, rather than always taking the same action. This stochasticity is crucial for exploration, 
        """
        
        obs = torch.from_numpy(obs).float()
        mean = self.actor(obs)
        # Multivariate Normal Distribution
        cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create the covariance matrix
        cov_mat = torch.diag(cov_var)
        dist = torch.distributions.MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # detach the action and log_prob from the computational graph to avoid backpropagation
        return action.detach().numpy(), log_prob.detach()
    
    def rollout(self):
        # Batch data collection
        # observations: (number of timesteps per batch, dimension of observation)
        # actions: (number of timesteps per batch, dimension of action) 
        # log probabilities: (number of timesteps per batch)
        # rewards: (number of episodes, number of timesteps per episode)
        # reward-to-go's: (number of timesteps per batch)
        # batch lengths: (number of episodes)
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards (Immediate Reward) at each timestep
        batch_rtgs = []            # batch rewards-to-go  cumulative sum of future rewards  
        batch_lens = []            # episodic lengths in batch
        t = 0 # number of timesteps ran for the current batch
        while t < self.timesteps_per_batch:
            # Reset the environment
            obs = self.env.reset()
            done = False
            ep_rews = []
            obs = self.env.reset()
            for ep_t in range(self.max_timesteps_per_episode):
                # ALG STEP 4 : Run the policy for one episode
                # Get the action and log probability
                t += 1
                batch_obs.append(obs)
                action , log_prob = self.get_action(obs)
                batch_log_probs.append(log_prob)
                
                # Step in the environment
                obs, rew, done, _ = self.env.step(action)
                ep_rews.append(rew)
                batch_acts.append(action)
                if done:
                    break
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)
                
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens        
            
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        gamma = 0.98
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def learn(self,total_timesteps):
        timesteps = 0
        while t < timesteps: # ALG STEP 2
            # Step 3 : Collect set of trajectories
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()