import gym
import torch
import numpy as np
import utils
from policies.expert import ExpertPolicy
from policies.agent import AgentPolicy
from dataset import Dataset

# Set the experiment parameters
env_name = "HalfCheetah-v2"
eval_batch_size = 10
hidden_dim = 64
lr = 1e-3
make_gif_every = 1
max_traj_len = 1000
n_batch_updates_per_iter = 1000
n_iter = 20
seed = 0
train_batch_size = 64

env = gym.make(env_name)
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
observ_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expert = ExpertPolicy("expert_policies/{}.pkl".format(env_name), device)
agent = AgentPolicy(observ_dim, hidden_dim, action_dim, lr, device)
dataset = Dataset()
log = {"expert": {}, "agent": {}}

# Evaluate the expert's performance
mean, stdev = utils.eval(env, expert, max_traj_len, eval_batch_size)
log["expert"] = {"mean reward": mean, "stdev reward": stdev}
print("expert reward: {} (+/- {})".format(mean, stdev))
utils.make_gif(env, expert, "videos/{}/expert.gif".format(env_name), 150)

# Perform `num_iter` iterations of DAgger
for iter in range(n_iter + 1):
    print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))

    # Evaluate the agent's performance
    mean, stdev = utils.eval(env, agent, max_traj_len, eval_batch_size)
    log["agent"][iter] = {"mean reward": mean, "stdev reward": stdev}
    print("agent reward: {} (+/- {})".format(mean, stdev))
    if iter % make_gif_every == 0:
        utils.make_gif(env, agent, "videos/{}/iter_{}.gif".format(env_name, iter), 150)
    if iter == n_iter:
        break

    # Sample a trajectory with the agent and re-lable actions with the expert
    data = utils.sample_traj(env, agent, max_traj_len)
    data["actions"] = expert.get_action(data["observs"])

    # Aggregate the datasets
    dataset.add(data)

    # Train the agent
    for _ in range(n_batch_updates_per_iter):
        train_batch = dataset.sample(train_batch_size)
        agent.train(train_batch["observs"], train_batch["actions"])

utils.make_log(log, "logs/{}.json".format(env_name))