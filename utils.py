import numpy as np
from PIL import Image
import os, json

def sample_traj(env, policy, max_traj_len, render=False):
    """Samples a trajectory of at most `max_traj_len` timesteps by executing a policy.

    Args:
    - env: the environment.
    - policy: the policy.
    - max_traj_len: the maximum number of timesteps in the trajectory.
    - render (optional): whether to render the trajectory as a list of frames.

    Returns:
    A dict of:
    - observs: a (T, O) numpy array containing the observation at each of the T timesteps
        in the trajectory.
    - actions: a (T, A) numpy array containing the action taken at each timestep.
    - reward: the total reward obtained throughout the trajectory.
    - frames (included only if render is set to true): a length-T list of (3, H, W) numpy
        arrays containing the RGB pixel values of the rendered image at each timestep.
    """
    traj = {"observs": [], "actions": [], "reward": 0}
    if render:
        traj["frames"] = []
    done = False
    observ = env.reset()
    for _ in range(max_traj_len):
        if render:
            traj["frames"].append(env.render(mode="rgb_array"))
        traj["observs"].append(observ)
        action = policy.get_action(observ)
        observ, reward, done, _ = env.step(action)
        traj["actions"].append(action)
        traj["reward"] += reward
        if done:
            break
    traj["observs"] = np.vstack(traj["observs"])
    traj["actions"] = np.vstack(traj["actions"])
    return traj

def sample_trajs(env, policy, max_traj_len, n_trajs):
    """Samples `n_trajs` trajectories by repeatedly calling sample_traj().
    
    Returns:
    A dict of:
    - observs: a (T', O) numpy array containing the observation at each of the T' timesteps
        across all `n_trajs` trajectories.
    - actions: a (T', A) numpy array containing the action taken at each timestep.
    - rewards: a (`n_trajs`, ) numpy array containing the total reward obtained throughout
        each trajectory.
    """
    data = {"observs":[], "actions":[], "rewards":[]}
    for _ in range(n_trajs):
        traj = sample_traj(env, policy, max_traj_len)
        data["observs"].append(traj["observs"])
        data["actions"].append(traj["actions"])
        data["rewards"].append(traj["reward"])
    data["observs"] = np.concatenate(data["observs"])
    data["actions"] = np.concatenate(data["actions"])
    data["rewards"] = np.array(data["rewards"])
    return data

def eval(env, policy, max_traj_len, eval_batch_size):
    """Evaluates the performance of a policy over `eval_batch_size` trajectories.
    
    Returns:
    A tuple whose two entries, respectively, are the average and standard deviation of
    the total reward per trajectory.
    """
    rewards = sample_trajs(env, policy, max_traj_len, eval_batch_size)["rewards"]
    return np.mean(rewards), np.std(rewards)

def make_gif(env, policy, filename, max_traj_len):
    """Executes a policy and saves the frames as a GIF."""
    frames = sample_traj(env, policy, max_traj_len, render=True)["frames"]
    imgs = [Image.fromarray(frame) for frame in frames]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], loop=0)

def make_log(log, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(log, f)