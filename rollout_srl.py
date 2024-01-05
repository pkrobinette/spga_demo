"""
Demo a trained SRL agent.

author: Preston Robinette
date modified: 5.2.23
"""

from utils.masking_model import ActionMaskModel
from utils.custom_amask_cpole import CartPole

import ray
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.tune.utils.util import SafeFallbackEncoder
import ray.rllib.agents.ppo as ppo
from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
from train_srl import get_ppo_trainer

import sys
sys.path.append(sys.path[0]+"/results")
sys.path.append(sys.path[0]+"/trained_agents")
sys.path.append(sys.path[0]+"/utils")

# DEFAULT ARGS
# ---------------------
NUM_ROLLOUTS = 1
SEED = 4
X_THRESH = 0.75
MAX_T = 1000

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS, help="Number of times to rollout agent in env")
    parser.add_argument("--render", action='store_true', help='Render the rollout in real-time.')
    parser.add_argument("--strategy", default="action_masking")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-r", "--rollout_seed", type=int, default=40)
    parser.add_argument("--x_thresh", type=float, default=X_THRESH)
    parser.add_argument("-t", "--max_t", type=int, default=MAX_T)
    args = parser.parse_args()

    return args

def rollout(trainer, init_pts, env_config={}, render=False):
    """
    Rollout the trainer
    """
    action_masking = env_config.get("use_action_masking", False)
    env = CartPole(env_config)
    print(env.T)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    traces = []
    actions = []
    
    for pt in init_pts:
        obs = env.reset(init=True, state=pt)
        r = 0
        steps = 0
        safe = True
        trace = []
        while True:
            if action_masking:
                pos, vel, theta, theta_vel = obs["actual_obs"]
            else:
                pos, vel, theta, theta_vel = obs
            trace.append([pos, theta])
            # Check Safety Criteria
            if pos > 1.5 or pos < -1.5:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            if render:
                env.render()
            action = trainer.compute_single_action(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                traces.append(trace)
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return round(np.mean(eval_rewards),4), round(np.mean(eval_time), 4), v_total, v_eps, traces

def main():
    args = get_args()
    #
    # Demo
    #
    print("\n ---- ENV ARGS ----")
    print("MAX TIME: ", args.max_t)
    mask_agent, mask_env_config = get_ppo_trainer(args)
    print(mask_env_config)
    mask_env_config["max_t"] = args.max_t
    name = "cartpole_ppo_masking_seed-{}_checkpoint-{}".format(args.seed, args.x_thresh)
    mask_agent.restore("trained_agents/{}/{}".format(name, name))
    #
    # Make same init points as rollout_spga
    #
    np.random.seed(args.rollout_seed)
    init_pts = [np.random.uniform(low=-0.05, high=0.05, size=(4,)) for _ in range(args.num_rollouts)]
    print("\nINIT STATE: ", init_pts[0])
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, traces = rollout(mask_agent, init_pts, env_config=mask_env_config, render=args.render)
    
    with open("results/srl_trace.pkl", 'wb') as f:
        pickle.dump(traces[0], f)
    print("\nTrace saved to: results/spga_trace.pkl")
    
    print("\n----- Demo -----")
    print("Rollout Reward: ", mask_eval_reward)
    print("Total Violations: ", mask_v_total)
    print("Percentage of Safe Rollouts: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    
    
if __name__ == "__main__":
    main()