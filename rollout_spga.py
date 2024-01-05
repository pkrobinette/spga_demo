"""
Demo an SPGA agent.

author: Preston Robinette
date modified: 5.2.23
"""

from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import pandas

from utils.ga_masking import Agent
from utils.custom_amask_cpole import CartPole

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
    # parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-r", "--rollout_seed", type=int, default=40)
    parser.add_argument("--x_thresh", type=float, default=X_THRESH)
    parser.add_argument("-t", "--max_t", type=int, default=MAX_T)
    args = parser.parse_args()

    return args

def rollout(agent, init_pts, env_config={}, render=False):
    """
    Used for final evaluation policy rollout.
    
    Parameters:
    -----------
    agent : ga agent
    num_rollouts : int
        number of times to evaluate an agent
    env_config : dict
        environment configuration file
        
    Returns
    --------
    - mean of all eval rewards
    - mean of all rollout times
    - number of total violations
    - number of episodes with at least one violation
    """
    # set up the environment
    action_masking = env_config.get("use_action_masking", False)
    env = CartPole(env_config)
    traces = []
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    #
    # Rollout the agent
    #
    for pt in init_pts:
        safe = True
        steps = 0
        r = 0
        obs = env.reset(init=True, state=pt)
        trace = []
        while True:
            if action_masking:
                pos, pos_dot, th, theta_dot = obs["actual_obs"]
            else:
                pos, pos_dot, th, theta_dot = obs
            trace.append([pos, th])
            if pos > 1.5 or pos < -1.5:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
                
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            if render:
                env.render()
            r += reward
            steps += 1
            if done:
                eval_rewards.append(r)
                eval_time.append(steps)
                break
        traces.append(trace)
    #
    # Return information
    #
    return np.mean(eval_rewards), np.mean(eval_time), v_total, v_eps, traces

def main():
    args = get_args()
    print("\n ---- ENV ARGS ----")
    print("MAX TIME: ", args.max_t)
    #
    # Demo With Action Asking
    #
    env_config = {"use_action_masking": True, "max_t":args.max_t}
    agent = Agent()
    agent.load("trained_agents/cartpole_ga_masking_seed-{}_checkpoint-{}.json".format(args.seed, str(args.x_thresh)))
    agent.strategy = "action_masking"
    #
    # Make same init position.
    #
    np.random.seed(args.rollout_seed)
    init_pts = [np.random.uniform(low=-0.05, high=0.05, size=(4,)) for _ in range(args.num_rollouts)]
    print("\nINIT STATE: ", init_pts[0])
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, traces = rollout(agent, init_pts, env_config, render=args.render)
    
    #
    # Save the trace
    #
    with open("results/spga_trace.pkl", 'wb') as f:
        pickle.dump(traces[0], f)
    print("\nTrace saved to: results/spga_trace.pkl")
    
    print("\n----- Demo  -----")
    print("Avg. Rollout Reward: ", mask_eval_reward)
    print("Total Violations: ", mask_v_total)
    print("Percentage of Safe Rollouts: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    
    
if __name__ == "__main__":
    main()