"""
Train genetic algorithm with action masking in CartPole-v0.
author: Preston Robinette
date modified: 5.2.23

"""

from utils.ga_masking import Generation
from utils.custom_amask_cpole import CartPole

import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle

import sys
sys.path.append(sys.path[0]+"/results")
sys.path.append(sys.path[0]+"/trained_agents")

# DEFAULT ARGS
# ---------------------
SOLVED_SCORE = 190
SEED = 4
NUM_TRIALS = 1
NUM_EVAL_EPS = 20
MAX_STEPS=500
X_THRESH = 0.75


def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_trials", type=int, default=NUM_TRIALS, help="Number of times to repeat training")
    parser.add_argument("--stop_reward", type=int, default=SOLVED_SCORE, help="Stopping reward criteria for training")
    parser.add_argument("--env_name", type=str, default="cpole", help="Name of the environment")
    parser.add_argument("--strategy", type=str, default='action_masking', help="Training strategy")
    parser.add_argument("--num_eval_eps", type=int, default=NUM_EVAL_EPS, help="Number of episodes to evaluate the trained agent on after training")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Max number of generations to train")
    parser.add_argument("--x_thresh", type=float, default=X_THRESH, help="Action masking threshold used in training")
    parser.add_argument("--seed", type=int, default=SEED, help="Training seed to set randomization for training")
    args = parser.parse_args()

    return args


def final_evaluation(agent, num_rollouts, env):
    """
    Used for final evaluation policy rollout.
    
    Parameters:
    -----------
    agent : spga agent
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
    #
    # set up the environment
    #
    action_masking = env.use_action_masking
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    #
    # Rollout the agent
    #
    for _ in range(num_rollouts):
        safe = True
        steps = 0
        r = 0
        obs = env.reset()
        while True:
            if action_masking:
                pos, pos_dot, th, theta_dot = obs["actual_obs"]
            else:
                pos, pos_dot, th, theta_dot = obs
            if pos > 1.5 or pos < -1.5:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
                
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    #
    # Return information
    #
    return np.mean(eval_rewards), np.mean(eval_time), v_total, v_eps



def main():
    """
    Main function for training
    """
    #
    # Set up training
    #
    args = get_args()
    train_time = []
    agent = None
    #
    # Train the agent for each seed
    #
    print("\nPreparing the environment ...")
    for i in range(args.num_trials):
        env = CartPole(env_config={"use_action_masking": True, "x_threshold": args.x_thresh, "seed":args.seed})
        agent = Generation(env=env, n=30, g=args.max_steps, solved_score=args.stop_reward, e_rate=0.2, mute_rate=0.3, hidd_l=16, strategy=args.strategy, seed=args.seed)
        print('\n-------------------------------------------')
        print('                 Training SPGA             ')
        print('-------------------------------------------')
        start = time.time()
        agent.simulate()
        end = time.time()
        train_time.append(abs(end-start))
        #
        # Save the agent
        #
        agent.best_agent.save("trained_agents/cartpole_ga_masking_seed-{}_checkpoint-{}".format(str(args.seed), abs(args.x_thresh)))
    #
    # Evaluate without action masking
    #
    print('\n--------- INITIAL TESTING ... ')
    env.use_action_masking = False
    agent.best_agent.strategy = None
    norm_eval_reward, norm_eval_time, norm_v_total, norm_v_eps = final_evaluation(agent.best_agent, args.num_eval_eps, env)
    #
    # Save Data
    #
    avg_train_time = round(np.mean(train_time), 4)
    norm_safe_rolls = 100-(norm_v_eps/args.num_eval_eps*100)
    # data = {
    #     "avg_train_time": avg_train_time,
    #     "train_time": train_time,
    #     # "mask_eval_reward": mask_eval_reward,
    #     # "mask_eval_time": mask_eval_time,
    #     # "mask_safe_rolls": mask_safe_rolls,
    #     # "mask_v_total": mask_v_total,
    #     "norm_eval_reward": norm_eval_reward,
    #     "norm_eval_time": norm_eval_time,
    #     "norm_safe_rolls": norm_safe_rolls,
    #     "norm_v_total": norm_v_total,
    #     "ep_reward": agent.best_fitness,
    #     "avg_ep_reward": agent.avg_fitness_history,
    # }
    # with open('results/cpole_ga_masking_seeded_results-{}.pkl'.format(abs(args.x_thresh)), 'wb') as f:
    #     pickle.dump(data, f)
    #
    # Print Values
    #
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('           SPGA TRAINING RESULTS           ')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("Average Time to Train: ", avg_train_time)
    print("\n-----Evaluation -------")
    print("Average Evaluation Reward: ", norm_eval_reward)
    print("Number of Safety Violations: ", norm_v_total)
    print("Percentage of Safe Rollouts: {}%".format(norm_safe_rolls))
    print("Average Rollout Episode Length: ", norm_eval_time)
    
    
        
        
    
if __name__ == "__main__":
    main()