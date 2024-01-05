"""
Train agent with PPO using action masking in the CartPole-v0 environment.

Tuned Hyperparameters From : https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml

author: Preston Robinette
date modified: 5.2.23
"""

# Presto note:         # "model": {"fcnet_hiddens": [16]}
from utils.masking_model import ActionMaskModel
from utils.custom_amask_cpole import CartPole

import ray
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.tune.utils.util import SafeFallbackEncoder
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.filter import MeanStdFilter
from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import os
from os.path import isdir, join, isfile
import shutil

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
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Max number of training steps during the training process")
    parser.add_argument("--x_thresh", type=float, default=X_THRESH, help="X Threshold safety constraint to be used for action masking")
    parser.add_argument("--seed", type=int, default=SEED, help="Training seed to set randomization for training")
    args = parser.parse_args()

    return args


def min_print(result):
    """ Print results for each training step """
    result = result.copy()
    info_keys = [
        'episode_len_mean', 
        'episode_reward_max', 
        'episode_reward_mean',
        'episode_reward_min'
            ]
    out = {}
    for k, v in result.items():
        if k in info_keys:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


def get_ppo_trainer(args= None):
    """ Configure the ppo trainer based on training strategy """
    print("\nPreparing the environment ...")
    config = ppo.DEFAULT_CONFIG.copy()
  
    cpole_config = {
        "gamma": 0.99,
        "lr": 0.0003,
        # "model": {"fcnet_hiddens": [16]}
    }
    config.update(cpole_config)
    config["env"] = CartPole
    config["env_config"] = {"use_action_masking": False}
    
    if args:
        if args.strategy == "action_masking":
            # print('\nUsing action masking to train ...')
            ModelCatalog.register_custom_model("kp_mask", ActionMaskModel)
            config["env_config"] = {"use_action_masking": True}
            config["model"] = {
                "custom_model": "kp_mask",
            }
            
        config["env_config"]["seed"] = args.seed
        config.update({'seed': args.seed})
        print("Using Seed: ", args.seed)
        
        if vars(args).get("x_thresh"):
            print("Using X Threshold: ", args.x_thresh, "\n\n")
            config["env_config"]["x_threshold"] = args.x_thresh
    
    trainer = ppo.PPOTrainer(config=config)
    return trainer, config["env_config"]


def final_evaluation(trainer, n_final_eval, env_config={}):
    """
    Used for final evaluation policy rollout.
    
    Parameters:
    -----------
    trainer : ppo agent
    n_final_eval : int
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
    action_masking = env_config.get("use_action_masking", False)
    env = CartPole(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    
    for _ in range(n_final_eval):
        obs = env.reset()
        r = 0
        steps = 0
        safe = True
        while True:
            if action_masking:
                pos, vel, theta, theta_vel = obs["actual_obs"]
            else:
                pos, vel, theta, theta_vel = obs
            # Check Safety Criteria
            if pos > 1.5 or pos < -1.5:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            action = trainer.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return np.mean(eval_rewards), np.mean(eval_time), v_total, v_eps


def main():
    """
    main function
    """
    #
    # Setup and seeds
    #
    args = get_args()
    ray.shutdown()
    ray.init()
    # np.random.seed(args.seed)
    #
    # Train
    #
    # to be able to access checkpoint after training
    checkpoint = None
    train_time = []
    trainer = None
    ep_reward = None
    avg_ep_reward = None
    for i in range(args.num_trials):
        trainer, env_config = get_ppo_trainer(args)
        results = None
        training_steps = 0
        # Training
        ep_reward = []
        avg_ep_reward = []
        print('\n-------------------------------------------')
        print('                 Training SRL              ')
        print('-------------------------------------------')
        while True:
            results = trainer.train()
            if (training_steps)%10 == 0:
                print(f"Training Step: {training_steps} | Ep. Max Return: {results['episode_reward_max']:.2f} | Ep. Mean Return {results['episode_reward_mean']:.2f}")
            
            ep_reward.append(results["episode_reward_mean"])
            avg_ep_reward.append(np.mean(ep_reward[-30:]))
            training_steps += 1
            if (avg_ep_reward[-1] >= args.stop_reward and training_steps >=30) or training_steps > args.max_steps:
                break
        #
        # save the trained agent
        #
        name = "cartpole_ppo_masking_seed-{}_checkpoint-{}".format(str(args.seed), abs(args.x_thresh))
        train_time.append(results["time_total_s"])
        checkpoint= trainer.save("./trained_agents/"+name)
        for f in os.listdir("./trained_agents/"+name):
            ckpt_dir = join("./trained_agents/"+name, f)
            if not f.startswith('.') and isdir(ckpt_dir):
                for p in os.listdir(ckpt_dir):
                    test = join(ckpt_dir, p)
                    if not p.startswith('.'):
                        if "tune_metadata" in p:
                            os.rename(test, "./trained_agents/"+name+"/"+name+".tune_metadata")
                        else:
                            os.rename(test, "./trained_agents/"+name+"/"+name)
                shutil.rmtree(ckpt_dir)
                        
        checkpoint = "trained_agents/"+name+"/"+name
    #
    # Evaluate with Action Masking
    #
    agent, env_config = get_ppo_trainer()
    agent.restore(checkpoint)
    print("Training Complete. Testing the trained agent with action masking ...\n")
    
#     mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps = final_evaluation(trainer, args.num_eval_eps, {"use_action_masking":True})
    #
    # Evaluate without Action Masking
    #
    print('\n--------- INITIAL TESTING ... ')
    norm_eval_reward, norm_eval_time, norm_v_total, norm_v_eps = final_evaluation(agent, args.num_eval_eps, {"use_action_masking":False})
    #
    # Data
    #
    avg_train_time = round(np.mean(train_time), 4)
    norm_safe_rolls = 100-(norm_v_eps/args.num_eval_eps*100)
    # 
    # Print Values
    #
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('           SRL TRAINING RESULTS           ')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("Average Time to Train: ", avg_train_time)
    print("\n-----Evaluation -------")
    print("Average Evaluation Reward: ", norm_eval_reward)
    print("Number of Safety Violations: ", norm_v_total)
    print("Percentage of Safe Rollouts: {}%".format(norm_safe_rolls))
    print("Average Rollout Episode Length: ", norm_eval_time)
    #
    # Save Training Data and Agent
    #
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
    #     "ep_reward": ep_reward,
    #     "avg_ep_reward": avg_ep_reward,
    # }
    # with open('results/cpole_ppo_masking_seeded_results-{}.pkl'.format(abs(args.x_thresh)), 'wb') as f:
    #     pickle.dump(data, f)
        

if __name__ == "__main__":
    main()
    
    
    
    