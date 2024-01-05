"""
Quick way to run the demo files.
author: Preston
"""

import subprocess
import os

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n %%%%%%%%%%%%% SPGA vs. SRL DEMO MENU %%%%%%%%%%%%%% \n")
    print(" --- Training ---")
    print("[1] Train Self-Preserving Genetic Algorithm (SPGA)")
    print("[2] Train Safe Reinforcement Learning (SRL)")
    print("--- Testing ---")
    print("[3] Test Self-Preserving Genetic Algorithm (SPGA)")
    print("[4] Test Safe Reinforcement Learning (SRL)")
    print("--- Compare ---")
    print("[5] Compare Rollouts (SPGA vs. SRL)")
    
    a = int(input("\nSelect an option from the menu above (1-5).\n\t >> "))
    
    assert (a in [1, 2, 3, 4, 5]), "Incorrect input. Ensure you are using a number 1-5."
    
    if a == 1:
        # Run train spga
        subprocess.run(["python", "train_spga.py"])
    
    elif a == 2:
        # Run train srl
        subprocess.run(["python", "train_srl.py"])
    
    elif a == 3:
        # Run test spga
        subprocess.run(["python", "rollout_spga.py", "--render"])
       
    elif a == 4:
        # Run test srl
        subprocess.run(["python", "rollout_srl.py", "--render"])
        
    else:
        # Compare
        subprocess.run(["python", "compare_rollouts.py"])
    
    