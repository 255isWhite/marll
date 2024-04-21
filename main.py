import numpy as np
import torch
import torch.nn as nn
from smac.env import StarCraft2Env
import argparse
import sys
from agents import Agents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an agent in the SMAC environment")
    
    parser.add_argument("--alg",type=str,default="QMIX",help="Algorithm to use for the agent")
    parser.add_argument("--lr",type=float,default=5e-4,help="Learning rate for the agent")
    parser.add_argument("--use_lr_decay",type=bool,default=True,help="Use learning rate decay for the agent")
    parser.add_argument("--seed",type=int,default=0,help="Seed for the agent")
    parser.add_argument("--eval_freq",type=int,default=1e3,help="Frequency of evaluation")
    parser.add_argument("--eval_episodes",type=int,default=32,help="Number of episodes to evaluate the agent")
    parser.add_argument("--max_steps",type=int,default=1e6,help="Maximum number of steps for the agent training")
    parser.add_argument("--gamma",type=float,default=0.99,help="Discount factor for the agent")
    parser.add_argument("--rnn_hidden_dim",type=int,default=64,help="Hidden dimension for the RNN")
    parser.add_argument("--mlp_hidden_dim",type=int,default=64,help="Hidden dimension for the MLP")
    parser.add_argument("--hyper_hidden_dim",type=int,default=64,help="Hidden dimension for the hyper network")
    parser.add_argument("--qmix_hidden_dim",type=int,default=32,help="Hidden dimension for the QMIX network")
    parser.add_argument("--epsilon",type=float,default=1.0,help="Epsilon for the epsilon-greedy policy")
    parser.add_argument("--epsilon_decay_final",type=float,default=1e-4,help="Final epsilon after decay")
    parser.add_argument("--epsilon_decay_steps",type=int,default=5e6,help="Steps before epsilon decay")
    parser.add_argument("--batch_size",type=int,default=32,help="Batch size for the update")
    parser.add_argument("--buffer_size",type=int,default=int(1e4),help="Size of the replay buffer")
    parser.add_argument("--use_hard_update",type=bool,default=True,help="Use hard updates for the target network")
    parser.add_argument("--target_update_freq",type=int,default=200,help="Frequency of target network hard updates")
    parser.add_argument("--tau",type=float,default=5e-3,help="Soft update rate for the target network")
    parser.add_argument("--use_rnn",type=bool,default=True,help="Use an RNN for the agent")
    parser.add_argument("--use_orthogonal_init",type=bool,default=True,help="Use orthogonal initialization for the weights")
    
    parser.add_argument("--use_last_action",type=bool,default=True,help="Use the last action as part of the state")
    parser.add_argument("--use_agent_id",type=bool,default=True,help="Use agent id as part of the state")
    parser.add_argument("--map",type=str,default="2s3z",help="Map to run the agent on")
    args = parser.parse_args()
    
    army = Agents(args)
    army.run()
    print("Training Complete!")
    
    
