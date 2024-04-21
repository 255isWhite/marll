import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from smac.env import StarCraft2Env
import argparse
import time
import sys

from buffer import Buffer
from network import Q_Net, Q_Net_RNN, QMIX_Net, VDN_Net

class Agents():
    def __init__(self,args) -> None:
        # SMAC environment
        self.env = StarCraft2Env(map_name=args.map)
        self.env_info = self.env.get_env_info()
        
        args.obs_dim = self.env_info["obs_shape"]
        args.state_dim = self.env_info["state_shape"]
        args.action_dim = self.env_info["n_actions"]
        args.N = self.env_info["n_agents"]
        args.episode_limit = self.env_info["episode_limit"]
        print("----ENVIRONMENT INFO----")
        print("Observation Dimension: ",args.obs_dim)
        print("State Dimension: ",args.state_dim)
        print("Number of Actions: ",args.action_dim)
        print("Number of Agents: ",args.N)
        print("Episode Limit: ",args.episode_limit)
        args.input_dim = args.obs_dim
        if args.use_last_action:
            args.input_dim += args.action_dim
            print("Using Last Action")
        if args.use_agent_id:
            args.input_dim += args.N
            print("Using Agent ID")
            
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            args.device = self.device
            print("Using GPU: ",torch.cuda.get_device_name(0))
            if torch.cuda.device_count() > 1:
                print("Using {} GPUs".format(torch.cuda.device_count()))
                gpu_ids = list(range(torch.cuda.device_count()))
        else :
            self.device = torch.device("cpu")
            args.device = self.device
            print("Using CPU")
        
        # argparse parameters transition
        args.epsilon_decay = (args.epsilon - args.epsilon_decay_final) / args.epsilon_decay_steps
        self.args = args
        
        # Random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        # Networks
        if args.use_rnn:
            if False and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.q_net = nn.DataParallel(Q_Net_RNN(args),device_ids=gpu_ids)
                self.target_q_net = nn.DataParallel(Q_Net_RNN(args),device_ids=gpu_ids)
                self.q_net.to(self.device)
                self.target_q_net.to(self.device)            
            else:
                self.q_net = Q_Net_RNN(args).to(self.device)
                self.target_q_net = Q_Net_RNN(args).to(self.device)
        else:
            if False and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.q_net = nn.DataParallel(Q_Net(args),device_ids=gpu_ids)
                self.target_q_net = nn.DataParallel(Q_Net(args),device_ids=gpu_ids)
                self.q_net.to(self.device)
                self.target_q_net.to(self.device)
            else:
                self.q_net = Q_Net(args).to(self.device)
                self.target_q_net = Q_Net(args).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        if args.alg == "QMIX":
            if False and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.mixer = nn.DataParallel(QMIX_Net(args),device_ids=gpu_ids)
                self.target_mixer = nn.DataParallel(QMIX_Net(args),device_ids=gpu_ids)
                self.mixer.to(self.device)
                self.target_mixer.to(self.device)
            else:
                self.mixer = QMIX_Net(args).to(self.device)
                self.target_mixer = QMIX_Net(args).to(self.device)
        elif args.alg == "VDN":
            if False and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.mixer = nn.DataParallel(VDN_Net(args),device_ids=gpu_ids)
                self.target_mixer = nn.DataParallel(VDN_Net(args),device_ids=gpu_ids)
                self.mixer.to(self.device)
                self.target_mixer.to(self.device)
            else:
                self.mixer = VDN_Net(args).to(self.device)
                self.target_mixer = VDN_Net(args).to(self.device)
        else:
            raise Exception("Algorithm not implemented")
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        self.net_parameters = list(self.q_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.net_parameters,lr=self.args.lr)
        
        self.str_time = time.strftime("%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir="./runs/{}/{}_seed{}_{}".format(args.alg,args.map,args.seed,self.str_time))
        self.eval_times = 0
        self.buffer = Buffer(self.args)
        self.train_times = 0
        self.steps = 0
        
    def run(self):
        
        with tqdm( total = self.args.max_steps, desc='Training' ) as pbar:
            
            while self.steps < self.args.max_steps:
                self.q_net.to('cpu')
                win, reward, episode_steps = self.run_one_episode()
                self.steps +=  episode_steps
            
                # Attention: if exists episode_length > 2*eval_freq, eval total times will be less than max_steps/eval_freq
                if self.steps // self.args.eval_freq >= self.eval_times:
                    self.eval_times += 1
                    self.evaluate()
                
                if self.buffer.current_size > self.args.batch_size:
                    self.train()
                    
                pbar.update(episode_steps)
                
        self.evaluate()
        self.writer.flush()
        self.writer.close()
        self.env.close()
                
    def run_one_episode(self,evaluate = False):
        self.env.reset()
        episode_reward = 0
        step = 0
        win = False
        if self.args.use_rnn:
            self.q_net.rnn_hidden = None
            self.target_q_net.rnn_hidden = None
        last_actions = np.zeros((self.args.N,self.args.action_dim)) #(N,action_dim)
        
        for episode_step in range(self.args.episode_limit):
            obs = self.env.get_obs() # (N,obs_dim)
            state = self.env.get_state() # (state_dim,)
            avail_actions = self.env.get_avail_actions() # (N,action_dim)
            
            epsilon = 0 if evaluate else self.args.epsilon
            actions = self.get_actions(obs,avail_actions,epsilon,last_actions) # (N,)
            last_actions = np.eye(self.args.action_dim)[actions] # (N,action_dim)
            reward, done, info = self.env.step(actions)
            episode_reward += reward
            win = True if done and 'battle_won' in info and info['battle_won'] else False
            
            if not evaluate:
                # Since every episode will reach the steps limit, DoW (Death or Win) is induced to fill the buffer
                if done and episode_step + 1 != self.args.episode_limit:
                    DoW = True
                else:
                    DoW = False
                self.buffer.store_one_step(obs,state,avail_actions,last_actions,actions,reward,DoW)   
            
            if done:
                break        
        
        if not evaluate:
            obs = self.env.get_obs()
            state = self.env.get_state()
            avail_actions = self.env.get_avail_actions()
            self.buffer.store_last_step(obs,state,avail_actions)
        
        return  win, episode_reward, episode_step+1
    
    def evaluate(self):
        eval_reward = 0
        win_times = 0
        
        for _ in range(self.args.eval_episodes):
            obs = self.env.reset()
            win, episode_reward, _ = self.run_one_episode(evaluate=True) 
            if win:
                win_times += 1
            eval_reward += episode_reward
        
        win_rate = win_times / self.args.eval_episodes
        avg_reward = eval_reward / self.args.eval_episodes
        self.writer.add_scalar("win_rate_{}".format(self.args.map),win_rate,self.steps)
        self.writer.add_scalar("avg_reward_{}".format(self.args.map),avg_reward,self.steps)
        # print("Steps: {}, Win Rate: {}, Average Reward: {}".format(self.steps,win_rate,avg_reward))
            
    def train(self):
        self.q_net.to(self.device)
        self.train_times += 1
        if self.args.use_lr_decay:
            self.args.lr = self.args.lr * (1.0 - self.steps/self.args.max_steps)
        batch, max_episode_len = self.buffer.sample()
        inputs = self.get_inputs(batch, max_episode_len)
        
        q_evals = []
        q_tars = []
        if self.args.use_rnn:
            self.q_net.rnn_hidden = None
            self.target_q_net.rnn_hidden = None
            for i in range(max_episode_len):
                q_eval = self.q_net(inputs[:,i].reshape(-1,self.args.input_dim)) # (batch_size*N,input_dim) -> (batch_size*N,action_dim)
                q_tar = self.target_q_net(inputs[:,i+1].reshape(-1,self.args.input_dim)) # (batch_size*N,input_dim) -> (batch_size*N,action_dim)
                q_evals.append(q_eval.reshape(-1,self.args.N,self.args.action_dim)) # (batch_size,N,action_dim)
                q_tars.append(q_tar.reshape(-1,self.args.N,self.args.action_dim)) # (batch_size,N,action_dim)
            q_evals = torch.stack(q_evals,dim=1) # (batch_size,max_episode_len,N,action_dim)
            q_tars = torch.stack(q_tars,dim=1) # (batch_size,max_episode_len,N,action_dim)
        else:
            q_evals = self.q_net(inputs[:,:-1]) # (batch_size,max_episode_len,N,action_dim)
            q_tars = self.target_q_net(inputs[:,1:]) # (batch_size,max_episode_len,N,action_dim)
            
        # consider available actions
        with torch.no_grad():
            # Double DQN
            # do reshape just for rnn compatibility
            q_eval_last = self.q_net(inputs[:,-1].reshape(-1,self.args.input_dim)) # (batch_size*N,input_dim) -> (batch_size*N,action_dim)
            q_eval_last = q_eval_last.reshape(-1,1,self.args.N,self.args.action_dim) # (batch_size,N,action_dim)
            q_evals_next = torch.cat([q_evals[:,1:],q_eval_last],dim=1) # (batch_size,max_episode_len,N,action_dim)
            q_evals_next[batch['avail_actions'][:,1:] == 0] = -float("inf")
            arg_action_index = torch.argmax(q_evals_next,dim=-1,keepdim=True) # (batch_size,max_episode_len,N,1)
            q_tars = torch.gather(q_tars,dim=-1,index=arg_action_index).squeeze(-1) # (batch_size,max_episode_len,N)
            
            # Normal DQN
            # q_tars[batch['avail_actions'][:,1:] == 0] = 0
            # q_tars = torch.max(q_tars,dim=-1)[0] # (batch_size,max_episode_len,N)
            
        # Q(s,a)
        q_evals = q_evals.gather(dim=-1,index=batch['actions'].unsqueeze(-1)).squeeze(-1) # (batch_size,max_episode_len,N)
        
        if self.args.alg == "QMIX":
            q_tot_eval = self.mixer(q_evals,batch['state'][:,1:])
            q_tot_tar = self.target_mixer(q_tars,batch['state'][:,1:])
        else:
            q_tot_eval = self.mixer(q_evals) # (batch_size,max_episode_len,N,1)
            q_tot_tar = self.target_mixer(q_tars) # (batch_size,max_episode_len,1)
        
        targets = batch['reward'] + self.args.gamma * q_tot_tar * (1-batch['DoW']) # (batch_size,max_episode_len,1)
        td_error = q_tot_eval - targets # (batch_size,max_episode_len,1)
        # for timestep without data, set td_error to 0
        td_error = td_error * batch['active'] # (batch_size,max_episode_len,1)
        
        # MSE Loss
        loss = (td_error ** 2).sum() / batch['active'].sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # target network update
        if self.args.use_hard_update:
            if self.train_times % self.args.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        else:
            for param, target_param in zip(self.q_net.parameters(),self.target_q_net.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.mixer.parameters(),self.target_mixer.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        
        
    
    def get_actions(self,obs,avail_actions,epsilon,last_actions):
        with torch.no_grad():
            if np.random.uniform() < epsilon:
                actions = [np.random.choice(np.where(avail_action)[0]) for avail_action in avail_actions]
            else:
                inputs = []
                obs = torch.tensor(np.array(obs),dtype=torch.float32) # (N,obs_dim)
                inputs.append(obs)
                if self.args.use_last_action:
                    last_actions = torch.tensor(last_actions,dtype=torch.float32) # (N,action_dim)
                    inputs.append(last_actions)
                if self.args.use_agent_id:
                    agent_id = torch.eye(self.args.N) # (N,N)
                    inputs.append(agent_id)
                    
                inputs = torch.cat([x for x in inputs],dim=-1) # (N,obs_dim+action_dim+N)
                Q_value = self.q_net(inputs) # (N,action_dim)
                avail_actions = torch.tensor(np.array(avail_actions),dtype=torch.float32) # (N,action_dim)
                Q_value[avail_actions == 0] = -float("inf")
                actions = torch.argmax(Q_value,dim=-1).numpy() # (N,)
            return actions
    
    def get_inputs(self,batch,max_episode_len):
        inputs = []
        inputs.append(batch['obs']) # (batch_size,max_episode_len+1,N,obs_dim)
        if self.args.use_last_action:
            inputs.append(batch['last_actions']) # (batch_size,max_episode_len+1,N,action_dim)
        if self.args.use_agent_id:
            ids = torch.eye(self.args.N).unsqueeze(0).unsqueeze(0).cuda() # (1,1,N,N)
            ids = ids.repeat(self.args.batch_size,max_episode_len+1,1,1) # (batch_size,max_episode_len+1,N,N)
            inputs.append(ids)
        inputs = torch.cat([x for x in inputs],dim=-1) # (batch_size,max_episode_len+1,N,obs_dim+action_dim+N)
        return inputs
        