import numpy as np
import torch

class Buffer:
    def __init__(self,args):
        self.args = args
        self.buffer = {
            'obs': np.zeros([args.buffer_size,args.episode_limit+1,args.N,args.obs_dim]),
            'state': np.zeros([args.buffer_size,args.episode_limit+1,args.state_dim]),
            'avail_actions': np.ones([args.buffer_size,args.episode_limit+1,args.N,args.action_dim]), # Attention: change ones to zeros
            'last_actions': np.zeros([args.buffer_size,args.episode_limit+1,args.N,args.action_dim]),
            'actions': np.zeros([args.buffer_size,args.episode_limit,args.N]),
            'DoW': np.ones([args.buffer_size,args.episode_limit,1]), # Attention: here np.ones
            'reward': np.zeros([args.buffer_size,args.episode_limit,1]),
            'active': np.zeros([args.buffer_size,args.episode_limit,1]),
        }
        self.buffer_index = 0
        self.step_index = 0
        self.current_size = 0
        self.episode_lens = np.zeros(args.buffer_size,dtype=np.int32)
        
    def store_one_step(self,obs,state,avail_actions,last_actions,actions,reward,dow):
        self.buffer['obs'][self.buffer_index,self.step_index] = obs
        self.buffer['state'][self.buffer_index,self.step_index] = state
        self.buffer['avail_actions'][self.buffer_index,self.step_index] = avail_actions
        self.buffer['last_actions'][self.buffer_index,self.step_index] = last_actions
        self.buffer['actions'][self.buffer_index,self.step_index] = actions
        self.buffer['reward'][self.buffer_index,self.step_index] = reward
        self.buffer['active'][self.buffer_index,self.step_index] = 1
        self.buffer['DoW'][self.buffer_index,self.step_index] = dow
        
        self.step_index += 1
    
    def store_last_step(self,obs,state,avail_actions):
        self.buffer['obs'][self.buffer_index,self.step_index] = obs
        self.buffer['state'][self.buffer_index,self.step_index] = state
        self.buffer['avail_actions'][self.buffer_index,self.step_index] = avail_actions
        
        self.episode_lens[self.buffer_index] = self.step_index
        self.buffer_index = (self.buffer_index + 1) % self.args.buffer_size
        self.current_size = min(self.current_size + 1,self.args.buffer_size)
        self.step_index = 0
        
    def sample(self):
        index = np.random.choice(self.current_size,self.args.batch_size,replace=False)
        max_episode_len = int(np.max(self.episode_lens[index]))
        
        batch = {}
        for key in self.buffer.keys():
            if key=='obs' or key=='avail_actions' or key=='last_actions' or key=='state':
                batch[key] = torch.tensor(self.buffer[key][index,:max_episode_len+1],dtype=torch.float32).to(self.args.device)
            elif key=='actions':
                batch[key] = torch.tensor(self.buffer[key][index,:max_episode_len],dtype=torch.int64).to(self.args.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][index,:max_episode_len],dtype=torch.float32).to(self.args.device)
        
        return batch, max_episode_len