import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from env import Yacht
import copy
import matplotlib.pyplot as plt

learning_rate = 0.0005
gamma = 1
buffer_limit = 6400
batch_size = 64

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            
        return torch.tensor(s_lst, dtype=torch.float).to('cuda'), torch.tensor(a_lst).to('cuda'), torch.tensor(r_lst).to('cuda'), torch.tensor(s_prime_lst, dtype=torch.float).to('cuda'), torch.tensor(done_mask_lst).to('cuda')
    
    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.load_model = False
        self.fc1 = nn.Linear(17,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,12)
        
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x),0.01)
        x = F.leaky_relu(self.fc2(x),0.01)
        x = F.leaky_relu(self.fc3(x),0.01)
        x = self.fc4(x)
        return x
    
    def score_not_yet(self, s_list):
        new_list = []
        for i in range(len(s_list)):
            if(s_list[i] == 0):
                new_list.append(i)
        return new_list
    
    def sample_action(self, s, epsilon):
        state_list = s[5:]
        if np.random.rand() <= epsilon:
            # 무작위 행동 반환
            return random.choice(self.score_not_yet(state_list)) if 0 in state_list else 0
        
        else:
            # 모델로부터 행동 산출
            action_val = []
            s_norm = copy.deepcopy(s)
            for i in range(5):
                s_norm[i] /= 6
            q_values = self.forward(torch.from_numpy(s_norm).to('cuda'))
            q_values = q_values.cpu().detach().numpy()
            score_not_yet_list = self.score_not_yet(state_list)
            for i in range(len(score_not_yet_list)):
                action_val.append(q_values[score_not_yet_list[i]])
            #print(state)
            #print(q_values)
            #print(score_not_yet_list)
            #print(action_val)
            return score_not_yet_list[np.argmax(action_val)] if score_not_yet_list else 0
        
    def train(self, q, q_target, memory, optimizer):
        for i in range(10):
            s,a,r,s_prime,done_mask = memory.sample(batch_size)
            
            q_out = q(s)
            q_a = q_out.gather(1,a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma*max_q_prime*done_mask
            loss = F.smooth_l1_loss(q_a, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            
class A2Cnet(nn.Module):
    def __init__(self):
        super(A2Cnet, self).__init__()
        self.data = []
        self.load_model = False
        self.fc1 = nn.Linear(17, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_pi = nn.Linear(32, 32)
        
        self.fc4 = nn.Linear(128, 32)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.000005)
        
    def pi(self, x, softmax_dim = 0):
        x = F.leaky_relu(self.fc1(x),0.01)
        x = F.leaky_relu(self.fc2(x),0.01)
        x = F.leaky_relu(self.fc3(x),0.01)
        prob = F.softmax(self.fc_pi(x), dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.leaky_relu(self.fc1(x),0.01)
        x = F.leaky_relu(self.fc4(x),0.01)
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float).to('cuda'),torch.tensor(a_lst).to('cuda'), torch.tensor(r_lst, dtype=torch.float).to('cuda'), torch.tensor(s_prime_lst, dtype=torch.float).to('cuda'), torch.tensor(done_lst, dtype=torch.float).to('cuda')
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma*self.v(s_prime)*done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
    
    def make_reward(self, q, score):
        for i in range(len(self.data)):
            self.data[i][2] = (score - q.forward(torch.from_numpy(self.data[i][0]).to('cuda'))[q.sample_action(self.data[i][0][0:5]*6, 0)]) / 30
            
def main():
    EPISODES = 20001
    scores = []
    avg_score = []
    avg_scores = []
    avg_episodes = []
    episodes = []
    
    env = Yacht()
    q = Qnet().cuda()
    q_target = Qnet().cuda()
    a2c = A2Cnet().cuda()
    
    mode = 2
    
    
    if mode == 1:
        q.load_model = False
        a2c.load_model = False
        eps_st = 1
    elif mode == 2:
        q.load_model = True
        a2c.load_model = False
        eps_st = 0
    elif mode == 3:
        q.load_model = True
        a2c.load_model = True
        eps_st = 0
    if q.load_model:
        q = torch.load('C:\python\save_model\DQN.pt')
    if a2c.load_model:
        a2c = torch.load('C:\python\save_model\A2C.pt')
    q_target.load_state_dict(q.state_dict())
    
    memory = ReplayBuffer()
    
    global_step = 0
    print_interval = 100
    
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    
        
    
    for n_epi in range(EPISODES):
        score = 0.0
        epsilon = max(0.01, eps_st-0.000055*n_epi)
        s = env.reset()
        s = np.array(s, dtype=np.float32)
        done = False
        
        while not done:
            global_step += 1
            if mode == 2:
                for _ in range(2):
                    
                    s_norm = copy.deepcopy(s)
                    for i in range(5):
                        s_norm[i] /= 6.0
                    
                    prob = a2c.pi(torch.from_numpy(s_norm).float().to('cuda'))
                    m = Categorical(prob)
                    a = m.sample().item()
                    print("state:",s)
                    print("prob:",prob)
                    print("action:",a)
                    
                    print("value:",a2c.v(torch.from_numpy(s_norm).float().to('cuda')))
                    s_prime, r, done = env.step_reroll(s,a)    
                    s_prime = np.array(s_prime, dtype=np.float32)
                        
                    #r = avg_next_value - q.forward(torch.from_numpy(s_norm).to('cuda'))[q.sample_action(s, 0)]
                    #_, reward_prime, _ = env.step_try(s_prime, q.sample_action(s_prime, 0))
                    #_, reward, _ = env.step_try(s, q.sample_action(s,0))
                    #r = reward_prime-reward
                    
                    
                    
                
                
                    s_prime_norm = copy.deepcopy(s_prime)
                    for i in range(5):
                        s_prime_norm[i] /= 6.0
                        
                    print("s_prime:",s_prime)
                    print("action of s_prime:",q.sample_action(s_prime,0))
                    '''if q.forward(torch.from_numpy(s_prime_norm).to('cuda'))[q.sample_action(s_prime, 0)] >= q.forward(torch.from_numpy(s_norm).to('cuda'))[q.sample_action(s, 0)]:
                        r = 1.0
                    else : r = -1.0'''
                    print("next q:",q.forward(torch.from_numpy(s_prime_norm).to('cuda')))
                    print("q:",q.forward(torch.from_numpy(s_norm).to('cuda')))
                    print("reward:",r)
                    print("*************************")
                    print()
                    a2c.put_data([s_norm,a,r,s_prime_norm,done])
                    #a2c.train_net()
                    s = s_prime
                env.state = s.tolist()
                   
            
            a = q.sample_action(s, epsilon)
            s_prime, r, done = env.step(s,a)
            s_prime = np.array(s_prime, dtype=np.float32)
            done_mask = 0.0 if done else 1.0
            
            s_norm = copy.deepcopy(s)
            for i in range(5):
                s_norm[i] /= 6.0
            s_prime_norm = copy.deepcopy(s_prime)
            for i in range(5):
                s_prime_norm /= 6.0
            print("state:",s)
            print("q:",q.forward(torch.from_numpy(s_norm).to('cuda'))[q.sample_action(s_norm, epsilon)])
            print("*************************")
            print()
            memory.put((s_norm,a,r/30,s_prime_norm, done_mask))
            a2c.data[-1][2] = r/60
            a2c.data[-2][2] = r/60
            s = s_prime
            score += r
            if done:
                break
            
        if memory.size()>2000 and mode != 3:
            #q.train(q, q_target, memory, optimizer)
            pass
            
        if done:
            #a2c.make_reward(q, score)
            a2c.train_net()
            
            scores.append(score*30)
            avg_score.append(score*30)
            episodes.append(n_epi)
            plt.figure(1)
            plt.plot(episodes, scores, 'b')
            if n_epi == EPISODES-1:
                plt.savefig("./save_graph/DQN.png")
            if n_epi%print_interval == 0 and n_epi != 0 :
                avg_scores.append(sum(avg_score)/print_interval)
                avg_episodes.append(n_epi)
                plt.figure(2)
                #plt.ylim([0,100])
                plt.plot(avg_episodes,avg_scores, 'b')
                avg_score.clear()
                plt.savefig("./save_graph/avg_DQN_{}.png".format(n_epi))
            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("episode:", n_epi, "  score:", score*30, "global_step", global_step)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
                
        if n_epi%print_interval==0 and n_epi != 0:
            torch.save(q, "./save_model/DQN.pt")
            torch.save(a2c, "./save_model/A2C.pt")
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval, memory.size(),epsilon*100))
            score = 0.0
            
    
main()