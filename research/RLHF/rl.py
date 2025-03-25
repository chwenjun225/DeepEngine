import random
import fire 



import torch
import torch.nn as nn 
import torch.optim as optim 



from const_vars import (
	N_STATES, 
	N_ACTIONS,
	LR, 
	EPISODES, 
	EPSILON, 
	EPSILON_DECAY,
	EPSILON_MIN, 
)
from utils import one_hot, get_valid_actions



class DQN(nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(state_size, 64), 
			nn.ReLU(), 
			nn.Linear(64, action_size)
		)
	def forward(self, x): 
		return self.fc(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(N_STATES, N_ACTIONS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()


def main():
	for ep in range(EPISODES):
		state = random.randint(0, N_STATES-2) 
		done = False 
		
		while not done:
			state_tensor = torch.FloatTensor(one_hot(state)).unsqueeze(0).to(device)
			### Epsilon-greedy 
			if random.random() < EPSILON:
				action = random.randint(0, N_ACTIONS-1)
			else:
				with torch.no_grad():
					q_values = model(state_tensor)



if __name__ == "__main__":
	fire.Fire(main)
