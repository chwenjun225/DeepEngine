from collections import deque 



MAZE = {
	0: [4],
	1: [3],
	2: [3],
	3: [1, 2, 4, 5],
	4: [0, 3, 5],
	5: []  
}



N_STATES = 6 
N_ACTIONS = 6 



EPISODES = 1000
GAMMA = 0.8
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.01
BATCH_SIZE = 32
MEMORY = deque(MAXLEN=1000)
