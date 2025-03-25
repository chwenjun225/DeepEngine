import numpy as np 



from const_vars import MAZE



def one_hot(state, size=6):
	vec = np.zeros(size)
	vec[state] = 1
	return vec



def get_valid_actions(state):
	return MAZE[state]
