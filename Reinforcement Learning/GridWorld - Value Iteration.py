import numpy as np

size = 3
board = np.zeros((size,size))

def update(size, transition_reward, board, i,j):
	if(i==0 and j==0):
		return 0
	if(i==0 and j!=size-1):
		val1 = board[i][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j+1]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==0 and j==size-1):
		val1 = board[i][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==size-1 and j!=size-1 and j!=0):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i][j]+transition_reward
		val3 = board[i][j+1]+transition_reward
		val4 = board[i][j-1]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==size-1 and j==0):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i][j]+transition_reward
		val3 = board[i][j+1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==size-1 and j==size-1):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i!=0 and i!=size-1 and j==size-1):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i!=0 and i!=size-1 and j==0):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j]+transition_reward
		val4 = board[i][j+1]+transition_reward
		return max(val1,val2,val3,val4)
	if(i!=0 and i!=size-1 and j!=0 and j!=size-1):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j]+transition_reward
		val4 = board[i][j+1]+transition_reward
		return max(val1,val2,val3,val4)
	return 1

def evaluate(transition_reward,size,board):
	new_board = np.zeros((size,size))
	for i in range(size):
		for j in range(size):
			new_board[i][j] = update(size,transition_reward,board,i,j)
	return new_board

def run(transition_reward,size, cycles):
	board = np.zeros((size,size))
	for i in range(cycles):
		updated_board = evaluate(transition_reward,size,board)
		if abs(updated_board[size-1][size-1]-board[size-1][size-1])<0.0001:
			print('Converged at cycle num: ', i)
			return updated_board
		board = updated_board

	return board

print(run(-1,4,10))
