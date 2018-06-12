
# coding: utf-8

# # Starting by building the game and the user interface

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop


# In[2]:


def showBoard():
    '''Shows the board'''
    for val in board:
        print(val)

def checkFree(x):
    '''Takes an x,y postion and checks if that point on the board is free'''
    b = board.reshape(1,9)
    if b[0][x] ==0:
        return True
    else:
        return False
def checkWin(toggle):
    '''checks wins for each player for diagonals, rows, columns'''
    if toggle:
        high = 6
    else:
        high = 15
    if board.diagonal().sum() == high:
        return True
    if np.flip(board,1).diagonal().sum() == high:
        return True
    for val in range(0,3):
        if board[val,:].sum()==high:
            return True
    for val in range(0,3):
        if board[:,val].sum()==high:
            return True
    return False

def getAvailablePositions():
    pos = []
    for i in range(3):
        for j in range(3):
            if checkFree(i*j):
                pos.append(i*j)
    return pos

def placePiece(x,y,nought_or_cross,board):
    '''Takes a x, y position and a X or O with X=1, and O=2'''
    new_board = np.zeros((3,3))
    new_board[x,y] = nought_or_cross
    for i in range(3):
        for j in range(3):
            new_board [i][j] = board[i][j]
    return new_board

def getReward(result,num_moves):
    '''Reward a game won or lost'''
    if result == 'lost':
        return -10+(-num_moves)
    elif result== 'win':
        return 10+(-num_moves)
    else:
        return 0+(-num_moves)
    
def getMove(action):
    '''Gets the move based on the chosen action number'''
    moves = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    return moves[action]


# In[3]:


model = Sequential()
model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(9,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(9, kernel_initializer='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


# In[5]:


import random
#run function

for epoch in range(1,2):
    
    toggle = False
    board = np.zeros((3,3))
    count = 0
    moves = []
    epsilon = 1
    result = 'play'
    gamma = 0.9
    
    while True:
        toggle = not toggle
        qval = model.predict(board.reshape(1,9), batch_size=1)
        if epsilon > random.random():
            action = np.random.randint(0,9)
        else:
            action = (np.argmax(qval[0][getAvailablePositions()]))
        move = getMove(action)
        x = move[0]
        y = move[1]
        if toggle == True:
            n_or_c = 5
        else:
            n_or_c = 2
        new_board = placePiece(int(x),int(y),n_or_c, board)
        
        reward = getReward(result, count)
        newQ = model.predict(new_board.reshape(1,9), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,9))
        y[:] = qval[:]
        if not checkWin(toggle) or not checkWin(not toggle) or getAvailablePositions(): #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output
        print("Game #: %s" % (epoch,))
        model.fit(board.reshape(1,9), y, batch_size=1, nb_epoch=1, verbose=1)
        board = new_board
        showBoard()
        if checkWin(toggle):
            result = 'lost'
            winner = -1
            break
        elif checkWin(not toggle):
            result = 'win'
            winner = 1
            break
        elif not getAvailablePositions():
            result = 'draw'
            winner = 0
            break
        count = count + 1
        if epsilon > 0.1:
            epsilon -= (1/1000)


# In[ ]:




