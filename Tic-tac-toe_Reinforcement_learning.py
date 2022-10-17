#TIC TAC TOE game is desinged using reinforcemnt learning, where trained state values has been abtained and then tested against human
# assumed learning_rate = 0.25
import itertools
import random#imported to get random values from set of values
import numpy as np
# during this  training, we iterate many times after each move, the value of the state is updated using the below rule
stateof_game = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]# intializing state of game as  3X3 matrix
players = ['X','O'] # considering two PLAYERS

def place_check(state, player, place):
    if state[int((place-1)/3)][(place-1)%3] is ' ':
        state[int((place-1)/3)][(place-1)%3] = player
    else:
        place = int(input("this place is not empty, please choose another value"))
        place_check(state, player, place)# checking whether block is empty or not
    # checking whether game is completed or not
def state_check(stateof_game):         
    # Checking whether verticals are matched or not
    if (stateof_game[0][0] == stateof_game[1][0] and stateof_game[1][0] == stateof_game[2][0] and stateof_game[0][0] is not ' '):
        return stateof_game[0][0], "completed"
    if (stateof_game[0][1] == stateof_game[1][1] and stateof_game[1][1] == stateof_game[2][1] and stateof_game[0][1] is not ' '):
        return stateof_game[0][1], "completed"
    if (stateof_game[0][2] == stateof_game[1][2] and stateof_game[1][2] == stateof_game[2][2] and stateof_game[0][2] is not ' '):
        return stateof_game[0][2], "completed"
    
    # Checking whether diagonals are matched or not
    if (stateof_game[0][0] == stateof_game[1][1] and stateof_game[1][1] == stateof_game[2][2] and stateof_game[0][0] is not ' '):
        return stateof_game[1][1], "completed"
    if (stateof_game[2][0] == stateof_game[1][1] and stateof_game[1][1] == stateof_game[0][2] and stateof_game[2][0] is not ' '):
        return stateof_game[1][1], "completed"

    # Checking whether horizontals are matched or not
    if (stateof_game[0][0] == stateof_game[0][1] and stateof_game[0][1] == stateof_game[0][2] and stateof_game[0][0] is not ' '):
        return stateof_game[0][0], "completed"
    if (stateof_game[1][0] == stateof_game[1][1] and stateof_game[1][1] == stateof_game[1][2] and stateof_game[1][0] is not ' '):
        return stateof_game[1][0], "completed"
    if (stateof_game[2][0] == stateof_game[2][1] and stateof_game[2][1] == stateof_game[2][2] and stateof_game[2][0] is not ' '):
        return stateof_game[2][0], "completed"
    
    # Check whether game is draw
    a = 0
    for i in range(3):
        for j in range(3):
            if stateof_game[i][j] is ' ':
                a = 1
    if a is 0:
        return None, "Draw"
    
    return None, "Not completed"

def print_board(stateof_game): # printing the outer layer of board
    print('----------------')
    print(' || ' + str(stateof_game[0][0]) + ' || ' + str(stateof_game[0][1]) + ' || ' + str(stateof_game[0][2]) + ' || ')
    print('----------------')
    print(' || ' + str(stateof_game[1][0]) + ' || ' + str(stateof_game[1][1]) + ' || ' + str(stateof_game[1][2]) + ' || ')
    print('----------------')
    print(' || ' + str(stateof_game[2][0]) + ' || ' + str(stateof_game[2][1]) + ' || ' + str(stateof_game[2][2]) + ' || ')
    print('----------------')
    
  
# Initialize state values
player = ['X','O',' ']
states_dict = {}
all_possible_states = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]#finding all possible states
total_states = len(all_possible_states) # 2 players, 9 spaces
total_actions = 9   # 9 spaces
state_values_for_AI_O = np.full((total_states),0.0)
state_values_for_AI_X = np.full((total_states),0.0)
print("n_actions = %i"%( total_actions))

# State values for AI 'O'
for i in range(total_states):
    states_dict[i] = all_possible_states[i]
    winner, _ = state_check(states_dict[i])
    if winner == 'O':   # AI won
        state_values_for_AI_O[i] = 1
    elif winner == 'X':   # AI lost
        state_values_for_AI_O[i] = -1
              
# State values for AI 'X'       
for i in range(total_states):
    winner, _ = state_check(states_dict[i])
    if winner == 'O':   # AI lost
        state_values_for_AI_X[i] = -1 # environment is gining a negative reward for taking that action
    elif winner == 'X':   # AI won
        state_values_for_AI_X[i] = 1 #the states that lead to a win get a positive state value. 
#The agent learns that being in such a POSITIVE VALUE state may lead to a win down the line, so it would be encouraged to be in this state.
def state_value_update_O(curr_state_value, next_state_value, learning_rate):
    new_value = state_values_for_AI_O[curr_state_value] + learning_rate*(state_values_for_AI_O[next_state_value]  -state_values_for_AI_O[curr_state_value])
    state_values_for_AI_O[curr_state_value] = new_value
    
def state_value_update_X(curr_state_value, next_state_value, learning_rate):
    new_value = state_values_for_AI_X[curr_state_value] + learning_rate*(state_values_for_AI_X[next_state_value]  - state_values_for_AI_X[curr_state_value])
    state_values_for_AI_X[curr_state_value] = new_value

def bestaction(state, player, epsilon):     
    moves = []#getting best action
    curr_state_values = []
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if state[i][j] is ' ':
                empty_cells.append(i*3 + (j+1))
    
    for empty_cell in empty_cells:
        moves.append(empty_cell)
        new_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
        for i in range(3):
            for j in range(3):
                new_state[i][j] = state[i][j]

        place_check(new_state, player, empty_cell)
        next_state_value = list(states_dict.keys())[list(states_dict.values()).index(new_state)]
        if player == 'X':
            curr_state_values.append(state_values_for_AI_X[next_state_value])
        else:
            curr_state_values.append(state_values_for_AI_O[next_state_value])
        
    print('Possible moves you can take = ' + str(moves))
    print('Move values = ' + str(curr_state_values))    
    best_move_idx = np.argmax(curr_state_values)
    # The values of these states are collected from a state_value vector, which contains values for all possible states in the game. 
    if np.random.uniform(0,1) <= epsilon:       # Exploration
        best_move = random.choice(empty_cells)
        print('Agent decides to take next move through exploration! Takes action = ' + str(best_move))
        epsilon *= 0.99
    else:   #Exploitation
        best_move = moves[best_move_idx]
        print('Agent decides to exploit! Takes action = ' + str(best_move))
    return best_move

#The agent can then choose the action which leads to the state with the highest value(exploitation), or chooses a random action(exploration), depending on the value of epsilon. 
# PLaying

learning_rate = 0.25
epsilon = 0.3 # here we are following epsilon decreasing strategy, e is assumed as 0.3
total_iterations = 10001
for j in range(total_iterations):
    stateof_game = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    current_state = "Not completed"
    print("\nIteration " + str(j) + "!")
    print_board(stateof_game)
    winner = None
    current_player_idx = random.choice([0,1])
        
    while current_state == "Not completed":
        curr_state_value = list(states_dict.keys())[list(states_dict.values()).index(stateof_game)]
        if current_player_idx == 0:     # AI_X's turn
            print("\nturn of X")         
            block_choice = bestaction(stateof_game, players[current_player_idx], epsilon)
            place_check(stateof_game ,players[current_player_idx], block_choice)
            next_state_value = list(states_dict.keys())[list(states_dict.values()).index(stateof_game)]
            
        else:       # AI_O's turn
            print("\nturn of O!")   
            block_choice = bestaction(stateof_game, players[current_player_idx], epsilon)
            place_check(stateof_game ,players[current_player_idx], block_choice)
            next_state_value = list(states_dict.keys())[list(states_dict.values()).index(stateof_game)]
        
        print_board(stateof_game)
        #print('State value = ' + str(state_values_for_AI[next_state_value]))
        state_value_update_O(curr_state_value, next_state_value, learning_rate)
        state_value_update_X(curr_state_value, next_state_value, learning_rate)
        winner, current_state = state_check(stateof_game)
        if winner is not None:
            print(str(winner) + " won!")
        else:
            current_player_idx = (current_player_idx + 1)%2
        
        if current_state is "Draw":
            print("Draw!")

np.savetxt('TRAINED1_values_X.txt', state_values_for_AI_X, fmt = '%.4f')                   
print('Training Complete!')    

# Save state values for future use

np.savetxt('TRAINED1_values_O.txt', state_values_for_AI_O, fmt = '%.4f')# saving all state values in .txt file and .txt file if used  further while testing


def place_check(state, player, PLACE):# checking whether place in board in empty or not
    if state[int((PLACE-1)/3)][(PLACE-1)%3] is ' ':
        state[int((PLACE-1)/3)][(PLACE-1)%3] = player
    else:
        PLACE = int(input("This place or block  is not empty"))
        place_check(state, player, PLACE)
stateof_game = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
players = ['X','O']# DEFINING THE PLAYERS

def state_check(state_of_game):    
    # Check horizontals
    if (state_of_game[0][0] == state_of_game[0][1] and state_of_game[0][1] == state_of_game[0][2] and state_of_game[0][0] is not ' '):
        return state_of_game[0][0], "completed"
    if (state_of_game[1][0] == state_of_game[1][1] and state_of_game[1][1] == state_of_game[1][2] and state_of_game[1][0] is not ' '):
        return state_of_game[1][0], "completed"
    if (state_of_game[2][0] == state_of_game[2][1] and state_of_game[2][1] == state_of_game[2][2] and state_of_game[2][0] is not ' '):
        return state_of_game[2][0], "completed"
    
    # Checking whether all the places in verticals are same 
    if (state_of_game[0][0] == state_of_game[1][0] and state_of_game[1][0] == state_of_game[2][0] and state_of_game[0][0] is not ' '):
        return state_of_game[0][0], "completed"
    if (state_of_game[0][1] == state_of_game[1][1] and state_of_game[1][1] == state_of_game[2][1] and state_of_game[0][1] is not ' '):
        return state_of_game[0][1], "completed"
    if (state_of_game[0][2] == state_of_game[1][2] and state_of_game[1][2] == state_of_game[2][2] and state_of_game[0][2] is not ' '):
        return state_of_game[0][2], "completed"
    
    # Checking whether all the places in diagonals are same
    if (state_of_game[0][0] == state_of_game[1][1] and state_of_game[1][1] == state_of_game[2][2] and state_of_game[0][0] is not ' '):
        return state_of_game[1][1], "completed"
    if (state_of_game[2][0] == state_of_game[1][1] and state_of_game[1][1] == state_of_game[0][2] and state_of_game[2][0] is not ' '):
        return state_of_game[1][1], "completed"
    
    # Checking if draw
    a = 0
    for k in range(3):
        for j in range(3):
            if state_of_game[k][j] is ' ':
                a = 1
    if a is 0:        
        return None, "Draw"
    
    return None, "Not completed"

  
# Initialize state values
player = ['X','O',' ']
states_dict = {}
possible_game_states = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]
n_states = len(possible_game_states) # 2 players, 9 spaces
n_actions = 9   # 9 spaces
AI_state_values = np.full((n_states),0.0)
print("n_states = %i \nn_actions = %i"%(n_states, n_actions))

for i in range(n_states):
    states_dict[i] = possible_game_states[i]
    winner, _ = state_check(states_dict[i])
    if winner == 'O':   # AI won the game
        AI_state_values[i] = 1
    elif winner == 'X':   # AI lost the game
        AI_state_values[i] = -1

def best_action(state, player):#policy
     
    moves = []
    curr_state_values = []
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if state[i][j] is ' ':
                empty_cells.append(i*3 + (j+1))
    
    for empty_cell in empty_cells:
        moves.append(empty_cell)
        new_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
        for i in range(3):
           for j in range(3):
            new_state[i][j] = state[i][j]
         
        place_check(new_state, player, empty_cell)
        next_state_value = list(states_dict.keys())[list(states_dict.values()).index(new_state)]        
        curr_state_values.append(AI_state_values[next_state_value])
        
    print('Possible moves you can take  = ' + str(moves))# possible moves shows u can take places
    print('Move values  = ' + str(curr_state_values))    
    best_move_idx = np.argmax(curr_state_values)
    best_move = moves[best_move_idx]# epsilon is considered as zero, which means next move is achieved through explotation
    return best_move

# TRAINED STATE VALUES has being loaded by epsilon decreasing strategy
AI_state_values = np.loadtxt('TRAINED1_values_O.txt', dtype=np.float64)

play= 1
while play == 1:
    state_of_game = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    current_state = "Not completed"
    print("\nNew Game!")
    print_board(state_of_game)
    place_value = input("please enter the player who is going to play first-1(you -human,choosen symbol is X) or 0 (AI-choosen symbol is 0)")
    winner = None
    
    if place_value ==1:
        present_player = 0
    else:
        present_player = 1
        
    while current_state == "Not completed":# if game is not completed , go to this loop
        curr_state_value = list(states_dict.keys())[list(states_dict.values()).index(state_of_game)]
        if present_player == 0:
            place = int(input("please enter the place value at which you want to place the 'X'(1 to 9): "))
            place_check(state_of_game ,players[present_player], place)
            
        else:   
            place = best_action(state_of_game, players[present_player])# epsilon is considered as zero, which means next move is  achieved through exploitation
            place_check(state_of_game ,players[present_player], place)
            print("AI has taken the move: " + str(place)) # PLACE INDICATES A VALUE BETWEEN 1 TO 9
        
        print_board(state_of_game)
        winner, current_state = state_check(state_of_game)
        if winner is not None:
            print(" the one who have choosen "+str(winner) + " won the game!")     
        else:
            present_player = (present_player + 1)%2        
        if current_state is "Draw":
            print("the result of the game is Draw!")
 
    play=input("press 1 , if you want to play again")
      
