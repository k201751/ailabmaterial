
print("TASK # 1\n")
def minimax(board, depth, maximizing_player):
    # Base case: if the game is over or the maximum depth is reached
    winner = check_win(board)
    if winner != '' or depth == 0:
        return get_score(board, depth)
    # for maximier
    if maximizing_player:
        max_eval = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-': # check for empty spot
                    board[i][j] = 'X' # place the mark
                    eval_score = minimax(board, depth - 1, False) # call the function for opponent
                    board[i][j] = '-' # reset spot occupied in case its not optimal
                    max_eval = max(max_eval, eval_score) 
        return max_eval
    # for minimizer
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-': # check for empty spot
                    board[i][j] = 'O' # place the mark
                    eval_score = minimax(board, depth - 1, True) # call the function for opponent
                    board[i][j] = '-'# reset spot occupied in case its not optimal
                    min_eval = min(min_eval, eval_score)
        return min_eval

def get_best_move(board):
    best_move = (-1, -1)
    max_eval = -float('inf')
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                board[i][j] = 'X'
                eval_score = minimax(board, 9, False)
                board[i][j] = '-'
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (i, j)
    return best_move

def check_win(board):
    # Check rows
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != '-':
            return board[i][0]
    # Check columns
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != '-':
            return board[0][i]
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '-':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '-':
        return board[0][2]
    # Check for tie
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                return ''
    return 'Draw'

def get_score(board, depth):
    winner = check_win(board)
    if winner == 'X':
        return 10 - depth
    elif winner == 'O':
        return depth - 10
    else:
        return 0

# board state provided in question
board = [['O', '-', 'X'],
         ['X', '-', '-'],
         ['X', '-', 'O']]
best_move = get_best_move(board)
print("Best move:", best_move)

############################### TASK # 1 ENDS #######################################

print("\nTASK # 2 \n")
# Initial values of Alpha and Beta
MAX, MIN = 9999, -9999


def minimax(depth, nodeIndex, maximizingPlayer,values, alpha, beta):
    # check for leaf node 
	if depth == 3:
		return values[nodeIndex]

	if maximizingPlayer: # for maximizer
		best = MIN
		for i in range(0, 2):
			# left child , right child
			val = minimax(depth + 1, nodeIndex * 2 + i,False, values, alpha, beta)
			best = max(best, val)
			alpha = max(alpha, best)

			# Alpha Beta Pruning
			if beta <= alpha:
				break
		
		return best
	
	else: # for minimizer
		best = MAX
		for i in range(0, 2):
            # left child , right child
			val = minimax(depth + 1, nodeIndex * 2 + i,True, values, alpha, beta)
			best = min(best, val)
			beta = min(beta, best)

			# Alpha Beta Pruning
			if beta <= alpha:
				break
		
		return best
	

values = [2,4,6,8,1,2,10,12]
print("The Obtained value is  :", minimax(0, 0, True, values, MIN, MAX))


####################### TASK # 2 ###############################################

print("\nTASK # 3\n")
import numpy as np

#CSP object used to instantiate nqueens setup 
class csp:
    def __init__(self, col=0, row=0):
        self.board = np.zeros((8,8))
        self.board[row][col] = 1
        self.start= [col,row]
        self.queen_domains = [[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]
        self.placed = 1
        
    # Modifies the board
    def place_queen(self, val, loc):
        self.board[loc[0]][loc[1]] = val
    
    # Checks a possible queen placement  
    def check_placement(self, col, row):
        #Check column
        if 1 in self.board[row]:
            return False
        #Check row
        elif 1 in self.board[:, col]:
            return False
        #Check forward diagonal
        elif 1 in np.diag(self.board, col-row):
            return False
        #Check backwards diagonal
        elif 1 in np.diag(np.fliplr(self.board), (7-(row+col))):
            return False
        else:
            return True
            
    #Driver function to begin forward checking search and print results when finished    
    def forward_checking(self):
        if(self.forward_rec(self.start[0]+1)):
            print(f"Solution has been found using forward checking as follow:\n{self.board}")
        else:
            print(f'A solution could not be found with the given starting queen placement:\n{self.start}')
    
    # Recursive function that will iterate through the queen placements 
    def forward_rec(self, col):
        #Returns true when all queens have been placed
        if self.placed == 8:
            return True
        # compensates for when starting position is not the first column
        if col == 8:
            col = 0
        # check possible placements for the current column
        for val in self.queen_domains[col]:
            if not self.check_placement(col,val):
                self.queen_domains[col][val] = -9
        #Evaluate the next column using possible values from the modified domain of the current column
        for row in self.queen_domains[col]:
            if row != -9:
                self.placed+=1
                self.board[row][col] = 1
                if self.forward_rec(col+1):
                    return True
                self.board[row][col] = 0
                self.placed-=1
        self.queen_domains[col] = [0,1,2,3,4,5,6,7]
        return False
  
nqueen= csp(0,0)
nqueen.forward_checking()
        
        
########################## TASK # 3 ENDS ###################################################

print("\nTASK # 4\n")

def is_solvable(words, result):
    char_values = {
        'B': 7,
        'A': 4,
        'S': 8,
        'E': 3,
        'L': 5,
        'G': 1, 
        'M': 9
    }
    
    word_values = []
    for word in words + [result]:
        value = 0
        for char in word:
            value = value * 10 + char_values[char]
        word_values.append(value)
        
    print(word_values)

    used_digits = set()
    def solve(i, S):
        if i == len(words):
        # check if sum of word values equals the value of the result word
            print("Result obtained : " , S)
            return S == word_values[-1]



        # try all possible values for the current letter
        for digit in range(10):
        # check if the current digit has not been used before
            if digit not in used_digits:
                # add the current digit to used digits set
                used_digits.add(digit)
                # recursive call to try next letter with updated sum
                if solve(i+1, S+ word_values[i-1]*digit):
                    return True
                # remove the current digit from used digits set if it did not lead to a solution
                if digit in used_digits:
                    used_digits.remove(digit)
            
        # if no combination of values for the current letter led to a solution, return False
        return False


    return solve(0, 0)

arr = ["BASE", "BALL"]
S = "GAMES"


if is_solvable(arr, S):
    print("Yes")
else:
    print ("No")

######################################## TASK # 4 ENDS ####################################