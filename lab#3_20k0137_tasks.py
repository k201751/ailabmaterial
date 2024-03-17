from aima3.agents import Agent
from aima3.search import Problem
from math import sqrt

class EuclideanDistanceReflexBot(Agent):
    def __init__(self, reference_point=(4, 0)):
        self.reference_point = reference_point
        self.x = None
        self.y = None
        self.last_perception = None

    def get_distance(self, point):
        x_diff = self.reference_point[0] - point[0]
        y_diff = self.reference_point[1] - point[1]
        return sqrt(x_diff ** 2 + y_diff ** 2)

    def program(self, perception):
        self.last_perception = perception
        self.x = perception[0]
        self.y = perception[1]
        return self.get_distance(perception)

bot = EuclideanDistanceReflexBot()
print(bot.program((0,0))) 
print(bot.program((4,4))) 
print(bot.program((8,0))) 

from aima3.agents import Agent

class CarReflexBot(Agent):
    def __init__(self):
        self.front = None
        self.left = None
        self.right = None
        self.rear = None
        self.last_perception = None

    def program(self, perception):
        self.last_perception = perception
        self.front = perception[0]
        self.left = perception[1]
        self.right = perception[2]
        self.rear = perception[3]

        if self.front <= 8:
            return "Apply Brakes"
        elif self.left <= 2:
            return "Move Right"
        elif self.right <= 2:
            return "Move Left"
        elif self.rear <= 0.05:
            return "Apply Brakes"
        else:
            return "Continue Moving"

bot = CarReflexBot()
print(bot.program((10, 3, 4, 0.06))) # Continue Moving
print(bot.program((2, 3, 4, 0.04))) # Apply Brakes
print(bot.program((10, 1, 4, 0.06))) # Move Right
print(bot.program((10, 4, 1, 0.06))) # Move Left



from aima3.agents import Agent

class TemperatureReflexBot(Agent):
    def __init__(self, temperatures):
        self.temperatures = temperatures
        self.total_temperature = sum(temperatures)

    def program(self, perception):
        return (self.total_temperature / len(self.temperatures)) * 1.8 + 32

bot = TemperatureReflexBot([20, 22, 21, 20, 19, 18, 21, 22, 23])
print(bot.program(None)) # 68.4



from aima3.agents import Agent

class VacuumReflexBot(Agent):
    def __init__(self, room):
        self.room = room
        self.row = 0
        self.col = 0

    def program(self, perception):
        # Check if the current cell is dirty
        if self.room[self.row][self.col] == "D":
            self.room[self.row][self.col] = "C"
            return "Clean"
        
        # Check if the up cell is dirty
        if self.row > 0 and self.room[self.row-1][self.col] == "D":
            self.row = self.row - 1
            return "Move Up"
        
        # Check if the left cell is dirty
        if self.col > 0 and self.room[self.row][self.col-1] == "D":
            self.col = self.col - 1
            return "Move Left"
        
        # Check if the right cell is dirty
        if self.col < len(self.room[0])-1 and self.room[self.row][self.col+1] == "D":
            self.col = self.col + 1
            return "Move Right"
        
        # Check if the down cell is dirty
        if self.row < len(self.room)-1 and self.room[self.row+1][self.col] == "D":
            self.row = self.row + 1
            return "Move Down"
        
        return "Stop"

room = [["D", "B", "D", "C"],
        ["D", "C", "C", "C"],
        ["C", "B", "D", "D"],
        ["D", "C", "D", "D"]]

bot = VacuumReflexBot(room)
while True:
    action = bot.program(None)
    if action == "Stop":
        break
    print(action)

for row in room:
    print(row)


from aima3.agents import Agent

class TicTacToeBot(Agent):
    def __init__(self, game):
        self.game = game
        self.player = "O"

    def program(self, perception):
        # Check if the bot can win with the next move
        for i in range(len(self.game)):
            for j in range(len(self.game[0])):
                if self.game[i][j] == " ":
                    self.game[i][j] = self.player
                    if self.game_won(self.game):
                        return (i, j)
                    self.game[i][j] = " "
        
        # Check if the player can win with the next move
        for i in range(len(self.game)):
            for j in range(len(self.game[0])):
                if self.game[i][j] == " ":
                    self.game[i][j] = "X"
                    if self.game_won(self.game):
                        self.game[i][j] = self.player
                        return (i, j)
                    self.game[i][j] = " "
        
        # Pick a random move
        for i in range(len(self.game)):
            for j in range(len(self.game[0])):
                if self.game[i][j] == " ":
                    self.game[i][j] = self.player
                    return (i, j)
        
        return None

    def game_won(self, game):
        for i in range(len(game)):
            if game[i][0] == game[i][1] == game[i][2] != " ":
                return True
        for j in range(len(game[0])):
            if game[0][j] == game[1][j] == game[2][j] != " ":
                return True
        if game[0][0] == game[1][1] == game[2][2] != " ":
            return True
        if game[0][2] == game[1][1] == game[2][0] != " ":
            return True
        return False

game = [[" ", " ", " "],
        [" ", " ", " "],
        [" ", " ", " "]]

bot = TicTacToeBot(game)
total_moves = 0
while total_moves < 9:
    move = bot.program(None)
    if move is None:
        break
    i, j = move
    game[i][j] = "O"
    print("Bot move:", move)
    total_moves += 1
    
    if bot.game_won(game):
        print("Bot wins!")
        break
    if (total_moves >= 9):
        break
    
    move = None
    while move is None:
        move = input("Your move (i j): ")
        try:
            i, j = map(int, move.split())
            if game[i][j] != " ":
                print("Invalid move, try again.")
                move = None
        except:
            print("Invalid move, try again.")
            move = None
    game[i][j] = "X"
    total_moves += 1
    print("Player move:", move)
    if bot.game_won(game):
        print("Player wins!")
        break
    if (total_moves >= 9):
        break

if total_moves == 9:
    print("It's a draw.")


from aima3.search import *

class RoadTripProblem(Problem):
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.initial = start
        
    def actions(self, state):
        return list(self.graph[state].keys())

    def result(self, state, action):
        return action

    def path_cost(self, cost_so_far, state, action, new_state):
        return cost_so_far + self.graph[state][new_state]
    
    def goal_test(self, state):
        return state == self.goal
    
    def h(self, node):
        return 0

def find_shortest_path(graph, start, goal):
    problem = RoadTripProblem(graph, start, goal)
    return astar_search(problem)

if __name__ == "__main__":
    graph = {
    'Karsaz': {'Fast NU': 15, 'Korangi': 10, 'Gulshan': 8, 'Sadar': 10},
    'Fast NU': {'Karsaz': 15, 'Gulshan': 30, 'Korangi': 20},
    'Gulshan': {'Fast NU': 30, 'Korangi': 5, 'Karsaz': 8, 'Sadar': 20},
    'Korangi': {'Gulshan': 5, 'Fast NU': 20, 'Karsaz': 10, 'Sadar': 10},
    'Sadar': {'Karsaz': 10, 'Gulshan': 20, 'Korangi': 10}
}
    result = find_shortest_path(graph, 'Fast NU', 'Sadar')
    print("Shortest Path:", result.path())
    print("Cost:", result.path_cost)
