import random
import string

POPULATION_SIZE = 70
TARGET = "Artificial Intelligence Lab"

def create_initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        individual = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=len(TARGET)))
        population.append(individual)
    return population

def fitness(individual):
    return sum(c1 != c2 for c1, c2 in zip(individual, TARGET))

def selection(population, k):
    return random.choices(population, k=k, weights=[1/fitness(individual) for individual in population])

def crossover(parent1, parent2):
    index = random.randint(0, len(TARGET)-1)
    child = parent1[:index] + parent2[index:]
    return child

def mutation(individual):
    index = random.randint(0, len(TARGET)-1)
    new_char = random.choice(string.ascii_letters + string.digits + string.punctuation +' ')
    return individual[:index] + new_char + individual[index+1:]

def evolve_population(population):
    generation = 0
    while True:
        offspring = []
        for i in range(POPULATION_SIZE):
            parent1, parent2 = selection(population, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                child = mutation(child)
            offspring.append(child)
        population = offspring
        best_individual = min(population, key=fitness)
        print(f"Generation {generation}: {best_individual}")
        if best_individual == TARGET:
            print("Target string reached!")
            break
        generation += 1
    return best_individual

initial_population = create_initial_population()
result = evolve_population(initial_population)
print(f"Result: {result}")

print("\n\n TASK # 1 ENDS \n\n")
######################## TASK 1 ENDS ###########################

import random

cities = {
    'a': {'b': 3, 'c': 6, 'd': 2},
    'b': {'a': 3, 'c': 4, 'd': 7},
    'c': {'a': 6, 'b': 4, 'd': 4},
    'd': {'a': 2, 'b': 7, 'c': 4},
}

POPULATION_SIZE = 6
MUTATION_RATE = 0.1
NUM_GENERATIONS = 10

def create_initial_population(size):
    population = []
    for i in range(size):
        route = list(cities.keys())
        random.shuffle(route)
        population.append(route)
    return population

def calculate_cost(route):
    cost = 0
    for i in range(len(route)-1):
        cost += cities[route[i]][route[i+1]]
    cost += cities[route[-1]][route[0]]
    return cost

def tournament_selection(population, k=3):
    parents = []
    for i in range(2):
        competitors = random.sample(population, k)
        competitors.sort(key=lambda x: calculate_cost(x))
        parents.append(competitors[0])
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1)-1)
    child1 = parent1[:crossover_point] + [c for c in parent2 if c not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [c for c in parent1 if c not in parent2[:crossover_point]]
    return child1, child2

def mutate(route, rate):
    for i in range(len(route)):
        if random.random() < rate:
            j = random.randint(0, len(route)-1)
            route[i], route[j] = route[j], route[i]
    return route

def evolve_population(population):
    new_population = []
    for i in range(len(population)):
      
        parent1, parent2 = tournament_selection(population)
        child1, child2 = crossover(parent1, parent2)
       
        child1 = mutate(child1, MUTATION_RATE)
        child2 = mutate(child2, MUTATION_RATE)
      
        new_population.append(child1)
        new_population.append(child2)
    return new_population

population = create_initial_population(POPULATION_SIZE)
for i in range(NUM_GENERATIONS):
    print("Generation:", i+1)
    for route in population:
        print(route, calculate_cost(route))
    print("")
    population = evolve_population(population)

best_route = min(population, key=lambda x: calculate_cost(x))
print("Best Route:", best_route)
print("Cost:", calculate_cost(best_route))
print("\n\n TASK # 2 ENDS \n\n")

################################# TASK # 2 ENDS #########################

SuccList ={ 'S':[['A',3],['B',2]], 'A':[['C',4],['D',1]], 'B':[['E',3],['F',1]], 'E':[['H',5]],'F': [['I',2], ['G', 3]]}
Start='S'
Goal='H'
Closed = list()
Heuristics = {
    'A': 12,
    'B': 4,
    'C': 7,
    'D': 3,
    'E': 8,
    'F': 2,
    'H': 4,
    'I': 9,
    'S': 13,
    'G':  0
}
SUCCESS=True
FAILURE=False
State=FAILURE

def MOVEGEN(N):
	New_list=list()
	if N in SuccList.keys():
		New_list=SuccList[N]
	
	return New_list
	
def GOALTEST(N):
	if N == Goal:
		return True
	else:
		return False

def APPEND(L1,L2):
	New_list=list(L1)+list(L2)
	return New_list
	
def SORT(L):
	L.sort(key = lambda x: x[1])
	return L 
	
def BestFirstSearch():
	OPEN=[[Start,5]]
	CLOSED=list()
	global State
	global Closed
	while (len(OPEN) != 0) and (State != SUCCESS):
		print("------------")
		N= OPEN[0]
		print("N=",N)
		del OPEN[0]
		
		if GOALTEST(N[0])==True:
			State = SUCCESS
			CLOSED = APPEND(CLOSED,[N])
			print("CLOSED=",CLOSED)
		else:
			CLOSED = APPEND(CLOSED,[N])
			print("CLOSED=",CLOSED)
			CHILD = MOVEGEN(N[0])
			print("CHILD=",CHILD)
			for val in CLOSED:
				if val in CHILD:
					CHILD.remove(val)
			for val in OPEN:
				if val in CHILD:
					CHILD.remove(val)
			OPEN = APPEND(CHILD,OPEN) 
			print("Unsorted OPEN=",OPEN)
			SORT(OPEN)
			print("Sorted OPEN=",OPEN)
			
	Closed=CLOSED
	return State

result=BestFirstSearch() #call search algorithm
print(Closed,result)


print("\n\n TASK # 3 ENDS \n\n")

###################################### TASK # 3 ENDS ####################################

def A_star(start_node, stop_node):
         
        open_set = set(start_node) 
        closed_set = set()
        g = {} 
        root = {}
 
        g[start_node] = 0
        root[start_node] = start_node
         
        while len(open_set) > 0:
            n = None
            for v in open_set:
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                    n = v                   
            if n == stop_node or Graph_nodes[n] == None:
                pass
            else:
                for (m, weight) in get_neighbors(n):
                    if m not in open_set and m not in closed_set:
                        open_set.add(m)
                        root[m] = n
                        g[m] = g[n] + weight                       
                    else:
                        if g[m] > g[n] + weight:
                            g[m] = g[n] + weight
                            root[m] = n
                            if m in closed_set:
                                closed_set.remove(m)
                                open_set.add(m)
 
            if n == None:
                print('Path does not exist!')
                return None
            if n == stop_node:
                path = []
                while root[n] != n:
                    path.append(n)
                    n = root[n]
                path.append(start_node)
                path.reverse()
                print('Path found: {}'.format(path))
                return path
            open_set.remove(n)
            closed_set.add(n)
        print('Path does not exist!')
        return None     

def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

def heuristic(n):
        H_dist = {
            'S': 5,
            'A': 3,
            'B': 4,
            'C': 2,
            'D': 6,
            'G': 0,    
        }
        return H_dist[n]
   
Graph_nodes = {
   'S': [('A', 1), ('G', 10)],
    'A': [('B', 2),('C', 1)],
    'C': [('D', 3), ('G', 4)],
    'B': [('D', 5)],
    'D': [('G', 1)],
}
A_star('S', 'G')
print("\n\n TASK # 4 ENDS \n\n")

################################ TASK # 4 ENDS ##################################


from queue import PriorityQueue
import heapq

cost = 0

def greedy_best_first_search(maze, start, goal):
    visited = set()
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    distance = {}
    came_from[start] = None
    distance[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, distance[goal]

        visited.add(current)

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                came_from[neighbor] = current
                distance[neighbor] = distance[current] + 1
                
                priority = heuristic_distance(neighbor, goal)
                frontier.put(neighbor, priority)

    return None, float('inf')


def get_neighbors(maze, cell):
    row, col = cell
    neighbors = []
    for delta_row, delta_col in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
        neighbor_row, neighbor_col = row + delta_row, col + delta_col
        if (0 <= neighbor_row < len(maze) and
                0 <= neighbor_col < len(maze[0]) and
                maze[neighbor_row][neighbor_col] != 'B'):
            neighbors.append((neighbor_row, neighbor_col))
    return neighbors


def heuristic_distance(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


# Example usage:
maze = [
    ['B', '10',  '9',  '8', '7',  '6',  '5',  '4',  '3',  '2',  '1', 'G'],
    ['B', '11',  'B',  'B', 'B',  'B',  'B',  'B',  'B',  'B',  'B', '1'],
    ['B', '12',  'B', '10', '9',  '8',  '7',  '6',  '5',  '4',  'B', '2'],
    ['B', '13',  'B', '11', 'B',  'B',  'B',  'B',  '5',  'B',  'B', '3'],
    ['B', '14', '13', '12', 'B', '10',  '9',  '8',  '7',  '6',  'B', '4'],
    ['B',  'B',  'B', '13', 'B', '11',  'B',  'B',  'B',  'B',  'B', '5'],
    ['A', '16', '15', '14', 'B', '12', '11', '10',  '9',  '8',  '7', '6'],
]

start = (6, 0)
goal = (0, 11)

path, distance = greedy_best_first_search(maze, start, goal)

print ("Solved using GREEDY BEST FIRST SEARCH.")
if path is not None:
    print('Path found:', path)
    print('Blocks traveled:', distance)
else:
    print('No path found.')

print("\n\n")



def a_star(maze):
    start = find_start(maze)
    goal = find_goal(maze)
    frontier = [(heuristic(start, goal), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next_pos in neighbors(maze, current):
            new_cost = cost_so_far[current] + int(next_pos[1])
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                heapq.heappush(frontier, (priority, new_cost, next_pos))
                came_from[next_pos] = current

    return came_from, cost_so_far


def find_start(maze):
    for i in range(len(maze)):
        if maze[i][0] == 'A':
            return (i, 0)
    return None


def find_goal(maze):
    for i in range(len(maze)):
        if maze[i][-1] == 'G':
            return (i, len(maze[0]) - 1)
    return None


def neighbors(maze, pos):
    result = []
    row, col = pos
    if col > 0 and maze[row][col-1] != 'B':
        result.append((row, col-1))
    if col < len(maze[0]) - 1 and maze[row][col+1] != 'B':
        result.append((row, col+1))
    if row > 0 and maze[row-1][col] != 'B':
        result.append((row-1, col))
    if row < len(maze) - 1 and maze[row+1][col] != 'B':
        result.append((row+1, col))
    return result


def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

maze = [
    ['B', '10',  '9',  '8', '7',  '6',  '5',  '4',  '3',  '2',  '1', 'G'],
    ['B', '11',  'B',  'B', 'B',  'B',  'B',  'B',  'B',  'B',  'B', '1'],
    ['B', '12',  'B', '10', '9',  '8',  '7',  '6',  '5',  '4',  'B', '2'],
    ['B', '13',  'B', '11', 'B',  'B',  'B',  'B',  '5',  'B',  'B', '3'],
    ['B', '14', '13', '12', 'B', '10',  '9',  '8',  '7',  '6',  'B', '4'],
    ['B',  'B',  'B', '13', 'B', '11',  'B',  'B',  'B',  'B',  'B', '5'],
    ['A', '16', '15', '14', 'B', '12', '11', '10',  '9',  '8',  '7', '6'],
]


came_from, cost_so_far = a_star(maze)

start = find_start(maze)
goal = find_goal(maze)
path = reconstruct_path(came_from, start, goal)
print ("Solved using A* BEST FIRST SEARCH.")
print(path)

print("\n\n TASK # 5 ENDS \n\n")
############################# TASK # 5 ENDS #################################



import heapq
import sys 

class Puzzle:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

    def h(self, node):
        return sum(s != g for s, g in zip(node, self.goal))

    def find_zero(self, board):
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return i, j

    def get_moves(self, board):
        moves = []
        i, j = self.find_zero(board)
        if i > 0:
            moves.append((i - 1, j))
        if i < 2:
            moves.append((i + 1, j))
        if j > 0:
            moves.append((i, j - 1))
        if j < 2:
            moves.append((i, j + 1))
        return moves

    def print_board(self, board):
        for row in board:
            print(row)


    
    def solve(self):
        heap = [(self.h(self.start), self.start)]
        visited = set()
        moves = {}
    
        moves[(tuple(map(tuple, self.start)), 0)] = None
        depth = 0
    
        while heap:
            _, board = heapq.heappop(heap)
            if board == self.goal:
                sys.exit("Goal state found") # exit if goal state is found
                path = []
                while board != self.start:
                    path.append(board)
                    board = moves[(tuple(map(tuple, board)), len(path))]
                path.append(self.start)
                path.reverse()
                for state in path:
                    self.print_board(state)
                    print()
                return path

            if tuple(map(tuple, board)) in visited:
                continue
            visited.add(tuple(map(tuple, board)))
        
            depth += 1
            for move in self.get_moves(board):
                new_board = [row[:] for row in board]
                i, j = self.find_zero(new_board)
                new_i, new_j = move
                new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
                new_tuple = tuple(map(tuple, new_board))
                if new_tuple not in visited:
                    new_depth = depth
                    if (new_tuple, new_depth) not in moves or moves[(new_tuple, new_depth)] is None:
                        moves[(new_tuple, new_depth)] = board
                    else:
                        continue
                    heapq.heappush(heap, (new_depth + self.h(new_board), new_board))
                    self.print_board(new_board)
                    print()
                
        return None

start = [[0, 2, 3], [1, 4, 6], [7, 5, 8]]
goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
puzzle = Puzzle(start, goal)
puzzle.solve()
#################### TASK # 6 ENDS ################