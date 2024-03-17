from sys import maxsize
from itertools import permutations

V = 4

def travellingSalesmanProblem(graph, s):
	vertex = []
	for i in range(V):
		if i != s:
			vertex.append(i)

	min_path = maxsize
	next_permutation = permutations(vertex)
	path = None
	for i in next_permutation:
		current_pathweight = 0
		k = s
		for j in i:
			current_pathweight += graph[k][j]
			k = j
		current_pathweight += graph[k][s]
		if current_pathweight < min_path:
			min_path = current_pathweight
			path = [s] + list(i) + [s]
	
	print("Minimum Cost: ", min_path)
	print("Path Taken: ", path)
	return min_path


graph = [[0, 10, 15, 20], [10, 0, 35, 25],
			[15, 35, 0, 30], [20, 25, 30, 0]]
s = 0
travellingSalesmanProblem(graph, s)
print('\n')
################## task 1 ends ######################33

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}
print("DFS graph traversal")
dfs(graph, '0')

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def dfs_preorder(root):
    if not root:
        return
    print(root.val, end=' ')
    dfs_preorder(root.left)
    dfs_preorder(root.right)


root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right = TreeNode(3)
root.right.right = TreeNode(6)
print("DFS tree traversal")
dfs_preorder(root)

print('\n')
######################## task 2 ends ############################


import copy
from queue import Queue
from collections import deque 
class states:
   
    def __init__(self, representation):
        self.representation = representation

    def slide_left(self):
        new_state = states(copy.deepcopy(self.representation))
        empty_space = self.representation.index(0)
        if empty_space == 0 or empty_space == 3 or empty_space == 6:
            return new_state
        else:
            new_state.representation[empty_space - 1], new_state.representation[empty_space] = new_state.representation[empty_space], new_state.representation[empty_space - 1]
            return new_state

    def slide_right(self):
        new_state = states(copy.deepcopy(self.representation))
        empty_space = self.representation.index(0)
        if empty_space == 2 or empty_space == 5 or empty_space == 8:
            return new_state
        else:
            new_state.representation[empty_space+1], new_state.representation[empty_space] = new_state.representation[empty_space], new_state.representation[empty_space+1]
            return new_state

    def slide_up(self):
        new_state = states(copy.deepcopy(self.representation))
        empty_space = self.representation.index(0)
        if empty_space == 0 or empty_space == 1 or empty_space == 2:
            return new_state
        else:
            new_state.representation[empty_space-3], new_state.representation[empty_space] = new_state.representation[empty_space], new_state.representation[empty_space-3]
            return new_state

    def slide_down(self):
        new_state = states(copy.deepcopy(self.representation))
        empty_space = self.representation.index(0)
        if empty_space == 6 or empty_space == 7 or empty_space == 8:
            return new_state
        else:
            new_state.representation[empty_space+3], new_state.representation[empty_space] = new_state.representation[empty_space], new_state.representation[empty_space+3]
            return new_state

    def print_board(self):
        for i in range(len(self.representation)):
            if i%3 ==0:
                print('\n', end=' ')
            print(self.representation[i], end=' ')
        print('\n\n\n')

    def applyOperators(self):
       
        return [self.slide_left(), self.slide_right(), self.slide_up(), self.slide_down()]

def BFS(start_state, goal_state):

    queue = Queue() 
    queue.put(start_state) 
    while queue:
        path = queue.get() 
        entity = states(path) 
        leaves = entity.applyOperators() 

        if path == goal_state:
            return path

        for child in leaves:
            if (child.representation != path):
                queue.put(child.representation)

def DFS(start_state, goal_state):

    stack = deque()
    stack.append(start_state)

    while stack:
        path = stack.pop()
        entity = states(path)
        leaves = entity.applyOperators()

        if path == goal_state:
            return path

        for child in leaves:
            if child.representation != path:
                stack.append(child.representation)

start_state = [1,2,3,4,0,5,7,8,6]
goal_state = [1,2,3,4,5,6,7,8,0] 
Final=BFS(start_state, goal_state)
print ('Using BFS')
print(Final)
obj=states(Final)
obj.print_board()
print('\n')
start_state = [1,2,3,8,0,4,7,6,5]
goal_state = [1,2,3,8,6,4,7,0,5]
Final=DFS(start_state, goal_state)
print ('Using DFS')
print(Final)
obj=states(Final)
obj.print_board()




