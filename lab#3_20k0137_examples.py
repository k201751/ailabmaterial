from aima3.agents import *
from aima3.notebook import psource
import collections
collections.Callable = collections.abc.Callable

loc_A = (0, 0)
loc_B = (1, 0)

print("Example # 1 : SIMPLE REFLEX AGENT")
def SimpleReflexAgentProgram():

	def program(percept): 
		loc, status = percept
		return ('Suck' if status == 'Dirty' else'Right' if loc == loc_A else'Left')
	return program
# Create a simple reflex agent the two-state environment 
program = SimpleReflexAgentProgram() 

simple_reflex_agent = Agent(program)

# These are the two locations for the two-state environment
loc_A, loc_B = (0, 0), (1, 0)
# Initialize the two-state environment
trivial_vacuum_env = TrivialVacuumEnvironment()
# Check the initial state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))
trivial_vacuum_env.add_thing(simple_reflex_agent)
print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))


for x in range(3):
# Run the environment
	trivial_vacuum_env.step()
	print("State of the Environment: {}.".format(trivial_vacuum_env.status))
	print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))

print("\n\n")
trivial_vacuum_env.delete_thing(simple_reflex_agent)

print("Example # 2 : MODEL BASED REFLEX AGENT")
loc_A = (0, 0)
loc_B = (1, 0)

# These are the two locations for the two-state environment
loc_A, loc_B = (0, 0), (1, 0)
# Initialize the two-state environment
trivial_vacuum_env = TrivialVacuumEnvironment()



# Check the initial state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))
# TODO: Implement this function for the two-dimensional environment
def update_state(state, action, percept, model): 
	pass
	# Create a model-based reflex agent
	
model_based_reflex_agent = ModelBasedVacuumAgent()
# Add the agent to the environment
trivial_vacuum_env.add_thing(model_based_reflex_agent)
print("ModelBasedVacuumAgent is located at {}.".format(model_based_reflex_agent.location))
	
for x in range(3):
	# Run the environment
	trivial_vacuum_env.step()
	# Check the current state of the environment
	print("State of the Environment: {}.".format(trivial_vacuum_env.status))
	
	
print("ModelBasedVacuumAgent is located at {}.".format(model_based_reflex_agent.location))
print("\n\n")
trivial_vacuum_env.delete_thing(model_based_reflex_agent)


print("Example # 3 : GOAL BASED REFLEX AGENT")


loc_A = (0, 0)
loc_B = (1, 0)

def VacuumCleanerGoalAgentProgram():
    def program(percept):
        loc, status = percept
        if status == "Dirty":
            return "Suck"
        elif loc == loc_A:
            return "Right"
        elif loc == loc_B:
            return "Left"
        return "NoOp"
    return program

program = VacuumCleanerGoalAgentProgram()
goal_based_agent = Agent(program)
loc_A, loc_B = (0, 0), (1, 0)

trivial_vacuum_env = TrivialVacuumEnvironment()
print("State of the Environment: {}.".format(trivial_vacuum_env.status))
trivial_vacuum_env.add_thing(goal_based_agent)
print("GoalBasedVacuumAgent is located at {}.".format(goal_based_agent.location))
for x in range(3):
    trivial_vacuum_env.step()
    print("State of the Environment: {}.".format(trivial_vacuum_env.status))
    print("GoalBasedVacuumAgent is located at {}.".format(goal_based_agent.location))
print("\n\n")
trivial_vacuum_env.delete_thing(goal_based_agent)


print("Example # 4 : UTILITY BASED REFLEX AGENT")

loc_A = (0, 0)
loc_B = (1, 0)

def VacuumCleanerUtilityAgentProgram():
    def program(percept):
        loc, status = percept
        if status == "Dirty":
            return "Suck"
        elif loc == loc_A:
            if trivial_vacuum_env.status[loc_B] == "Dirty":
                return "Left"
            else:
                return "Right"
        elif loc == loc_B:
            if trivial_vacuum_env.status[loc_A] == "Dirty":
                return "Right"
            else:
                return "Left"
        return "NoOp"
    return program

program = VacuumCleanerUtilityAgentProgram()
utility_based_agent = Agent(program)
loc_A, loc_B = (0, 0), (1, 0)

trivial_vacuum_env = TrivialVacuumEnvironment()
print("State of the Environment: {}.".format(trivial_vacuum_env.status))
trivial_vacuum_env.add_thing(utility_based_agent)
print("UtilityBasedVacuumAgent is located at {}.".format(utility_based_agent.location))
for x in range(3):
    trivial_vacuum_env.step()
    print("State of the Environment: {}.".format(trivial_vacuum_env.status))
    print("UtilityBasedVacuumAgent is located at {}.".format(utility_based_agent.location))
print("\n\n")
trivial_vacuum_env.delete_thing(utility_based_agent)


print("Example # 5 : LEARNING BASED REFLEX AGENT")


class QLearningVacuumAgent(Agent):
    def __init__(self, alpha, epsilon, discount):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.last_state = None
        self.last_action = None
    
    def program(self, percept):
        loc, status = percept
        state = (loc, status)
        
        if self.last_state is not None:
            if self.last_state not in self.q:
                self.q[self.last_state] = {}
            if self.last_action not in self.q[self.last_state]:
                self.q[self.last_state][self.last_action] = 0
            
            reward = -1 if status == "Dirty" else 0
            max_q = max(self.q[state].values()) if state in self.q else 0
            q = self.q[self.last_state][self.last_action]
            self.q[self.last_state][self.last_action] = q + self.alpha * (reward + self.discount * max_q - q)
        
        if state not in self.q or random.random() < self.epsilon:
            action = random.choice(["Suck", "Left", "Right"])
        else:
            action = max(self.q[state], key=self.q[state].get)
        
        self.last_state = state
        self.last_action = action
        return action

loc_A, loc_B = (0, 0), (1, 0)
ql_agent = QLearningVacuumAgent(0.1, 0.1, 0.9)
trivial_vacuum_env = TrivialVacuumEnvironment()

trivial_vacuum_env.add_thing(ql_agent)
print("QLearningVacuumAgent is located at {}.".format(ql_agent.location))

for x in range(3):
    trivial_vacuum_env.step()
    print("State of the Environment: {}.".format(trivial_vacuum_env.status))
    print("QLearningVacuumAgent is located at {}.".format(ql_agent.location))
print("\n\n")
trivial_vacuum_env.delete_thing(ql_agent)