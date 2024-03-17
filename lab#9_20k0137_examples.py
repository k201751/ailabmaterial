print("\n\nExample # 1\n")

#Import required packages
import math
from pomegranate import *
from pomegranate.distributions import DiscreteDistribution

 
# Initially the door selected by the guest is completely random
guest =DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
 
# The door containing the prize is also a random process
prize =DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
 
# The door Monty picks, depends on the choice of the guest and the prize door
monty =ConditionalProbabilityTable(
[[ 'A', 'A', 'A', 0.0 ],
[ 'A', 'A', 'B', 0.5 ],
[ 'A', 'A', 'C', 0.5 ],
[ 'A', 'B', 'A', 0.0 ],
[ 'A', 'B', 'B', 0.0 ],
[ 'A', 'B', 'C', 1.0 ],
[ 'A', 'C', 'A', 0.0 ],
[ 'A', 'C', 'B', 1.0 ],
[ 'A', 'C', 'C', 0.0 ],
[ 'B', 'A', 'A', 0.0 ],
[ 'B', 'A', 'B', 0.0 ],
[ 'B', 'A', 'C', 1.0 ],
[ 'B', 'B', 'A', 0.5 ],
[ 'B', 'B', 'B', 0.0 ],
[ 'B', 'B', 'C', 0.5 ],
[ 'B', 'C', 'A', 1.0 ],
[ 'B', 'C', 'B', 0.0 ],
[ 'B', 'C', 'C', 0.0 ],
[ 'C', 'A', 'A', 0.0 ],
[ 'C', 'A', 'B', 1.0 ],
[ 'C', 'A', 'C', 0.0 ],
[ 'C', 'B', 'A', 1.0 ],
[ 'C', 'B', 'B', 0.0 ],
[ 'C', 'B', 'C', 0.0 ],
[ 'C', 'C', 'A', 0.5 ],
[ 'C', 'C', 'B', 0.5 ],
[ 'C', 'C', 'C', 0.0 ]], [guest, prize] )
 
d1 = State( guest, name="guest" )
d2 = State( prize, name="prize" )
d3 = State( monty, name="monty" )
 
#Building the Bayesian Network
network = BayesianNetwork( "Solving the Monty Hall Problem With Bayesian Networks" )
network.add_states(d1, d2, d3)
network.add_edge(d1, d3)
network.add_edge(d2, d3)
network.bake()

beliefs = network.predict_proba({ 'guest' : 'A' })
beliefs = map(str, beliefs)
print("n".join( "{}t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))

beliefs = network.predict_proba({'guest' : 'A', 'monty' : 'B'})
print("n".join( "{}t{}".format( state.name, str(belief) ) for state, belief in zip( network.states, beliefs )))
 

####################################### EXAMPLE # 1 ENDS ###################################
print("\n\nExample # 2\n")
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
# define the DBN structure
dbn = DBN()

# define the nodes (variables) and their temporal dependencies
dbn.add_nodes_from(['Environment', 'Temperature', 'Humidity', 'Pressure', 'Sensor1','Sensor2', 'Sensor3'])
dbn.add_edges_from([('Environment', 'Temperature'), ('Environment', 'Humidity'),
('Temperature', 'Pressure'), ('Humidity', 'Pressure'),
('Temperature', 'Sensor1'), ('Humidity', 'Sensor2'), ('Pressure', 'Sensor3')])
# define the temporal structure
dbn.add_temporal_edges([('Environment', 'Environment'), ('Temperature', 'Temperature'),
('Humidity', 'Humidity'), ('Pressure', 'Pressure'),
('Sensor1', 'Sensor1'), ('Sensor2', 'Sensor2'), ('Sensor3', 'Sensor3')])
# define the initial distributions for the nodes
environment_init = TabularCPD(variable='Environment', variable_card=2, values=[[0.5],
[0.5]])
temperature_init = TabularCPD(variable='Temperature', variable_card=2, values=[[0.5],
[0.5]])
humidity_init = TabularCPD(variable='Humidity', variable_card=2, values=[[0.5], [0.5]])
pressure_init = TabularCPD(variable='Pressure', variable_card=2, values=[[0.5], [0.5]])
sensor1_init = TabularCPD(variable='Sensor1', variable_card=2, values=[[0.5], [0.5]])
sensor2_init = TabularCPD(variable='Sensor2', variable_card=2, values=[[0.5], [0.5]])
sensor3_init = TabularCPD(variable='Sensor3', variable_card=2, values=[[0.5], [0.5]])
dbn.add_cpds(environment_init, temperature_init, humidity_init, pressure_init, sensor1_init,
sensor2_init, sensor3_init)
from pgmpy.sampling import DynamicBayesianNetworkSampler as DBNSampler
# generate a sample from the DBN
dbn_sampler = DBNSampler(dbn)
sample = dbn_sampler.forward_sample(n_samples=10)
print(sample)

####################################### EXAMPLE # 2 ENDS ###################################
print("\n\nExample # 3\n")

import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
# create a new dynamic Bayesian network
dbn = gdyn.DynBayesNet()
# define the nodes (variables) and their temporal dependencies
dbn.addNode("Environment", 2)
dbn.addNode("Temperature", 2)
dbn.addNode("Humidity", 2)
dbn.addNode("Pressure", 2)
dbn.addNode("Sensor1", 2)
dbn.addNode("Sensor2", 2)
dbn.addNode("Sensor3", 2)
dbn.addArc("Environment", "Temperature")
dbn.addArc("Environment", "Humidity")
dbn.addArc("Temperature", "Pressure")
dbn.addArc("Humidity", "Pressure")
dbn.addArc("Temperature", "Sensor1")
dbn.addArc("Humidity", "Sensor2")
dbn.addArc("Pressure", "Sensor3")
# define the initial distributions for the nodes
environment_init = gum.Prior(0.5)
temperature_init = gum.Prior(0.5)
humidity_init = gum.Prior(0.5)
pressure_init = gum.Prior(0.5)
sensor1_init = gum.Prior(0.5)
sensor2_init = gum.Prior(0.5)
sensor3_init = gum.Prior(0.5)
dbn.cpt("Environment")[0] = environment_init.toarray()
dbn.cpt("Environment")[1] = 1 - environment_init.toarray()
dbn.cpt("Temperature")[0] = temperature_init.toarray()
dbn.cpt("Temperature")[1] = 1 - temperature_init.toarray()
dbn.cpt("Humidity")[0] = humidity_init.toarray()
dbn.cpt("Humidity")[1] = 1 - humidity_init.toarray()
dbn.cpt("Pressure")[0] = pressure_init.toarray()
dbn.cpt("Pressure")[1] = 1 - pressure_init.toarray()
dbn.cpt("Sensor1")[0] = sensor1_init.toarray()
dbn.cpt("Sensor1")[1] = 1 - sensor1_init.toarray()
dbn.cpt("Sensor2")[0] = sensor2_init.toarray()
dbn.cpt("Sensor2")[1] = 1 - sensor2_init.toarray()
dbn.cpt("Sensor3")[0] = sensor3_init.toarray()
dbn.cpt("Sensor3")[1] = 1 - sensor3_init.toarray()
# generate a sample from the DBN
dbn.generate(10)
print(dbn)

####################################### EXAMPLE # 3 ENDS ###################################
print("\n\nExample # 4\n")

from hmmlearn import hmm
import numpy as np
# define the model
model = hmm.MultinomialHMM(n_components=5)
# specify the parameters of the model
model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
model.transmat_ = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
[0.5, 0.0, 0.5, 0.0, 0.0],
[0.0, 0.5, 0.0, 0.5, 0.0],
[0.0, 0.0, 0.5, 0.0, 0.5],
[0.0, 0.0, 0.0, 0.0, 1.0]])
model.emissionprob_ = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.5, 0.5, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 0.0],
[0.0, 0.0, 0.5, 0.5, 0.0],
[0.0, 0.0, 0.0, 0.0, 1.0]])
# generate a sequence of observations from the model
X, state_sequence = model.sample(n_samples=10)

# fit the model to the data
model.fit(X)
# predict the most likely sequence of hidden states for a new sequence of observations
logprob, state_sequence = model.decode(X)