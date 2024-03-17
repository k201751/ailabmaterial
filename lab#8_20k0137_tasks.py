import random
import numpy as np
import matplotlib.pyplot as plt

print("Task # 1\n")
def roll_the_dice(simulations= 100):
  count= 0
  for i in range(simulations):
    dice1= random.randint(1,6)
    dice2= random.randint(1,6)
    
    total = dice1+ dice2
    if total % 2 == 0 or total > 7:
      count+=1
  plt.hist(count/simulations*100, count)
  plt.show()
  return count/simulations

string = 'The probability of rolling an even number or greater than 7 is: '
print(string, np.round(roll_the_dice()*100, 2), '%')

################################### TASK # 1 ENDS ###################
print("\nTask # 2\n")

d = {}
for i in range(60):
  if i <10:
    d[i] = 'white'
  elif i> 9 and i<30:
    d[i]= 'red'
  else:
    d[i]= 'green'

simulations= 1000
partA = 0
partB= 0

for i in range(simulations):
  lst= []
  for i in range(5):
    lst.append(d[random.randint(0, 59)])

  lst= np.array(lst)

  white = sum(lst == 'white')
  red = sum(lst == 'red')
  green = sum(lst == 'green')

  if white == 3 and red ==2:
    partA+=1

  if red == 5 or white == 5 or green == 5:
    partB+=1

print('The probablity of 3 whites and 2 reds : ', np.round(partA/simulations*100,2), '%')
print('The probablity of all the balls of same color : ', np.round(partB/simulations*100,2), '%')

################################### TASK # 2 ENDS ###################
print("\nTask # 3\n")

ss= [          ['B', 'B', 'B', 'B'], 
               ['B', 'B', 'B', 'G'], 
               ['B', 'B', 'G', 'B'],
               ['B', 'G', 'B', 'B'],
               ['G', 'B', 'B', 'B'],
               ['G', 'G', 'B', 'B'],
               ['G', 'B', 'B', 'G'],
               ['B', 'B', 'G', 'G'],
               ['B', 'G', 'B', 'G'],
               ['G', 'B', 'G', 'B'],
               ['B', 'G', 'G', 'B'],
               ['B', 'G', 'G', 'G'],
               ['G', 'G', 'G', 'B'],
               ['G', 'B', 'G', 'G'],
               ['G', 'G', 'B', 'G'],
               ['G', 'G', 'G', 'G']] 

def calculate(): 
    count = 0
    for i in range(16):
        count1 = 0
        for j in range (4): 
            if ss[i][j]=='B': 
                count1=count1+1
        if count1==2: 
            count=count+1
    
    print("Probability of two boys: ", count/16*100 , '%')
calculate()


################################### TASK # 3 ENDS ###################
print("\nTask # 4\n")

import re
ss=['HH', 'TH', 'HT', 'TT']
def calculate():
    count=0
    for i in range(len(ss)):
        x=ss[i]
        if (re.search('[H]', x)):
            count=count+1
            
    print("The probability of obtaining at least one head: " , count/len(ss))
calculate()

################################### TASK # 4 ENDS ###################
print("\nTask # 5\n")

odd=1/6
even=2/6
def calculate():
    probability=0
    for i in range(1,7):
        if i<4: 
            if i%2==0: 
                probability=probability+even
            else: 
                probability=probability+odd
    print("The probability of event E is: ", probability)
    
calculate()

################################### TASK # 5 ENDS ###################
print("\nTask # 6\n")

a = []
b = []

total= 0

for i in range(1000):
    total = total + 1
    if i%2 == 0:
        a.append(i)
    if i%3 == 0:
        b.append(i)

count1 = 0
count2 = 0

r1 = len(a)
r2 = len(b)

if r1 > r2:
    for i in range(r1):
        count1 = count1 +1
    for i in range(r2):
        if a is not b[i]:
            count1 = count1 + 1

for i in range(r1):
    for j in range(r2):
        if a[i] == b[j]:
            count2 = count2+1

print("P(a u b) is: ",count1/total*100,"%")
print("P(a n b) is: ",count2/total*100,"%")


################################### TASK # 6 ENDS ###################
print("\nTask # 7\n")

PDP1 = 0.01
PDP2 = 0.03
PDP3 = 0.02

P1 = 0.3
P2 = 0.2
P3 = 0.5

ProbabilityP1 = (PDP1*P1)/((PDP1*P1) + (PDP2*P2) + (PDP3*P3))
ProbabilityP2 = (PDP2*P2)/((PDP1*P1) + (PDP2*P2) + (PDP3*P3))
ProbabilityP3 = (PDP3*P3)/((PDP1*P1) + (PDP2*P2) + (PDP3*P3))

print("Plan # 1: " , ProbabilityP1)
print("Plan # 2: " , ProbabilityP2)
print("Plan # 3: " , ProbabilityP3)
if (ProbabilityP1 > ProbabilityP2 ) and (ProbabilityP1 > ProbabilityP3):
    print("Part 1 will be responsible.")
elif (ProbabilityP2 > ProbabilityP1) and (ProbabilityP2 > ProbabilityP3):
    print("Part 2 will be responsible.")
elif (ProbabilityP3 > ProbabilityP1) and (ProbabilityP3 > ProbabilityP2):
    print("Part 3 will be responsible.")

