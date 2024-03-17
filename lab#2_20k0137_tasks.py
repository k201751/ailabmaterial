import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy import displacy

def task_1():
    temp1 = np.array ([1,2,3,4])
    temp2 = np.array ([5,6,7,8])
    temp3 = np.array ([1,3,5,4])
    print(temp1 + temp2)
    print(temp1 * 10)
    reshaped = temp1.reshape((2,2))
    print(reshaped)
    changed_type  = temp2.astype('float64')
    print(changed_type)
    spaced_array = np.arange(0,100,2,'int32')
    print(spaced_array)
    print(np.where(temp1 == temp3))
    
def task_2():
    y1 = [0,1,2,3,4,5]  # will be used as x-axis as well
    y2= [0,2,4,6,8,10]
    
    plt.plot(y1,y1, label = "line 1")
    plt.plot(y1, y2, label = "line 2", linestyle="--")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.show()
    
    frac = [50,40,100,170]
    mylabels = ["A","B","C","D"]
    myexplode = [0.1,0.1,0.1,0.1]
    plt.pie(frac, labels = mylabels, startangle = 90 , explode = myexplode)
    plt.show()
    
    
def task_3():
    raw_data = {'Duration': [60,60,60,45,100],
                'Pulse': [110,117,103,109,117],
                'Max Pulse': [130,145,135,175,148],
                'Calories': [409.1,380,400,380.01,420]}
     
    df = pd.DataFrame(raw_data, columns = ['Duration','Pulse','Max Pulse','Calories'])
    df.to_csv('TextSheet.csv', index = False)
    print(df)
    new_df = pd.read_csv('TextSheet.csv')
    new_df.loc[3,'Pulse'] = 110
    new_df['Duration'] = new_df['Duration'].replace({100:60})
    new_df.to_csv('TextSheet.csv', index = False)
    print(new_df)
    
def get_valid_movements(vacuum_location, room):
    x, y = vacuum_location
    movements = []
    if x > 0 and room[x-1][y] != 'B':
        movements.append((x-1, y))
    if x < len(room) - 1 and room[x+1][y] != 'B':
        movements.append((x+1, y))
    if y > 0 and room[x][y-1] != 'B':
        movements.append((x, y-1))
    if y < len(room[0]) - 1 and room[x][y+1] != 'B':
        movements.append((x, y+1))
    return movements

def get_dirt_cells(vacuum_location, room):
    x, y = vacuum_location
    dirt_cells = []
    if x > 0 and room[x-1][y] == 'D':
        dirt_cells.append((x-1, y))
    if x < len(room) - 1 and room[x+1][y] == 'D':
        dirt_cells.append((x+1, y))
    if y > 0 and room[x][y-1] == 'D':
        dirt_cells.append((x, y-1))
    if y < len(room[0]) - 1 and room[x][y+1] == 'D':
        dirt_cells.append((x, y+1))
    return dirt_cells

def task_4 (room):
    vacuum_location = (0, 0)
    movement_counter = 0
    while True:
        dirt_cells = get_dirt_cells(vacuum_location, room)
        if not dirt_cells:
            valid_movements = get_valid_movements(vacuum_location, room)
            print("Valid movements available : " , valid_movements)
            if not valid_movements:
                break
            vacuum_location = random.choice(valid_movements)
            movement_counter += 1
            print ("Movement Count : " , movement_counter)
            print("Random vaccum location : " ,vacuum_location)
            if movement_counter >= 10:
                break
        else:
            dirt_cell = random.choice(dirt_cells)
            room[dirt_cell[0]][dirt_cell[1]] = 'C'
            vacuum_location = dirt_cell
    return room

def task_5_1():
    text = '''
    Joe waited for the train. The train was late.
    Mary and Samantha took the bus.
    I looked for Mary and Samantha at the bus station.
    '''
    print("\nOriginal string:")
    print(text)
    token_text = sent_tokenize(text)
    print("\nSentence-tokenized copy in a list:")
    print(token_text)
    print("\nRead the list:")
    for s in token_text:
        print(s)

def task_5_2():
    
    text = "Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for Mary and Samantha at the bus station."
    print("\nOriginal string:")
    print(text)
    print("\nList of words:")
    print(word_tokenize(text))
    
def task_5_3():
     
    text = "Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for Mary and Samantha at the bus station."
    print("\nOriginal string:")
    print(text)
    print("\nTokenize words sentence wise:")
    result = [word_tokenize(t) for t in sent_tokenize(text)]
    print("\nRead the list:")
    for s in result:
        print(s)

def task_6_1():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("This is a sentence.")
    displacy.serve(doc, style='dep', port= 5001)
    #important note : Although in the terminal the link for rendered image is shown to be : http://0.0.0.0:5001/ 
    # but that link will not work instead use http://localhost:5001/ to view the displacy result.
    
    
def task_6_2():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp('This is the first sentence. This is the second sentence.')
    for tok in doc:
        print(tok)
    
if __name__ == '__main__':
    print("\nTask # 1")
    task_1()
    print("\nTask # 2")
    task_2()
    print("\nTask # 3")
    task_3()
    print("\nTask # 4")
    room = [['D', 'D', 'D', 'D', 'D', 'D', 'D', 'B'],
        ['B', 'B', 'C', 'D', 'B', 'B', 'B', 'B'],
        ['D', 'D', 'C', 'D', 'B', 'D', 'D', 'D'],
        ['D', 'D', 'D', 'D', 'C', 'C', 'C', 'D'],
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
        ['D', 'D', 'D', 'B', 'B', 'D', 'D', 'D']]

    cleaned_room = task_4(room)
    for row in cleaned_room:
        print(row)

    print("\nTask # 5_1")
    task_5_1()
    print("\nTask # 5_2")
    task_5_2()
    print("\nTask # 5_3")
    task_5_3()
    print("\nTask # 6_2")
    task_6_2()
    print("\nTask # 6_1")
    task_6_1()
    