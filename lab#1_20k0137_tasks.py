import csv


def task_1(x):
    if (x <=1):
        return x
    else:
        return task_1(x-1) + task_1(x-2)
    
def task_2():
    lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]
    print (str( lst[3][1][2][0]))
    
def task_3():
    d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
    print (str(d['k1'][3]['tricky'][3]['target'][3]))
    
    
def task_4(speed, anniversary):
    if anniversary:
        speed -= 10
    if speed <= 70:
        return "No fine"
    elif speed <= 80:
        return "Less Fine"
    else:
        return "Car seize"

def task_5(arr):
    unique_arr = []
    for i in arr:
        if i not in unique_arr:
            unique_arr.append(i)
    if len(unique_arr) == len(arr):
        return False, unique_arr
    else:
        return True, unique_arr
    
def task_6(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data
    
def task_7(string):
    letter_count = 0
    digit_count = 0
    for char in string:
        if char.isalpha():
            letter_count += 1
        elif char.isdigit():
            digit_count += 1
    return letter_count, digit_count

def task_8():
    
    x = [1, 2, 3, 4]
    print("Before delete : ", x)
    del x[2] 
    print("After delete : ", x)
    temp = 5
    assert temp > 0, "x should be positive"

if __name__ == '__main__':
    print("Task # 1")
    print (task_1(10))
    print("--------------------")
    print("Task # 2")
    task_2()
    print("--------------------")
    print("Task # 3")
    task_3()
    print("--------------------")
    print("Task # 4")
    print (task_4(100,1))
    print("--------------------")
    print("Task # 5")
    print (task_5("hello"))
    print("--------------------")
    print("Task # 6")
    print (task_6("university_records.csv"))
    print("--------------------")
    print("Task # 7")
    print (task_7(input("Enter any alphanumeric string : ")))
    print("--------------------")
    print("Task # 8")
    task_8()
    print("--------------------")
    
    
    