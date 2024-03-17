import numpy as np
import pandas
import csv

def example_1 ():
    a, b, c, d = 12, 2, 5, -3  
    print ("Example # 1")
    print("Addition : " , a + b )
    print("Subtraction : " , a - b )
    print("Multiplication : " , a * b )
    print("Division : " , a / b )
    print("Power : " , a ** b )
    print("Modulus : " , a % c )
    print("Absolute : " , abs(d) )
    
def example_2 ():
    print("Example # 2")
    angle = np.pi/4
    print(np.pi)
    print(np.e)
    print(np.sin(angle))
    print(np.cos(angle))
    print(np.tan(angle))
    
def example_3 ():
    print("Example # 3")
    name = input ("Enter employee name : ")
    salary = input ("Enter employee salary: ")
    company = input ("Enter company name: ")
    
    print("Printing Employee Details")
    print("Name","Salary","Company")
    print(name ,  salary , company)
    
def example_4 ():
    print("Example # 4")
    data = []
    print("Tell me about yourself :")
    while True:
        line = input()
        if line:
            data.append(line)
        else:
            break
    finaltext = '\n'.join(data)
    print("\n")
    print("Final text input :")
    print(finaltext)
    
def example_5 ():
    print("Example # 5")
    var1 = "Hello World!"
    var2 = "Python Programming"
    print ("var1[0] : ", var1[0])
    print("var2[1:5] : ",var2[1:5])
    print("Complete : ", var1)
    
def example_6 ():
    print("Example # 6")
    list1 =['physics','chemistry',1997,2000]
    list2 = [1,2,3,4,5,6,7]
    print("list1[0] : ",list1[0])
    print("list2[1:5] : ",list2[1:5])
    print(list1)
    
def example_7 (str1 , str2):
    print("Example # 7")
    print(str1)
    print(str2)
    return

def example_8 (mylist):
    print("Example # 8")
    mylist.append([1,2,3,4])
    print("values inside the function : ", mylist)
    return

def example_9 ():
    print("Example # 9")
    a, b, c, d = 12, -4, 3+4j, 7.9
    print(abs(a))
    print(abs(b))
    print(abs(c))
    print(abs(d))
 
def example_10 ():
    print("Example # 10") 
    stringobj = 'PALINDROME'
    print(len(stringobj))
    tupleobj = ('a','e','i','o','u')
    print(len(tupleobj))
    listobj = ['1','2','3','o','10u']
    print(len(listobj))
    
def example_11 ():
    print("Example # 11") 
    tup1 =['physics','chemistry',1997,2000]
    tup2 = [1,2,3,4,5,6,7]
    print("tup1[0] : ",tup1[0])
    print("tup2[1:5] : ",tup2[1:5])
    print(tup1) 

def example_12():
    print("Example # 12")
    tup1 = (12,34,56)
    tup2 = ('abc' , 'xyz')
    # tup1[0] = 100   ==> wrong as tuple cant be modified after declaration
    tup3 = tup1 + tup2
    print(tup3) 

def example_13():
    print("Example # 13")
    dict1 = {'Name' : 'Zara' , 'Age' : 7 , 'Class' : 'First'}
    dict1['Age'] = 8
    dict1['School'] = "DPS School"
    
    print("dict1['Age] : ", dict1['Age'])
    print("dict1['School'] : ", dict1['School'])    
    
def example_14():
    print("Example # 14")
    dict1 = {'Name' : 'Zara' , 'Age' : 7 , 'Class' : 'First'}
    print("dict1['Age] : ", dict1['Age'])
    print("dict1['Name'] : ", dict1['Name'])  
    
def example_15():
    print("Example # 15")
    count = 0
    while (count < 5 ):
        print (count , " is less than 5")
        count = count + 1
    else:
        print(count , " is not less than 5")
        return
    
def example_16():
    print("Example # 16")
    for letter in 'python':
        print ("Current Letter : ", letter)
    
    fruits = ['banana','apple','mango']
    for fruit in fruits:
        print ("Current fruit : ", fruit)
    print("Good bye")
    
def example_17():
    print("Example # 17")
    var1 = 100
    if var1:
        print("1 - Got a true expression value")
        print(var1)
    var2 = 0
    if var2:
        print("2 - Got a true expression value")
        print(var2)
    print("Good bye")
    
def example_18():
    print("Example # 18")
    var1 = 100
    if var1:
        print("1 - Got a true expression value")
        print(var1)
    else:
        print("1 - Got a false expression value")
        print(var1)
    var2 = 0
    if var2:
        print("2 - Got a true expression value")
        print(var2)
    else:
        print("2 - Got a false expression value")
        print(var2)
    print("Good bye")
    
def example_19_1(number):
    if (number == 1):
        return False
    
    for factor in range(2, number):
        if (number % factor == 0 ):
            return False
    return True

def example_19(n):
    for number in range (1,n):
        if (example_19_1(number)):
            print ("%d is prime" % number)
            
    return

def example_20(x):
    if (x==1):
        return 1
    else:
        return (x * example_20(x-1))
    
def example_21(filename):
    fields = ['Name' ,'Campus' ,'Year' , 'GPA']
    rows = [
            ['John','KHI','2','3.0'],
            ['Bret','KHI','2','2.1'],
            ['Tom','KHI','2','3.8']
           ]
    
    with open (filename,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
        
    return
    
if __name__ == '__main__':
    example_1()
    print("------------------")
    example_2()
    print("------------------")
    example_3()
    print("------------------")
    example_4()
    print("------------------")
    example_5()
    print("------------------")
    example_6()
    print("------------------")
    example_7("I'm first call to user defined function" , "Again second call to same function")
    print("------------------")
    mylist = [10,20,30]
    print("Old values outside function : " ,mylist)
    example_8(mylist)
    print("New values outside function : " ,mylist)
    print("------------------")
    example_9()
    print("------------------")
    example_10()
    print("------------------")
    example_11()
    print("------------------")
    example_12()
    print("------------------")
    example_13()
    print("------------------")
    example_14()
    print("------------------")
    example_15()
    print("------------------")
    example_16()
    print("------------------")
    example_17()
    print("------------------")
    example_18()
    print("------------------")
    n = 10
    print("Example # 19")
    example_19(n)
    print("------------------")
    print("Example # 20")
    print("The factorial of", n , " is " ,example_20(n))
    print("------------------")
    print("Example # 21")
    filename = "university_records.csv"
    example_21(filename)
    df = pandas.read_csv(filename)
    print(df)