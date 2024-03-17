import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def example_1():
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)
    print(type(arr))
    #Create a 0-D array with value 42
    arr = np.array(42)
    print(arr)
    #Create a 1-D array containing the values 1,2,3,4,5:
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)
    #Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr)
    #Create a 3-D array with two 2-D arrays, both containing two arrays with the values 1,2,3 and 4,5,6:
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    print(arr)

def example_2():
    import numpy as np
    arr = np.array([1, 2, 3, 4])
    print(arr[1])
    print(arr[2] + arr[3])
    # Access 2D array:
    # Access the element on the first row, second column:
    arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print('2nd element on 1st row: ', arr[0, 1])
    arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print('5th element on 2nd row: ', arr[1, 4])
    # Access 3d Array:
    # Access the third element of the second array of the first array:
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(arr[0, 1, 2])
    # Negative Indexing:
    arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print('Last element from 2nd dim: ', arr[1, -1])

def example_3():
    # Slicing arrays:
    # Slicing in python means taking elements from one given index to another given index.
    # We pass slice instead of index like this: [start:end].
    # We can also define the step, like this: [start:end:step]
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5, 6, 7])
    # Slice elements from index 1 to index 5 from the following array:
    print(arr[1:5])

    # Slice elements from index 4 to the end of the array:
    print(arr[4:])
    # Slice elements from the beginning to index 4 (not included):
    print(arr[:4])
    # Negative Slicing:
    # Slice from the index 3 from the end to index 1 from the end:
    print(arr[-3:-1])
    # STEP
    # Use the step value to determine the step of the slicing:
    # Return every other element from index 1 to index 5:
    print(arr[1:5:2])
    # Return every other element from the entire array:
    print(arr[::2])
    # Slicing 2-D Arrays
    # From the second element, slice elements from index 1 to index 4 (not included):
    arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print(arr[1, 1:4])
    # From both elements, return index 2:
    print(arr[0:2, 2])
    # From both elements, slice index 1 to index 4 (not included), this will return a 2-D array:
    print(arr[0:2, 1:4])

def example_4():
    # Checking the Data Type of an Array
    # The NumPy array object has a property called dtype that returns the data type of the array:
    # Get the data type of an array object:
    arr = np.array([1, 2, 3, 4])
    print(arr.dtype)
    # Get the data type of an array containing strings:
    arr = np.array(['apple', 'banana', 'cherry'])
    print(arr.dtype)
    # Iterating Arrays
    # Iterating means going through elements one by one.
    # As we deal with multi-dimensional arrays in numpy, we can do this using basic for loop of python.
    # If we iterate on a 1-D array it will go through each element one by one.
    arr = np.array([1, 2, 3])
    for x in arr:
        print(x)
    # Iterate on each scalar element of the 2-D array:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    for x in arr:
        for y in x:
            print(y)
    # Iterate on the elements of the following 3-D array:
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    for x in arr:
        print(x)
    # To return the actual values, the scalars, we have to iterate the arrays in each dimension.
    # Iterate down to the scalars:
    for x in arr:
        for y in x:
            for z in y:
                print(z)

def example_5():
    # Plotting x and y points
    # The plot() function is used to draw points (markers) in a diagram.
    # By default, the plot() function draws a line from point to point.
    # The function takes parameters for specifying points in the diagram.
    # Parameter 1 is an array containing the points on the x-axis.
    # Parameter 2 is an array containing the points on the y-axis.
    # If we need to plot a line from (1, 3) to (8, 10), we have to pass two arrays [1, 8] and [3, 10] to the plot

    # Draw a line in a diagram from position (1, 3) to position (8, 10):

    xpoints = np.array([1, 8])
    ypoints = np.array([3, 10])
    plt.plot(xpoints, ypoints)
    plt.show()

def example_6():
    # Draw two points in the diagram, one at position (1, 3) and one in position (8, 10):
    xpoints = np.array([1, 8])
    ypoints = np.array([3, 10])
    plt.plot(xpoints, ypoints, 'o')
    plt.show()
    # Draw a line in a diagram from position (1, 3) to (2, 8) then to (6, 1) and finally to position (8, 10):
    xpoints = np.array([1, 2, 6, 8])
    ypoints = np.array([3, 8, 1, 10])
    plt.plot(xpoints, ypoints)
    plt.show()

def example_7():
    # You can use also use the shortcut string notation parameter to specify the marker.
    # This parameter is also called fmt, and is written with this syntax:
    # marker|line|color
    ypoints = np.array([3, 8, 1, 10])
    plt.plot(ypoints, 'o:r')
    plt.show()
    ypoints = np.array([3, 8, 1, 10])
    plt.plot(ypoints, linestyle='dotted')
    plt.show()

def example_8():
    x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
    y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
    plt.plot(x, y)
    plt.title("Sports Watch Data")
    plt.xlabel("Average Pulse")
    plt.ylabel("Calorie Burnage")
    plt.grid()
    plt.show()

def example_9():
    # Subplots
    # With the subplot() function you can draw multiple plots in one figure:
    # plot 1:
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 8, 1, 10])
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    # plot 2:
    x = np.array([0, 1, 2, 3])
    y = np.array([10, 20, 30, 40])
    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.show()

def example_10():
    import matplotlib.pyplot as plt
    import numpy as np
    # With Pyplot, you can use the bar() function to draw bar graphs:
    x = np.array(["A", "B", "C", "D"])
    y = np.array([3, 8, 1, 10])
    plt.plot(x, y)
    plt.show()
    # for horizontal bar use 'barh'
    plt.barh(x, y)
    plt.show()
    x = np.array(["A", "B", "C", "D"])
    y = np.array([3, 8, 1, 10])
    plt.bar(x, y, color="#4CAF50")
    plt.show()
    # histogram
    x = np.random.normal(170, 10, 250)
    plt.hist(x)
    plt.show()
    # Pie Chart
    y = np.array([35, 25, 25, 15])
    mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
    plt.pie(y, labels=mylabels, startangle=90)
    plt.show()

def example_11():
    df = pd.read_csv("data.csv")
    df.head()
    print(df.shape)
    print(df.columns)
    print(df.info())
    df.describe()
    df["Pulse"].mean()

def example_12():
    mydataset = {
        'cars': ["BMW", "Volvo", "Ford"],
        'passings': [3, 7, 2]
    }
    myvar = pd.DataFrame(mydataset)
    print(myvar)

def example_13():
    data = {
        "calories": [420, 380, 390],
        "duration": [50, 40, 45]
    }
    # load data into a DataFrame object:
    df = pd.DataFrame(data)

    print(df)
    # refer to the row index:
    print(df.loc[0])
    # use a list of indexes:
    print(df.loc[[0, 1]])

def example_14():
    # read CSV:
    df = pd.read_csv('data.csv')
    print(df)
    # Analyzing dataframe:
    # The head() method returns the headers and a specified number of rows, starting from the top.
    df = pd.read_csv('data.csv')
    # printing the first 10 rows of the DataFrame:
    print(df.head(10))
    # There is also a tail() method for viewing the last rows of the DataFrame.
    # The tail() method returns the headers and a specified number of rows, starting from the bottom.
    # Print the last 5 rows of the DataFrame:
    print(df.tail())
    # The DataFrames object has a method called info(), that gives you more information about the data set.
    # Print information about the data:
    print(df.info())
    # Cleaning Empty cell:
    new_df = df.dropna()
    # If you want to change the original DataFrame, use the inplace = True argument:
    # Remove all rows with NULL values:
    df.dropna(inplace=True)
    # The fillna() method allows us to replace empty cells with a value:
    # Replace NULL values with the number 130:
    df.fillna(130, inplace=True)
    # Replace NULL values in the "Calories" columns with the number 130:
    df["Calories"].fillna(130, inplace=True)


def example_15():
    import nltk
    nltk.download('punkt')


def example_16():
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    print(stopwords.words('english'))
    # The following program removes stop words from a piece of text:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    example_sent = """This is a sample sentence,
    showing off the stop words filtration."""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(example_sent)
    # converts the words in word_tokens to lower case and then checks whether
    # they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # with no lower case conversion
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    print(word_tokens)
    print(filtered_sentence)

def example_17():
    import spacy
    nlp = spacy.load('en_core_web_sm')
    sentence = "Apple is looking at buying U.K. startup for $1 billion"
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

def example_18():
    # First we need to import spacy
    import spacy
    # Creating blank language object then
    # tokenizing words of the sentence
    nlp = spacy.blank("en")
    doc = nlp("GeeksforGeeks is a one stop\
    learning destination for geeks.")
    for token in doc:
        print(token)

def example_19():
    # Here is an example to show what other functionalities can be enhanced by adding modules to the
    #pipeline.
    import spacy
    # loading modules to the pipeline.
    nlp = spacy.load("en_core_web_sm")
    # Initialising doc with a sentence.
    doc = nlp("If you want to be an excellent programmer \
    , be consistent to practice daily on GFG.")
    # Using properties of token i.e. Part of Speech and Lemmatization
    for token in doc:
        print(token, " | ",
              spacy.explain(token.pos_),
              " | ", token.lemma_)


def example_20():
    import spacy
    from spacy import displacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp ('Wall Street Journal published an interesting piece on crypto currencies.')
    displacy.render(doc, style='dep', jupyter=True , options={'distance':90})







if __name__ == '__main__':
    print("Example 1\n")
    example_1()
    print("Example 2\n")
    example_2()
    print("Example 3\n")
    example_3()
    print("Example 4\n")
    example_4()
    print("Example 5\n")
    example_5()
    print("Example 6\n")
    example_6()
    print("Example 7\n")
    example_7()
    print("Example 8\n")
    example_8()
    print("Example 9\n")
    example_9()
    print("Example 10\n")
    example_10()
    print("Example 11\n")
    example_11()
    print("Example 12\n")
    example_12()
    print("Example 13\n")
    example_13()
    print("Example 14\n")
    example_14()
    print("Example 15\n")
    example_15()
    print("Example 16\n")
    example_16()
    print("Example 17\n")
    example_17()
    print("Example 18\n")
    example_18()
    print("Example 19\n")
    example_19()
    print("Example 20\n")
    example_20()