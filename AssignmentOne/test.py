

 

import pandas as pd


def task_1():
    movies = pd.read_excel("movie_reviews.xlsx")
    df = pd.DataFrame(movies, columns = ['Sentiment', 'Review', 'Split'])

    train_data = df[df['Split'] == "train"]
    train_data = train_data['Review']


    train_labels = df[df['Split'] == "train"]
    train_labels = train_labels['Sentiment']


    test_data = df[df['Split'] == "test"]
    test_data = test_data['Review']

    test_labels = df[df['Split'] == "test"]
    test_labels = test_labels['Sentiment']

    print(train_labels.value_counts())
    print(test_labels.value_counts())

    return train_data, train_labels,  test_data, test_labels

def task_2():
    task_1_data = task_1()
    #test =task_1_data[0].str.replace('[^a-zA-Z0-9 ]', ' ').str.lower().str.split().tolist()
    test =task_1_data[0].str.replace('[^a-zA-Z0-9 ]', ' ').str.lower().str.split(expand=True).stack().value_counts()
   # task_1_data[0].Series(' '.join(df['text']).lower().split()).value_counts()[:100]
    # task_1_data[0].str.replace('[^a-zA-Z0-9 ]', '').str.lower()
  #  print(task_1_data[0].replace('[^a-zA-Z0-9 ]', '').str.lower().str.split(expand=True).stack().value_counts())
    print(test)
def main():
    task_2()
    
main()
#machine.txt
#Displaying machine.txt.