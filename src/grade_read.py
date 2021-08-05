'''
   this file contains function read csv
'''
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


def fetch_data():
    '''
        this function can fetch the dataset file
    '''
    path = "https://raw.githubusercontent."\
            "com/EKU-Summer-2021/ml-assignmen"\
            "t-YangYizhou99/master/student-mat.csv"
    urllib.request.urlretrieve(path, "student-mat.csv")

def load_data():
    '''
        this function can load csv file data
    '''
    fetch_data()
    data=pd.read_csv("student-mat.csv")
    target = data.G3.copy()
    data=data.drop('G3',axis=1)
    data = data.drop("address", axis=1)
    data_cat = data[["school","sex","famsize","Pstatus","Mjob","Fjob",\
                     "reason","guardian",\
                     "schoolsup","famsup","paid","activities",\
                     "nursery","higher","internet","romantic"]]
    data_num = data.drop(["school","sex","famsize","Pstatus","Mjob","Fjob",\
                     "reason","guardian",\
                     "schoolsup","famsup","paid","activities",\
                     "nursery","higher","internet","romantic"], \
                         axis=1).to_numpy()
    cat_encoder = OneHotEncoder(sparse=False)
    data_cat = cat_encoder.fit_transform(data_cat)
    data=np.concatenate((data_num, data_cat), axis=1)
    input_train, input_test, target_train, target_test = train_test_split(data, target, test_size=0.15, random_state=42)
    return input_train, input_test, target_train, target_test
