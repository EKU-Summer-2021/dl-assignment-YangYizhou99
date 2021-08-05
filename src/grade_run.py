'''
library
'''
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from src import grade_read,GradeModel


def grade_run():
    '''
    run the model
    '''
    input_train, input_test, target_train, \
    target_test = grade_read.load_data()
    model = GradeModel().model
    history=model.fit(input_train, target_train, epochs=30,
              validation_data=(input_test, target_test))
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    grade_path = "grade_result"
    if not os.path.exists(grade_path):
        os.mkdir(grade_path)
    path = grade_path + '/' + time_str
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + "/" + time_str + ".png")
    plt.show()
