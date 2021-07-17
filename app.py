import sys
from time import sleep

# sys.path.append("C:/Python_Projects/Analytics_4/computervision-recipes/")

import os
from pathlib import Path
import warnings

from utils_cv.action_recognition.dataset import VideoDataset
from utils_cv.action_recognition.model import VideoLearner 
from utils_cv.common.gpu import system_info
# from utils_cv.common.data import data_path
from flask import Flask,render_template,request
import pandas as pd

system_info()
warnings.filterwarnings('ignore')
pred_class = []

# DATA_PATH = "C:\Python_Projects\Analytics_4\SVW\Videos"
DATA_PATH = "C:\Python_Projects\Analytics_4\deploy\SVW_dataset"

classes_df = pd.read_csv("classes.csv")
classes = classes_df['Classes'].to_list()

print(classes)

data = VideoDataset(DATA_PATH)

learner = VideoLearner(data, num_classes=30)

# learner.load("C:/Users/parab/Analytics_4/checkpoints/r2plus1d_34_8_ig65m_003")
learner.load("C:/Python_Projects/Analytics_4/deploy/model/r2plus1d_34_8_ig65m_003")

app =Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction", methods = ['POST'])
def prediction():

    vid = request.files['vid']

    # print(vid)

    vid.save("vid.mp4")

    pred_class= learner.predict_video("vid.mp4")

    print(pred_class) 

    return render_template("prediction.html", data=pred_class)

    # return render_template("prediction.html", data="bye")

if __name__=="__main__":
    app.run(debug=True)
    