#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from wordcloud import WordCloud
import nltk
nltk.download('all',quiet=True)
from PIL import Image

#Model libraries
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# mounting drive
from google.colab import drive
drive.mount('/content/drive')

#Assigning variable
df_orignal=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Capstone Projects /Twitter Sentiment Analysis/Coronavirus')
#copying data to preserve orignal file
df1=df_orignal.copy()
#checking Head
df1.head()
#checking info
df1.info()
#checking Columns
df1.columns
#For sentiment analysis we only want tweet and sentiment Features
df=df1[['OriginalTweet','Sentiment']]
df.head()
#Stastastical analysis of dataset
df.describe().T
#checking Unique values
df.Sentiment.unique()
#checking Shape of the dataset
df.shape
#check duplicate entries
len(df[df.duplicated()])