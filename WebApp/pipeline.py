import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, confusion_matrix
#import holoviews as hv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

viz1 = pd.read_csv('viz_1.csv')
initial_low = viz1['infection_level'].value_counts()[0]
initial_mod = viz1['infection_level'].value_counts()[2]
initial_high = viz1['infection_level'].value_counts()[1]

def scaling(df):

  # Create train and test set
  df = pd.read_csv(df)
  X = df.drop(['area', 'infection_level'], axis=1)
  y = df['infection_level']

  # Scaling
  sc = MinMaxScaler()
  X_scaled = sc.fit_transform(X)

  return X_scaled, y

def compute_samples(mod_percent, high_percent):

  # Compute number of records as per the input received from the user
  low = initial_low 
  mod = initial_mod * (1 + mod_percent/100)
  high = initial_high * (1 + high_percent/100)
  
  return low, int(mod), int(high)

def oversample(mod_percent, high_percent, df):

  X_scaled, y = scaling(df)
  print(mod_percent)
  print(high_percent)
  low, mod, high = compute_samples(mod_percent, high_percent)
  
  # Minority oversampling and shuffling of dataset
  s_dict = {'Low':low, 'Moderate':mod, 'High':high}
  sm = SMOTE(sampling_strategy=s_dict, random_state=42)
  X_res, y_res = sm.fit_resample(pd.DataFrame(X_scaled), y)

  # Concat for reshuffling
  df_res = pd.concat([X_res, y_res], axis=1)

  # Reshuffle data and split in X and y
  df_res = df_res.reindex(np.random.permutation(df_res.index))
  X_res, y_res = df_res.iloc[:, :-1], df_res.iloc[:, -1]

  return X_res, y_res