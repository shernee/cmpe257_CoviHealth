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

classifier_name = [
    'Gaussian Naive Bayes',
    'KNeighbors Classifier',
    'Support Vector Classifier', 
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'AdaBoost Classifier',
    'Gradient Boosting Classifier'
]
classifiers = [
    GaussianNB(),
    KNeighborsClassifier(algorithm='ball_tree', n_neighbors=4),
    SVC(kernel='rbf', C=0.05, random_state=42),
    DecisionTreeClassifier(max_depth=2, random_state=42),
    RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42),
    AdaBoostClassifier(n_estimators=10, learning_rate=0.01, random_state=42),
    GradientBoostingClassifier(n_estimators=10, max_depth=2, learning_rate=0.1, random_state=42)
]

def scaling(df):

  # Create train and test set
  # df = pd.read_csv(df)
  X = df.drop(['area', 'infection_level'], axis=1)
  y = df['infection_level']

  # Scaling
  sc = MinMaxScaler()
  X_scaled = sc.fit_transform(X)
  X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

  return X_scaled, y

def compute_samples(mod_percent, high_percent):

  # Compute number of records as per the input received from the user
  low = initial_low 
  mod = initial_mod * (1 + mod_percent/100)
  high = initial_high * (1 + high_percent/100)
  
  return low, int(mod), int(high)

def oversample(mod_percent, high_percent, df):

  X_scaled, y = scaling(df)
  low, mod, high = compute_samples(mod_percent, high_percent)
  
  # Minority oversampling and shuffling of dataset
  s_dict = {'Low':low, 'Moderate':mod, 'High':high}
  sm = SMOTE(sampling_strategy=s_dict, random_state=42)
  X_res, y_res = sm.fit_resample(pd.DataFrame(X_scaled, columns=X_scaled.columns), y)

  # Concat for reshuffling
  df_res = pd.concat([X_res, y_res], axis=1)

  # Reshuffle data and split in X and y
  df_res = df_res.reindex(np.random.permutation(df_res.index))
  X_res, y_res = df_res.iloc[:, :-1], df_res.iloc[:, -1]

  return X_res, y_res

def top_features(X: pd.DataFrame, y:pd.Series):
  # X_scaled = scaling(X)
  rf_clf = RandomForestClassifier(min_samples_split=5, random_state=1)
  rf_clf.fit(X, y)

  imp_features = pd.Series(data=rf_clf.feature_importances_, index=X.columns).sort_values(ascending=False)

  top3 = list(imp_features.index[:3])
  top10 = list(imp_features.index[:10])

  return top3, top10

def choose_best_classifier(X, y, classifier_name, classifiers):

  max_weigted_score = 0.0

  for name, clf in zip(classifier_name, classifiers):
    cv_results = cross_validate(clf, X, y, cv=15, scoring=('f1_weighted'), return_train_score=True)

    train_score = np.mean(cv_results['train_score'])
    test_score = np.mean(cv_results['test_score'])

    model_weighted_score = ((0.8*test_score)+(0.2*train_score))/2
   
    if model_weighted_score >= max_weigted_score:
      max_weigted_score = model_weighted_score
      classifier_name = name

  return classifier_name

def controller(mod_percent, high_percent, df):
  X, y = oversample(mod_percent, high_percent, df)
  top3, top10 = top_features(X, y)
  classifier = choose_best_classifier(X, y, classifier_name, classifiers)

  idx = classifier_name.index(classifier)

  model = classifiers[idx]
  model.fit(X, y)
  pred = model.predict(X)

  f1 = f1_score(y, pred, average='weighted')
  cf = confusion_matrix(y, pred)
  print(cf)
  return f1, cf, y.value_counts(), y, top3, classifier
