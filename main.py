import pingouin as pg
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from matplotlib.pyplot import figure
import seaborn as sns

iris = load_iris()
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = iris_pd.drop(columns=['target'])
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, iris_pd[['target']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('main component 1', fontsize = 15)
ax.set_ylabel('main component 2', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['pink', 'yellow', 'orange']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#plt.show()

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_Components = tsne.fit_transform(data)
tsne_Df = pd.DataFrame(data=tsne_Components
                       , columns=['Component 1', 'Component 2'])
finalDf = pd.concat([tsne_Df, iris_pd[['target']]], axis=1)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('main component 1', fontsize=15)
ax.set_ylabel('main component 2', fontsize=15)
ax.set_title('t-SNE', fontsize=20)
targets = [0, 1, 2]
colors = ['pink', 'yellow', 'orange']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Component 1']
               , finalDf.loc[indicesToKeep, 'Component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
#plt.show()

from sklearn.model_selection import train_test_split
train, test = train_test_split(finalDf, test_size=0.2)

from sklearn.model_selection import cross_validate

def vis_data(X, Y, X_t, Y_t, pred, pred_t):
    pred = pd.Series(pred)
    pred_t = pd.Series(pred_t)
    """fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Data')
    fig.set_figheight(10)
    fig.set_figwidth(10)"""

    # train original
    Df = pd.DataFrame(data=X
                      , columns=['Component 1', 'Component 2'])
    finalDf = pd.concat([Df, Y], axis=1)
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Component 1', fontsize=15)
    ax1.set_ylabel('Component 2', fontsize=15)
    ax1.set_title('Реальные классы_train', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax1.scatter(finalDf.loc[indicesToKeep, 'Component 1']
                    , finalDf.loc[indicesToKeep, 'Component 2']
                    , c=color
                    , s=50)
    ax1.legend(targets)
    ax1.grid()
    #plt.show()

    # train pred

    Df = pd.DataFrame(data = X.reset_index(drop=True)
                , columns = ['Component 1', 'Component 2'])
    Df['target'] = pred
    fig = plt.figure(figsize = (8,8))
    ax2 = fig.add_subplot(1,1,1)
    ax2.set_xlabel('Component 1', fontsize = 15)
    ax2.set_ylabel('Component 2', fontsize = 15)
    ax2.set_title('Работа классификатора_train', fontsize = 20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets,colors):
        indicesToKeep = Df['target'] == target
        ax2.scatter(Df.loc[indicesToKeep, 'Component 1']
                  , Df.loc[indicesToKeep, 'Component 2']
                  , c = color
                  , s = 50)
    ax2.legend(targets)
    ax2.grid()
    #plt.show()

    # test original

    Df = pd.DataFrame(data=X_t
            , columns = ['Component 1', 'Component 2'])
    finalDf = pd.concat([Df, Y_t], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax3 = fig.add_subplot(1,1,1)
    ax3.set_xlabel('Component 1', fontsize=15)
    ax3.set_ylabel('Component 2', fontsize=15)
    ax3.set_title('Реальные классы_test', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
    ax3.scatter(finalDf.loc[indicesToKeep, 'Component 1']
            , finalDf.loc[indicesToKeep, 'Component 2']
            , c = color
            , s = 50)
    ax3.legend(targets)
    ax3.grid()
    #plt.show()

    # test pred

    Df = pd.DataFrame(data=X_t.reset_index(drop=True)
            , columns = ['Component 1', 'Component 2'])
    Df['target'] = pred_t
    fig = plt.figure(figsize=(8, 8))
    ax4 = fig.add_subplot(1,1,1)
    ax4.set_xlabel('Component 1', fontsize=15)
    ax4.set_ylabel('Component 2', fontsize=15)
    ax4.set_title('Работа классификатора_test', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = Df['target'] == target
    ax4.scatter(Df.loc[indicesToKeep, 'Component 1']
            , Df.loc[indicesToKeep, 'Component 2']
            , c = color
            , s = 50)
    ax4.legend(targets)
    ax4.grid()
    #plt.show()

def my_cross_val(X, Y, model, metrics, cv=4):
    cv_results = cross_validate(model, X, Y, cv=cv, scoring=metrics)
    print(model.predict)
    print(cv_results)
    return cv_results


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def train_model(train, test, model, metrics=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']):
    X = train.drop(columns=['target'])
    Y = train['target']
    X_t = test.drop(columns=['target'])
    Y_t = test['target']
    metrics = my_cross_val(X, Y, model, metrics)
    model.fit(X, Y)
    answers_t = model.predict(X_t)
    answers = model.predict(X)
    metrics_test = []
    metrics_test.append(accuracy_score(Y_t, answers_t))
    metrics_test.append(precision_score(Y_t, answers_t, average='weighted'))
    metrics_test.append(recall_score(Y_t, answers_t, average='weighted'))
    metrics_test.append(f1_score(Y_t, answers_t, average='weighted'))

    vis_data(X, Y, X_t, Y_t, answers, answers_t)
    return model, metrics, metrics_test

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
ans = train_model(train, test, neigh)
print(ans)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
ans = train_model(train, test, model)
print(ans)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
ans = train_model(train, test, model)
print(ans)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
ans = train_model(train, test, model)
print(ans)

from sklearn.model_selection import GridSearchCV
X = train.drop(columns=['target'])
Y = train['target']

parameters = {
  'n_neighbors' : (list(range(1, 30))),
  'weights' : ('uniform', 'distance'),
  'algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute'),
  'p' : [1,2,3,5,10,100],
  'metric' : ('minkowski', 'cityblock' , 'euclidean', 'manhattan')
}
"""neigh = KNeighborsClassifier()
clf = GridSearchCV(neigh, parameters)
clf.fit(X, Y)
print(clf.best_params_)

grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], 'solver' : ['newton-cg', 'lbfgs', 'liblinear'],}# l1 lasso l2 ridge
model = LogisticRegression()
clf = GridSearchCV(model, grid)
clf.fit(X, Y)
print(clf.best_params_)

grid = {"n_estimators": list(range(1, 300, 50)),
        "max_depth": list(range(10, 100, 5))
        }
model = RandomForestClassifier()
clf = GridSearchCV(model, grid)
clf.fit(X, Y)
a = clf.best_params_
print(a)

grid = {'criterion': ['gini', 'entropy'],
        "splitter": ["best", "random"],
        'min_samples_split': list(range(2, 20))
        }

model = DecisionTreeClassifier()
clf = GridSearchCV(model, grid)
clf.fit(X, Y)
print(clf.best_params_)"""


def vis_data_2(X, Y, X_t, Y_t, pred, pred_t, pred_param, pred_t_param):
    pred = pd.Series(pred)
    pred_t = pd.Series(pred_t)
    """fig, ((ax1, ax2, ax22), (ax3, ax4, ax44)) = plt.subplots(2, 3)
    fig.suptitle('Data')
    fig.set_figheight(10)
    fig.set_figwidth(10)"""

    # train original
    Df = pd.DataFrame(data=X
                      , columns=['Component 1', 'Component 2'])
    finalDf = pd.concat([Df, Y], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Component 1', fontsize=15)
    ax1.set_ylabel('Component 2', fontsize=15)
    ax1.set_title('train original', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax1.scatter(finalDf.loc[indicesToKeep, 'Component 1']
                    , finalDf.loc[indicesToKeep, 'Component 2']
                    , c=color
                    , s=50)
    ax1.legend(targets)
    ax1.grid()
    plt.show()

    # train pred

    Df = pd.DataFrame(data=X.reset_index(drop=True)
                      , columns=['Component 1', 'Component 2'])
    Df['target'] = pred
    fig = plt.figure(figsize=(8, 8))
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_xlabel('Component 1', fontsize=15)
    ax2.set_ylabel('Component 2', fontsize=15)
    ax2.set_title('train pred', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = Df['target'] == target
        ax2.scatter(Df.loc[indicesToKeep, 'Component 1']
                    , Df.loc[indicesToKeep, 'Component 2']
                    , c=color
                    , s=50)
    ax2.legend(targets)
    ax2.grid()
    plt.show()

    # train pred_metr

    Df = pd.DataFrame(data=X.reset_index(drop=True)
                      , columns=['Component 1', 'Component 2'])
    Df['target'] = pred_param
    fig = plt.figure(figsize=(8, 8))
    ax22 = fig.add_subplot(1, 1, 1)
    ax22.set_xlabel('Component 1', fontsize=15)
    ax22.set_ylabel('Component 2', fontsize=15)
    ax22.set_title('train pred_metr', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = Df['target'] == target
        ax22.scatter(Df.loc[indicesToKeep, 'Component 1']
                     , Df.loc[indicesToKeep, 'Component 2']
                     , c=color
                     , s=50)
    ax22.legend(targets)
    ax22.grid()
    plt.show()

    # test original

    Df = pd.DataFrame(data=X_t
                      , columns=['Component 1', 'Component 2'])
    finalDf = pd.concat([Df, Y_t], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax3 = fig.add_subplot(1, 1, 1)
    ax3.set_xlabel('Component 1', fontsize=15)
    ax3.set_ylabel('Component 2', fontsize=15)
    ax3.set_title('test original', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax3.scatter(finalDf.loc[indicesToKeep, 'Component 1']
                    , finalDf.loc[indicesToKeep, 'Component 2']
                    , c=color
                    , s=50)
    ax3.legend(targets)
    ax3.grid()
    plt.show()

    # test pred

    Df = pd.DataFrame(data=X_t.reset_index(drop=True)
                      , columns=['Component 1', 'Component 2'])
    Df['target'] = pred_t
    fig = plt.figure(figsize=(8, 8))
    ax4 = fig.add_subplot(1, 1, 1)
    ax4.set_xlabel('Component 1', fontsize=15)
    ax4.set_ylabel('Component 2', fontsize=15)
    ax4.set_title('test pred', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = Df['target'] == target
        ax4.scatter(Df.loc[indicesToKeep, 'Component 1']
                    , Df.loc[indicesToKeep, 'Component 2']
                    , c=color
                    , s=50)
    ax4.legend(targets)
    ax4.grid()
    plt.show()

    # test pred_metr

    Df = pd.DataFrame(data=X_t.reset_index(drop=True)
                      , columns=['Component 1', 'Component 2'])
    Df['target'] = pred_t_param
    fig = plt.figure(figsize=(8, 8))
    ax44 = fig.add_subplot(1, 1, 1)
    ax44.set_xlabel('Component 1', fontsize=15)
    ax44.set_ylabel('Component 2', fontsize=15)
    ax44.set_title('test pred_metr', fontsize=20)
    targets = [0, 1, 2]
    colors = ['pink', 'yellow', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = Df['target'] == target
        ax44.scatter(Df.loc[indicesToKeep, 'Component 1']
                     , Df.loc[indicesToKeep, 'Component 2']
                     , c=color
                     , s=50)
    ax44.legend(targets)
    ax44.grid()
    plt.show()

def train_model_2(train,test, model, model_param, metrics = ['accuracy', 'precision_weighted','recall_weighted','f1_weighted'] ):
    X = train.drop(columns=['target'])
    Y = train['target']
    X_t = test.drop(columns=['target'])
    Y_t = test['target']
    metric = my_cross_val(X,Y, model, metrics)
    metric_param = my_cross_val(X,Y, model_param, metrics)
    model.fit(X, Y)
    model_param.fit(X, Y)
    answers_t = model.predict(X_t)
    answers = model.predict(X)

    metrics_test = []
    metrics_test.append(accuracy_score(Y_t, answers_t))
    metrics_test.append(precision_score(Y_t, answers_t, average='weighted'))
    metrics_test.append(recall_score(Y_t, answers_t, average='weighted'))
    metrics_test.append(f1_score(Y_t, answers_t, average='weighted'))

    answers_t_param = model_param.predict(X_t)
    answers_param = model_param.predict(X)
    metrics_test_param = []
    metrics_test_param.append(accuracy_score(Y_t, answers_t_param))
    metrics_test_param.append(precision_score(Y_t, answers_t_param, average='weighted'))
    metrics_test_param.append(recall_score(Y_t, answers_t_param, average='weighted'))
    metrics_test_param.append(f1_score(Y_t, answers_t_param, average='weighted'))
    vis_data_2(X, Y, X_t,Y_t,answers,answers_t,answers_param, answers_t_param, )
    return model,model_param, metric, metric_param, metrics_test, metrics_test_param


model = KNeighborsClassifier()
model_param = KNeighborsClassifier(algorithm='auto',
                                   metric='minkowski',
                                   n_neighbors=4,
                                   p=1,
                                   weights='uniform')
ans = train_model_2(train, test, model, model_param)
print(ans)

model = LogisticRegression()
model_param = LogisticRegression(C=0.1, penalty='l2', 'solver': 'newton-cg')
ans = train_model_2(train, test, model, model_param)
print(ans)

model_param = RandomForestClassifier(
    max_depth: 5,
    min_samples_leaf: 3,
    min_samples_split: 6,
    n_estimators: 6
)
model = RandomForestClassifier()
ans = train_model_2(train, test, model, model_param)
print(ans)

model_param = DecisionTreeClassifier(
    criterion =  'gini',
    min_samples_split = 5,
    splitter = 'random'
)
model= DecisionTreeClassifier()
ans = train_model_2(train,test, model, model_param)
print(ans)







