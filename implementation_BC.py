import numpy as np
import pandas as pd
from random_forest_BC import RandomForest

# menghitung akurasi
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# load database
breast_cancer = pd.read_csv('data.csv')
names = ['id', 'diagnosis', 'radius_mean', 'texture_mean',
         'perimeter_mean', 'area_mean', 'smoothness_mean',
         'compactness_mean', 'concavity_mean', 'concave_points_mean',
         'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
         'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
         'compactness_se', 'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se', 'radius_worst',
         'texture_worst', 'perimeter_worst', 'area_worst',
         'smoothness_worst', 'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

dx = ['Benign', 'Malignan']
# mengubah fitur id menjadi index
breast_cancer.set_index(['id'], inplace=True)
# mengubah data diagnosis menjadi 1 dan 0
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
# mengecek data apakah terdapat kolom yang kosong
breast_cancer.apply(lambda X: X.isnull().sum())
# menghapus kolom yang kosong
names_index = names[2:]
del breast_cancer['Unnamed: 32']

print("dimension of dataframe:\n", breast_cancer.shape)
print("data type of dataframe:\n", breast_cancer.dtypes)

pd.crosstab(index=breast_cancer['diagnosis'], columns='count')

X = breast_cancer.values[:, 1:]
y = breast_cancer.values[:, 0]

# training test split data
spl = 0.8
N = len(X)
sample = int(spl*N)
idx = np.random.permutation(X.shape[0])
train_idx, test_idx = idx[:sample], idx[sample:]
X_train, X_test, y_train, y_test = X[train_idx,:], X[test_idx,:], y[train_idx,], y[test_idx,]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# mengubah tipe data menjadi integer
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

accuracies = []
for i in range(10):
    # training data dengan randomforest
    clf = RandomForest(n_tress=5, max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # menghitung akurasi
    acc = accuracy(y_test, y_pred)
    print("accuracy: ", acc)
    accuracies.append(acc)

print("akurasi rata-rata: {}".format(np.array(accuracies).mean()))
