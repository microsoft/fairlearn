from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from fairlearn.postprocessing import _relabeling
from fairlearn.datasets import fetch_adult

round_value = 10

# Load the dataset
dataset = fetch_adult(as_frame=True)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['>50k'] = (dataset.target == '>50K') * 1

# Preprocessing
print("Preprocessing:\n--------------")
le = preprocessing.LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
onehot = ['workclass', 'education', 'marital-status', 'occupation', 'marital-status',
          'occupation', 'relationship', 'race']
df = pd.get_dummies(df, prefix=onehot, columns=onehot, drop_first=True)

# Reverse the labels to have positive discrimination
tmp = df['sex'].to_list()
for i in range(0, len(tmp)):
    if tmp[i] == 1:
        tmp[i] = 0
    else:
        tmp[i] = 1
df['sex'] = tmp
sensitive = df['sex']
y = df[">50k"]
X = df.loc[:, ~df.columns.isin(['sex', '>50k', 'native-country'])]
print(
    f"Discrimination in the dataset:"
    f" {_relabeling.discrimination_dataset(y, sensitive)}")

print("\nSummary:\n--------")
y = y.to_numpy()
sensitive = sensitive.to_numpy()
X = X.to_numpy()

clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, list(y))
y_pred = clf.predict(X)
accuracy = round(accuracy_score(y, y_pred), round_value)
print("Accuracy of the prediction before relabeling: ", accuracy)
discrimination = round(_relabeling.discrimination_dataset(y_pred, sensitive),
                       round_value)
print("Discrimination of classifier on the prediction before relabeling: ",
      discrimination)

# Relabeling the tree to reduce discrimination
threshold = 0.1
print(
    f"Let's find the leaves to relabel to have a discrimination lower than:"
    f" {threshold}")

_relabeling.relabeling(clf, X, y, y_pred, sensitive, threshold)
y_pred_relabel = clf.predict(X)
accuracy_relabel = round(accuracy_score(y, y_pred_relabel), round_value)
print("\nAccuracy of the prediction after relabeling: ", accuracy_relabel)
discrimination_relabel = round(
    _relabeling.discrimination_dataset(y_pred_relabel, sensitive), round_value)
print("Discrimination of classifier on the prediction after relabeling: ",
      discrimination_relabel)

# Detailed operations
print("\nDetailed steps (The leaves to be relabeled and their impact):"
      "\n-------------------------------------------------------------")
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(X, list(y))
y_pred = clf.predict(X)
accuracy = round(accuracy_score(y, y_pred), round_value)
print("Accuracy of the prediction before relabeling: ", accuracy)
discrimination = round(_relabeling.discrimination_dataset(y_pred, sensitive),
                       round_value)
print("Discrimination of classifier on the prediction before relabeling: ",
      discrimination)

threshold = 0  # The threshold of discrimination that we do not want to exceed
print(
    f"Let's find the leaves to relabel to have a discrimination lower than:"
    f" {threshold}\n")

# If you want to know which leaves will be relabeled,
# you can call "leaves_to_relabel(...)"
# which returns a set of "Leaf()" objects

leaves_relabel = _relabeling.leaves_to_relabel(clf, X, y, y_pred, sensitive, threshold)

sum_acc = 0
sum_disc = 0
for leaf in leaves_relabel:
    print(leaf)
    print()
    sum_acc += leaf.acc
    sum_disc += leaf.disc
sum_acc = round(sum_acc, round_value)  # The effect of relabeling the leaves on accuracy
sum_disc = round(sum_disc,
                 round_value)  # The effect of relabeling the leaves on discrimination

# The values are accurate to a certain decimal.
_relabeling.relabeling(clf, X, y, y_pred, sensitive, threshold)
y_pred_relabel = clf.predict(X)
accuracy_relabel = round(accuracy_score(y, y_pred_relabel), round_value)
discrimination_relabel = round(
    _relabeling.discrimination_dataset(y_pred_relabel, sensitive), round_value)
new_acc = round(accuracy + sum_acc, round_value)
new_disc = round(discrimination + sum_disc, round_value)

print("\nResult:\n-------")
print(f"Accuracy:\n"
      f"    Before      : {accuracy}\n"
      f"    After       : {accuracy_relabel}\n"
      f"    Expected    : {new_acc}\n"
      f"    Leafs       : {sum_acc}\n"
      f"    Difference  : {round(-accuracy + accuracy_relabel, round_value)}\n"
      f"    Check       : {abs(accuracy_relabel - new_acc) <= 0.000000001}")
print(f"Discrimination:\n"
      f"    Before      : {discrimination}\n"
      f"    After       : {discrimination_relabel}\n"
      f"    Expected    : {new_disc}\n"
      f"    Leafs       : {sum_disc}\n"
      f"    Difference  : {round(-discrimination + discrimination_relabel,
                                 round_value)}\n"
      f"    Check       : {abs(discrimination_relabel - new_disc) <= 0.000000001}")
