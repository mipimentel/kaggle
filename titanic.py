import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# ML
# for transformers creation
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# models and metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
# distributions for random search
from scipy.stats import randint, expon, reciprocal

import joblib  # for saving models from skikit-learn


# some utils for saving and reading later
def save(model, cv_info, classification_report, name="model", cv_scores=None):
    _model = {
        "cv_info": cv_info, "classification_report": classification_report, "model": model, "cv_scores": cv_scores
    }
    joblib.dump(_model, "models_saved/" + name + ".pkl")


def load(name="model", verbose=True, with_metadata=False):
    _model = joblib.load("models_saved/" + name + ".pkl")
    if verbose:
        print("\nLoading model with the following info:\n")
        [print("{key}: {val}".format(key=key, val=val)) for key, val in _model["cv_info"].items()]
        print("\nClassification Report:\n")
        print(_model["classification_report"])
    if not with_metadata:
        return _model["model"]
    else:
        return _model


train = pd.read_csv("./datasets/titanic/train.csv")
test = pd.read_csv("./datasets/titanic/test.csv")

# Transformers created by https://github.com/ageron/handson-ml2

# this transformers we will choose which attributes, late numerical and categorical
# to use some input strategies
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


class AgeGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, new_attribute="AgeGrp", attribute_name="Age", group_scale=15, del_originals=True):
        self.group_scale = group_scale
        self.attribute_name = attribute_name
        self.new_attribute = new_attribute
        self.del_originals = del_originals

    def fit(self, X, y=None):
        self.age_groups = X[self.attribute_name] // self.group_scale * self.group_scale
        return self

    def transform(self, X, y=None):
        X[self.new_attribute] = self.age_groups
        if self.del_originals:
            X.drop(columns=self.attribute_name, axis=1, inplace=True)
        return X


class AtributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, new_attribute="RelativesOnboard", attribute_names=["SibSp", "Parch"], del_originals=True):
        self.attribute_names = attribute_names
        self.final_attr = 0
        self.new_attribute = new_attribute
        self.del_originals = del_originals

    def fit(self, X, y=None):
        for attr in self.attribute_names:
            self.final_attr += X[attr]
        return self

    def transform(self, X, y=None):
        X[self.new_attribute] = self.final_attr
        if self.del_originals:
            X.drop(columns=self.attribute_names, axis=1, inplace=True)
        return X


# Numerical Pipeline
num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "Fare", "SibSp", "Parch"])),
        ("age_grouper", AgeGrouper(attribute_name="Age", group_scale=15)),
        ("total_relatives", AtributesAdder(attribute_names=["SibSp", "Parch"], del_originals=True)),
        ("imputer", SimpleImputer(strategy="median")),
    ])

# Categorical Pipeline
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

X_train = preprocess_pipeline.fit_transform(train)
y_train = train["Survived"]

X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42)


models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
}

randomized_params = {
    "KNeighborsClassifier": {
        "n_neighbors": randint(low=1, high=30),
    },
    "RandomForest": {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    },
    "SVM": {
        "kernel": ["linear", "rbf"],
        "C": reciprocal(0.1, 200000),
        "gamma": expon(scale=1.0),
    }
}

scoring = "accuracy"


for model_name in models.keys():
    grid = RandomizedSearchCV(models[model_name], param_distributions=randomized_params[model_name], n_iter=100,
                              scoring=scoring, cv=5, verbose=2, random_state=42,  n_jobs=4)
    grid.fit(X_train, y_train)

    scores = cross_val_score(grid.best_estimator_, X_train_val, y_train_val, cv=10,
                             scoring=scoring, verbose=0, n_jobs=4)

    CV_scores = scores.mean()
    STDev = scores.std()
    Test_scores = grid.score(X_test_val, y_test_val)

    cv_score = {'Model_Name': model_name, 'Parameters': grid.best_params_, 'Test_Score': Test_scores,
                'CV Mean': CV_scores, 'CV STDEV': STDev,}

    clf = grid.best_estimator_.fit(X_train_val, y_train_val)
    clf.score(X_test_val, y_test_val)
    y_pred = clf.predict(X_test_val)
    clf_report = classification_report(y_test_val, y_pred)
    save(grid, cv_score, clf_report, name="titanic_"+model_name+"_02", cv_scores=scores)
