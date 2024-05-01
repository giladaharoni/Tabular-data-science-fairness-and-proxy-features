import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from ucimlrepo import fetch_ucirepo
from hyperopt import hp, fmin, tpe, Trials, space_eval
import lightgbm as lgb
import sklearn as sk


class DataSet:
    def __init__(self):
        self.columns = []
        self.target = 0
        self.protected = 0
        self.data = pd.DataFrame()
        self.objective = 0

    def brute_force_filtering(self):
        graded_combos = []
        combinations = list(itertools.combinations(self.columns, 3))
        for combo in combinations:
            selected_columns = [col for col in self.data.columns if any(col.startswith(prefix + "_") or prefix == col
                                                                        for prefix in combo)]
            potential_proxy = self.data[selected_columns].copy()
            _, score = prediction(potential_proxy, self.protected)
            graded_combos.append((score, combo))
        best_combo = max(graded_combos, key=lambda x: x[0])
        return [col for col in self.data.columns if any(col.startswith(prefix + "_") or prefix == col
                                                        for prefix in best_combo[1])]


def preprocess_data(data):
    """
    variable with 2 values become binary, categorical is turning to one hot encoding
    :param data:
    :return:
    """
    for column in data.columns:
        # binary encoding
        if len(data[column].unique()) == 2:
            data[column] = data[column].map(lambda x: x == data[column].unique()[0]).astype(int)
        # one hot encoding for categorical variables
        elif data[column].dtype == 'object':
            one_hot_encoded = pd.get_dummies(data[column], prefix=column)
            data = pd.concat([data.drop(column, axis=1), one_hot_encoded], axis=1)
    return data


def binarization(column):
    """
    turn contious variable to binary by splitting with its median
    :param column: dataseries single feature
    :return: binary version
    """
    return column.map(lambda x: x > column.median()).astype(int)


class StudentsDS(DataSet):
    def __init__(self):
        super().__init__()
        student_performance = fetch_ucirepo(id=320)
        self.data = student_performance.data.features
        self.protected = self.data['sex'].map(lambda x: x != 'F').astype(int)
        self.data = self.data.drop(labels=['sex'], axis=1)
        self.target = binarization(student_performance.data.targets['G3'])
        self.columns = self.data.columns
        self.data = preprocess_data(self.data)


class AdultsDS(DataSet):
    def __init__(self):
        super().__init__()
        adult = fetch_ucirepo(id=2)
        self.data = adult.data.features
        self.target = adult.data.targets['income'].map(lambda x: x == "<=50K" or x == "<50K").astype(int)
        self.protected = self.data['race'].map(lambda x: x == 'White').astype(int)
        self.data = self.data.drop(labels=['race'], axis=1)
        self.columns = self.data.columns
        self.data = preprocess_data(self.data)


class GermanCredit(DataSet):
    def __init__(self):
        super().__init__()
        statlog_german_credit_data = fetch_ucirepo(id=144)
        X = statlog_german_credit_data.data.features
        y = statlog_german_credit_data.data.targets['class']
        protected_feature = 'Attribute9'
        self.protected = X[protected_feature].map(lambda x: x == 'A92' or x == 'A95').astype(int)
        self.data = X.drop(labels=[protected_feature], axis=1)
        self.target = y.map(lambda x: x == 1).astype(int)
        self.columns = self.data.columns
        self.data = preprocess_data(self.data)


class Compass(DataSet):
    def __init__(self):
        super().__init__()
        df = pd.read_csv('compas.csv')
        labels_to_keep = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                          'days_b_screening_arrest', 'sex', 'race', 'two_year_recid']
        df = df[labels_to_keep]
        df = df.dropna(thresh=int(len(df) * 0.9), axis=1).dropna()
        protected_feature = 'race'
        target_feature = 'two_year_recid'
        self.target = df[target_feature]
        self.protected = df[protected_feature].map(lambda x: x == 'African-American').astype(int)
        df = df.drop(labels=[target_feature, protected_feature], axis=1)
        self.columns = df.columns
        self.data = preprocess_data(df)


def check_parameters(params):
    params['num_leaves'] = max(int(params['num_leaves']), 2)
    params['max_depth'] = max(int(params['max_depth']), 2)
    params['n_estimators'] = max(int(params['n_estimators']), 1)
    return params


def prediction(X, y):
    """
    given data set, init a lfbm classsifier, finding the sutiable model by auroc.
    :param X: data points
    :param y: target feature
    :return: best model and score (Auroc)
    """
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.3)
    param_space = {
        'num_leaves': hp.randint('num_leaves', 130) + 20,
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
        'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    }

    def objective(params):
        params = check_parameters(params)
        model = lgb.LGBMClassifier(**params, verbose=-1)
        kf = sk.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
        auroc_scores = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict_proba(X_val_fold)
            for i in range(len(model.classes_)):
                class_label = model.classes_[i]

                # Extract the predicted probabilities for the current class label
                y_pred_proba_class = y_pred[:, i]

                # Compute the ROC AUC score for the current class label
                auroc_score = sk.metrics.roc_auc_score((y_val_fold == class_label).astype(int), y_pred_proba_class)

                # Append the ROC AUC score to the list
                auroc_scores.append(auroc_score)

        # Return the negative mean AUROC (as Hyperopt minimizes the objective)
        return -np.mean(auroc_scores)

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )
    best_params = check_parameters(best_params)
    best_model = lgb.LGBMClassifier(**best_params, verbose=-1)
    best_model.fit(X_train, y_train)

    predicts = best_model.predict_proba(X_test)
    auroc_scores = []
    for i in range(len(best_model.classes_)):
        class_label = best_model.classes_[i]

        # Extract the predicted probabilities for the current class label
        y_pred_proba_class = predicts[:, i]

        # Compute the ROC AUC score for the current class label
        auroc_score = sk.metrics.roc_auc_score((y_test == class_label).astype(int), y_pred_proba_class)

        # Append the ROC AUC score to the list
        auroc_scores.append(auroc_score)
    return best_model, np.mean(auroc_scores)


def one_columns_eval(y_pred, protected):
    numerator = ((y_pred == 1) & (protected == 1)).sum() / (protected == 1).sum()
    denominator = ((y_pred == 1) & (protected == 0)).sum() / (protected == 0).sum()
    return numerator / denominator


def evaluate_fairness(model, dataset: DataSet):
    y_pred = model.predict(dataset.data)
    return one_columns_eval(y_pred, dataset.protected)


def select_best_features(estimator, X, y, n_features=3):
    sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features)
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    return selected_features


def entire_process(datasetfunc):
    dataset = datasetfunc()
    model, score = prediction(dataset.data, dataset.target)
    init_fairness = evaluate_fairness(model, dataset)
    print(f"the first fairness balance is {init_fairness}, the first score is {score}")
    proxy_features = select_best_features(model, dataset.data, dataset.protected)
    print(proxy_features)
    reduced_dataset = dataset
    reduced_dataset.data = dataset.data.drop(labels=list(proxy_features), axis=1)
    fair_model, fair_score = prediction(reduced_dataset.data, dataset.target)
    fixed_fair = evaluate_fairness(fair_model, reduced_dataset)
    print(f"the last fairness balance is {fixed_fair}, the fixed auroc is now {fair_score}")
    print(f"the fairness was increased by {((init_fairness - fixed_fair) / init_fairness) * 100}%")

    proxy_features = dataset.brute_force_filtering()
    reduced_dataset = dataset
    reduced_dataset.data = dataset.data.drop(labels=list(proxy_features), axis=1)
    fair_model, fair_score = prediction(reduced_dataset.data, dataset.target)
    fixed_fair = evaluate_fairness(fair_model, reduced_dataset)
    print(proxy_features)
    print(f"the last fairness balance is {fixed_fair}, the fixed auroc is now {fair_score}")
    print(f"the fairness was increased by {((init_fairness - fixed_fair) / init_fairness) * 100}%")


if __name__ == '__main__':
    dataset_dict = {"students": StudentsDS, "compass": Compass, "adults": AdultsDS, "german": GermanCredit}
    try:
        if sys.argv[1] in dataset_dict.keys():
            entire_process(dataset_dict[sys.argv[1]])
        else:
            print("Please choose a valid sample dataset")
    except IndexError:
        print("Please choose a valid sample dataset")

