from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)


def exp_1_split(df, test_fraction):

    # Get one label per raw file
    # (all windows from a file have the same label)
    file_info = (
        df[["file_id", "Health label"]]
        .drop_duplicates()
    )

    # Split file IDs while preserving class distribution
    train_files, test_files = train_test_split(
        file_info,
        test_size=test_fraction,
        stratify=file_info["Health label"],
        random_state=42
    )

    # Extract file IDs
    train_ids = train_files["file_id"]
    test_ids = test_files["file_id"]

    # Select all windows belonging to those files
    train_df = df[df["file_id"].isin(train_ids)].copy()
    test_df = df[df["file_id"].isin(test_ids)].copy()


    # Check that no files overlap
    assert len(set(train_df["file_id"]) & set(test_df["file_id"])) == 0


    # Separate features and labels
    X_train = train_df.drop(
        columns=[
            "file_id",
            "Health label",
            "Operating frequency"
        ]
    )

    y_train = train_df["Health label"]


    X_test = test_df.drop(
        columns=[
            "file_id",
            "Health label",
            "Operating frequency"
        ]
    )

    y_test = test_df["Health label"]


    # Scale using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)


    return X_train_scaled, X_test_scaled, y_train, y_test







def exp_2_split(df, test_frequency):

    # Split by operating frequency
    train_df = df[df["Operating frequency"] != test_frequency].copy()
    test_df = df[df["Operating frequency"] == test_frequency].copy()

    # Separate features and labels
    X_train = train_df.drop(
        columns=["file_id", "Health label", "Operating frequency"]
    )
    y_train = train_df["Health label"]

    X_test = test_df.drop(
        columns=["file_id", "Health label", "Operating frequency"]
    )
    y_test = test_df["Health label"]

    # Scale using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test



def run_exp_1(df):

    results = []
    confusion_matrices = {}

    X_train, X_test, y_train, y_test = exp_1_split(
                                        df=df,
                                        test_fraction=0.25
                                        )


    #majority class predictor (baseline)
    majority_class_metrics = {}

    majority_class_metrics["Classifier type"] = "Majority class predictor"

    model = DummyClassifier(strategy="most_frequent")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    macro_precision = metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = metrics.recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)
    per_class_recall = metrics.recall_score(y_test, y_pred, average=None, zero_division=0)
    cm = metrics.confusion_matrix(y_test, y_pred)

    #store to metrics dataframe

    majority_class_metrics["Accuracy"] = acc
    majority_class_metrics["Macro precision"] = macro_precision
    majority_class_metrics["Macro recall"] = macro_recall
    majority_class_metrics["Macro F1"] = macro_f1
    majority_class_metrics["Per class recall"] = per_class_recall

    results.append(majority_class_metrics)
    confusion_matrices["Majority class predictor"] = cm


    #Support vector machine

    svm_metrics = {}

    svm_metrics["Classifier type"] = "Support vector machine"

    model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
            )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    macro_precision = metrics.precision_score(y_test, y_pred, average='macro')
    macro_recall = metrics.recall_score(y_test, y_pred, average='macro')
    macro_f1 = metrics.f1_score(y_test, y_pred, average='macro')
    per_class_recall = metrics.recall_score(y_test, y_pred, average=None)
    cm = metrics.confusion_matrix(y_test, y_pred)

    #store to metrics dataframe

    svm_metrics["Accuracy"] = acc
    svm_metrics["Macro precision"] = macro_precision
    svm_metrics["Macro recall"] = macro_recall
    svm_metrics["Macro F1"] = macro_f1
    svm_metrics["Per class recall"] = per_class_recall

    results.append(svm_metrics)
    confusion_matrices["Support vector machine"] = cm



    #Random forest

    rf_metrics = {}

    rf_metrics["Classifier type"] = "Random forest"

    model = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=10,
                                random_state=42
                                )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    macro_precision = metrics.precision_score(y_test, y_pred, average='macro')
    macro_recall = metrics.recall_score(y_test, y_pred, average='macro')
    macro_f1 = metrics.f1_score(y_test, y_pred, average='macro')
    per_class_recall = metrics.recall_score(y_test, y_pred, average=None)
    cm = metrics.confusion_matrix(y_test, y_pred)

    #store to metrics dataframe

    rf_metrics["Accuracy"] = acc
    rf_metrics["Macro precision"] = macro_precision
    rf_metrics["Macro recall"] = macro_recall
    rf_metrics["Macro F1"] = macro_f1
    rf_metrics["Per class recall"] = per_class_recall

    results.append(rf_metrics)
    confusion_matrices["Random forest"] = cm

    df = pd.DataFrame(results)
    df.to_csv("results/exp1_results.csv", index=False)

    return(df, confusion_matrices)

