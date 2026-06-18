from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


CLASSIFIERS = {
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "gnb": GaussianNB(),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}


def get_classifier(name):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            scale_pos_weight=1,
            random_state=42,
            eval_metric="logloss",
        )
    if name in CLASSIFIERS:
        return CLASSIFIERS[name]
    raise ValueError(f"Unknown classifier: {name}")
