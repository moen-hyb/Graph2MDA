
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import tree
import xgboost as xgb
from xgboost import XGBClassifier

def train_RF(X_train, Y_train, X_test, Y_test):

    print("==========================================")
    print("RandomForest Classifier")
    RF = RandomForestClassifier(n_estimators=100, random_state=11)
    RF.fit(X_train, Y_train)
    predictions = RF.predict(X_test)
    return predictions


def train_GBDT(X_train, Y_train, X_test, Y_test):
    print("==========================================")
    print("GBDT Classifier")
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions


def train_AdaBoost(X_train, Y_train, X_test, Y_test):
    print("==========================================")
    print("AdaBoost Classifier")
    clf = AdaBoostClassifier()
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions

def train_SVM(X_train, Y_train, X_test, Y_test):
    print("==========================================")
    print("SVM Classifier ")
    clf = SVC()
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions


def train_xgboost(X_train, Y_train, X_test, Y_test):
    print("==========================================")
    print("xgboost Classifier ")
    clf = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, eval_metric='rmse')
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions


###############################
def train_LR(X_train, Y_train, X_test, Y_test):
    print("==========================================")
    print("Logistic Regression Classifier")
    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions



def train_DT(X_train, Y_train, X_test, Y_test):
    print("==========================================")
    print("Decision Tree Classifier")
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions

