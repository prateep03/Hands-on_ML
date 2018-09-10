import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X = StandardScaler().fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Voting classifier
"""
def voting_classifier(X_train, X_val, y_train, y_val):
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    log_clf = LogisticRegression()
    rf_clf = RandomForestClassifier()
    '''
    By default, SVC doesn't have predict_proba method. So we need to set `probability` hyper-parameter to True.
    This will make SVC class use cross-validation to estimate class probabilities, slowing down training, 
    and it will add a predict_proba() method. 
    '''
    svm_clf = SVC(probability=True)

    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rf_clf), ('svc', svm_clf)], voting='soft')
    # voting_clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    for clf in (log_clf, rf_clf, svm_clf, voting_clf):
        clf.fit(X_train,y_train)
        y_prob = clf.predict_proba(X_val)
        y_pred = np.argmax(y_prob, axis=1)
        print(clf.__class__.__name__, accuracy_score(y_val, y_pred))

"""
Bagging and pasting in scikit-learn
- following code trains 500 DT classifiers, each trained on 100 training instances randomly sampled from the 
   training set with replacement(this is an example of bagging, for pasting, just set `bootstrap = False`
- Out-of-bagging(oob): With bagging, some instances may be sampled several times for any given predictor, while
   others may not be sampled at all. By default BaggingClassifier samples m training instances with replacement
   (bootstrap = True), where m is the size of the training set. Since a predictor never sees the oob instances during 
   training, it can be evaluated on these instances, without the need for a separate validation set or cv.
   In scikit-learn, set `oob_score=True` when creating BaggingClassifier to request an automatic oob evaluation after
   training.
"""

def bagging_pasting(X_train, X_val, y_train, y_val, doplot=False):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_val)
    print("DecisionTree classifier, accuracy score = %f\n" % accuracy_score(y_val, y_pred_tree))

    bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=500,
                                max_samples=100, bootstrap=True, n_jobs=1, oob_score=True)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_val)
    print("Bagging classifier, accuracy score = %f\n" % accuracy_score(y_val, y_pred))

    if doplot:
        from utils import plot_decision_boundary, save_fig
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        plot_decision_boundary(tree_clf, X, y)
        plt.title("Decision Tree", fontsize=14)
        plt.subplot(122)
        plot_decision_boundary(bag_clf, X, y)
        plt.title("Decision Trees with Bagging", fontsize=14)
        save_fig("DT_without_and_with_bagging_plot", "ensembles")
        plt.show()


bagging_pasting(X_train, X_val, y_train, y_val, doplot=True)

"""
Adaboost:
  - Weighted error rate of the j^{th} predictor
        
        r_j = \frac{\sum_{i=1}^{m}, \hat{y_j}^{(i)} != y^{(i)} w^{(i)}}{\sum_{i=1}^{m} w^{(i)}},
         
     where \hat{y_j}^{(i)} is the j^{th} predictor's prediction for ith data, m is number of samples.
  - Predictor weight
    
        \alpha_j = \eta log \frac{1 - r_j}{r_j}    
  - Weight update rule
   
        for i = 1,2 .. m
                 | w_i, if \hat{y_j}^{(i)} = y^{(i)}
           w_i = |
                 | w_i exp(\alpha_j), otherwise
                 
  - Finally, predictions 
   
        \hat{y}(x) = argmax_k \sum_{i=1}^{N}, \hat{y_j}^{(i)} = k \alpha_j 
"""

def adaboost(X_train, X_val, y_train, y_val):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), # weak classifier
                                 n_estimators=200, algorithm='SAMME.R', learning_rate=0.5)
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_val)
    print("Adaboost classifier, accuracy score = %f\n" % accuracy_score(y_val, y_pred))

# adaboost(X_train, X_val, y_train, y_val)

