from load_data import load_heart_data

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Read data
X, y = load_heart_data('heart.csv')

# Standardize data
X = preprocessing.scale(X)

#define common test parameters for learnings
cv=5  #cross validation
rs=25 # random state
#set values for selective testing
test_DT = True
test_NN = True
test_SVM = True
test_KNN = True
test_Boosting = True 
cm_plot = True
times_plot = True


# Split into training and test data. Use random_state to get the same results in every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

"""
Decision tree - test decision for learning curve, model complexity, hyper parm tuning
"""
# Learning Curve, sample size, fit times 
if test_DT:   
    #print('learning curve processing') 
    clf_dt = tree.DecisionTreeClassifier(random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    DT_train_mean = train_scores_mean
    DT_test_mean = test_scores_mean
    DT_fit_mean =  fit_times_mean 

# plot learning curve
    #print('plot learning curve sample vs score ') 
    plt.figure()
    plt.title('Learning curve for decision tree')
    plt.xlabel('Training Examples')
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",  label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_dt_learning_curve.png')

# Plot n_samples vs fit_times
    #print('DT sample and fit times') 
    plt.figure()
    plt.title('Scalibility of decision tree')
    plt.xlabel('Training Samples')
    plt.ylabel("Fit Times")
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,fit_times_mean + fit_times_std, alpha=0.1)
    plt.grid()
    plt.savefig('t1_dt_fit_scability.png')
    
    # Plot fit_time vs score
    plt.figure()
    plt.title('Performance of decision tree')
    plt.xlabel('Fit Times')
    plt.ylabel("Score")
    plt.plot(fit_times_mean, test_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    plt.grid()
    plt.savefig('t1_dt_fit_performance.png') 
    
if test_DT:
    clf_dt = tree.DecisionTreeClassifier(random_state=rs)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred)
    print('Decision Tree Accuracy - No hyperparameter tuning is %.2f%%' % (dt_accuracy*100))

# Model Complexity 
if test_DT:
    #print('DT model complexity analysis - Max Depth')
    x_size = np.size(X,1)
    #print('array col', x_size)
    param_range_1 = np.arange(1, np.size(X,1))
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=rs), X_train, y_train,
                                                 param_name="max_depth", param_range=param_range_1, cv=cv)

    plt.figure()
    plt.plot(param_range_1, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(param_range_1, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Decision Tree Validation curve - Max Depth')
    plt.xlabel('max_depth')
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_dt_model_complexity_depth.png')

    #print('DT model complexity analysis - Min samples split')
    param_range_2 = np.arange(2, x_size)
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=rs), X_train, y_train,
                                                 param_name="min_samples_split", param_range=param_range_2, cv=cv)

    plt.figure()
    plt.plot(param_range_2, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(param_range_2, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Decision Tree Validation curve - Min Sample Split')
    plt.xlabel('min_samples_split')
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_dt_model_complexity_leaf.png')

# Hyperparameter tuning
if test_DT:
    #print('DT hyper parameter tuning') 
    tuned_params = {'max_depth' : param_range_1, 'min_samples_split' : param_range_2}
    clf_dt = GridSearchCV(tree.DecisionTreeClassifier(random_state=rs), param_grid=tuned_params, cv=cv)

    start_time = time.time()  
    clf_dt.fit(X_train, y_train)
    end_time = time.time()
    dt_fit_time = end_time - start_time 
    print(' Decision Tree fit time', dt_fit_time)
    
    print("Best parameters set found on development set:")
    print(clf_dt.best_params_)
    print("DT Hyperparameter tuning, best score is :")
    print(clf_dt.best_score_)

    start_time = time.time()
    y_pred = clf_dt.predict(X_test)
    end_time = time.time()
    dt_query_time = end_time - start_time
    print(' Decision Tree query time', dt_query_time) 

    dt_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of decision tree is %.2f%%' % (dt_accuracy * 100))
    cm_dt = confusion_matrix(y_test, y_pred)
    #print('confusion matrix', cm_dt)

    #print('run new test with best parms')
    dt_best = tree.DecisionTreeClassifier(random_state=rs)
    dt_best.set_params(**clf_dt.best_params_)
    dt_best.fit(X_train, y_train)
    #print('get parms', dt_best.get_params())
    y_pred = dt_best.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of decision tree is %.2f%%' % (dt_accuracy * 100))
    cm_dt = confusion_matrix(y_test, y_pred)
    #print('confusion matrix', cm_dt) 

#decision tree capture test, training accurcy and fit times for best parms 
if test_DT:   
    #print('DT Learning curve with best parms')
    clf_dt = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    DT_train_mean = train_scores_mean
    DT_test_mean = test_scores_mean
    DT_fit_mean =  fit_times_mean 

  
"""
Neural network - test decision for learning curve, model complexity, hyper parm tuning
"""
if test_NN:   
    #print('Neural Network learning curve processing') 
    clf_dt = MLPClassifier(random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) 
    NN_train_mean = train_scores_mean
    NN_test_mean = test_scores_mean
    NN_fit_mean =  fit_times_mean 

# plot learning curve
    #print('plot NN learning curve sample vs score ') 
    plt.figure()
    plt.title('Learning curve for Neural Network')
    plt.xlabel('Training Examples')
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",  label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_NN_learning_curve.png')

if test_NN: 
    #print('neural network starting')
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=rs, early_stopping=True, validation_fraction=0.2)
    clf_nn.fit(X_train, y_train)
    y_pred = clf_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of neural network is %.2f%%' % (nn_accuracy * 100))

# Learing Curve 2 
if test_NN:
    #print('plot NN learning curve Epochs ')
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=rs, max_iter=1, warm_start=True)
    # Split validation set
    X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=rs)
    num_epochs = 1500
    #train_loss = train_scores = val_scores = np.empty(num_epochs)
    train_loss = np.empty(num_epochs)
    train_scores = np.empty(num_epochs)
    val_scores = np.empty(num_epochs)
    for i in range(num_epochs):
        clf_nn.fit(X_train1, y_train1)
        train_loss[i] = clf_nn.loss_
        train_scores[i] = accuracy_score(y_train1, clf_nn.predict(X_train1))
        val_scores[i] = accuracy_score(y_val, clf_nn.predict(X_val))

    y_pred = clf_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of neural network is %.2f%%' % (nn_accuracy * 100))

    xrange = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(xrange, train_loss)
    plt.title('Training loss curve for neural network')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig('t1_nn_learing_epochs_loss.png')
    
    plt.figure()
    plt.plot(xrange, train_scores, label='Training score')
    plt.plot(xrange, val_scores, label='Validation score')
    plt.title('Training and validation score curve for neural network')
    plt.xlabel('Epochs')
    plt.ylabel("Score")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig('t1_nn_score_curve.png')
    
  
    # model complexity 
if test_NN:
    f1_test = []
    f1_train = []
    hlist = np.linspace(1,50,20).astype('int')
    for i in hlist:         
            clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
                                learning_rate_init=0.01, random_state=rs)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
           
    plt.figure()  
    plt.plot(hlist, f1_test, 'o-', color='g', label='Test F1 Score')
    plt.plot(hlist, f1_train, 'o-', color = 'r', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Hidden Units') 
    plt.title('Neural Network Hident Layer-F1 score') 
    plt.legend(loc='best')
    plt.savefig('t1_nn_model_complexity.png')

# hyper parm tuning 1     
if test_NN:
    #parameters to search:
    #number of hidden units
    #learning_rate
    #print('Neural Network hyper parm tuning 1')
    h_units = [5, 10, 20, 30, 40, 50, 75, 100]
    learning_rates = [0.001, 0.01, 0.05, .1]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    param_grid = dict(solver=solver, activation=activation, hidden_layer_sizes=h_units, learning_rate_init=learning_rates)
    

    nnmodel = GridSearchCV(estimator = MLPClassifier(random_state=rs),
                       param_grid=param_grid, cv=cv)
    
    start_time = time.time() 
    nnmodel.fit(X_train, y_train) 
    end_time = time.time()
    nn_fit_time = end_time - start_time 
    print(' NN fit time', nn_fit_time)

    print("NN Hyperparameter tuning, best parameters are:")
    print(nnmodel.best_params_)
    print("NN Hyperparameter tuning, best score is :")
    print(nnmodel.best_score_)

    start_time = time.time()
    y_pred = nnmodel.predict(X_test)
    end_time = time.time()
    nn_query_time = end_time - start_time
    print(' NN query time', nn_query_time)
    
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of NN is %.2f%%' % (nn_accuracy * 100))
    cm_nn = confusion_matrix(y_test, y_pred)
    #print('confusion matrix', cm_nn)

#Neural Network capture test, training accurcy and fit times for best parms    
if test_NN:   
    #print('Neural Network best parms learning curve processing') 
    clf_dt = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=5, learning_rate_init=0.01,random_state=rs)

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) 
    NN_train_mean = train_scores_mean
    NN_test_mean = test_scores_mean
    NN_fit_mean =  fit_times_mean 

"""
SVM - test decision for learning curve, model complexity, hyper parm tuning
"""
# Learning Curve, sample size, fit times 
if test_SVM:   
    #print(' SVM learning curve processing') 
    clf_dt = svm.SVC(kernel='linear',random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    SVM_train_mean = train_scores_mean
    SVM_test_mean = test_scores_mean
    SVM_fit_mean =  fit_times_mean  

# plot learning curve
    #print('plot SVM learning curve sample vs score ') 
    plt.figure()
    plt.title('Learning curve for SVM')
    plt.xlabel('Training Examples')
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",  label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_SVM_learning_curve.png')

if test_SVM:
    svm_linear = svm.SVC(kernel='linear', random_state=rs)
    svm_linear.fit(X_train, y_train)
    y_pred = svm_linear.predict(X_test)
    svm_linear_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM with linear kernel is %.2f%%' % (svm_linear_accuracy * 100))

    svm_poly = svm.SVC(kernel='poly', degree =2, random_state=rs)
    svm_poly.fit(X_train, y_train)
    y_pred = svm_poly.predict(X_test)
    svm_poly_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM with polynomial kernel is %.2f%%' % (svm_poly_accuracy * 100))

    svm_rbf = svm.SVC(kernel='rbf', random_state=rs)
    svm_rbf.fit(X_train, y_train)
    y_pred = svm_rbf.predict(X_test)
    svm_rbf_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM with RBF kernel is %.2f%%' % (svm_rbf_accuracy * 100))
    #print(classification_report(y_test, y_pred))

if test_SVM:
    #print('SVM model analysis - RBF')
    f1_test = []
    f1_train = [] 
       
    for gv in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
        clf = svm.SVC(kernel='rbf', gamma=gv,random_state=rs)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))            
                
    plt.figure()
    xvals = ['1e-4', '1e-3', '1e-2', '0.1', '1', '10', '100']
    plt.plot(xvals, f1_test, 'o-', color='g', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'r', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Gamma')
    plt.title('SVM - RBF Gamma values F1 score')
    plt.legend(loc='best')
   
    plt.savefig('t1_svm_model_complexity_rbf.png')
    

if test_SVM:
    #print('SVM Model Compleixity')
    f1_test = []
    f1_train = []    
    for order in [2,3,4,5,6,7,8]:
        clf = svm.SVC(kernel='poly', degree=order,random_state=100)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))            
                
    plt.figure()
    xvals = ['poly2','poly3','poly4','poly5','poly6','poly7','poly8']
    plt.plot(xvals, f1_test, 'o-', color='g', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'r', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Polynomial Order')
    plt.title('SVM - Polynomial Degrees F1 score')
    plt.legend(loc='best')
    
    plt.savefig('t1_SVM_Model_complexity_curve.png')

#hyper parm tuning for SVM
if test_SVM:
    #parameters to search:
    #number of hidden units
    #learning_rate
    #print('SVM hyper parm tuning 1')
    param_grid = {'C': [0.1, 1, 10, 20, 30, 32, 35, 40, 50, 75, 100],  
              'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100], 
              'kernel': ['rbf']} 
    
    svmmodel = GridSearchCV(estimator = svm.SVC(random_state=rs),
                       param_grid=param_grid, cv=10)

    start_time = time.time() 
    svmmodel.fit(X_train, y_train)
    end_time = time.time()
    svm_fit_time = end_time - start_time 
    print(' SVM fit time', svm_fit_time)
    

    print("Per Hyperparameter tuning, best parameters are:")
    print(svmmodel.best_params_)
    print("Per Hyperparameter tuning, best score is :")
    print(svmmodel.best_score_)

    
    start_time = time.time()
    y_pred = svmmodel.predict(X_test)
    end_time = time.time()
    svm_query_time = end_time - start_time
    print(' SVM query time', svm_query_time)
    
    svm_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM is %.2f%%' % (svm_accuracy * 100))
    cm_svm = confusion_matrix(y_test, y_pred)
    #print('confusion matrix', cm_svm)

#SVM Tree capture test, training accurcy and fit times for best parms
if test_SVM:   
    #print(' SVM best parm learning curve processing') 
    clf_dt = svm.SVC(C=30, gamma=0.001,kernel='rbf',random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    SVM_train_mean = train_scores_mean
    SVM_test_mean = test_scores_mean
    SVM_fit_mean =  fit_times_mean  
  
"""
kNN - test decision for learning curve, model complexity, hyper parm tuning
"""
if test_KNN:   
    #print(' KNN learning curve processing') 
    clf_dt = KNeighborsClassifier()
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) 
    KNN_train_mean = train_scores_mean
    KNN_test_mean = test_scores_mean
    KNN_fit_mean =  fit_times_mean 


# plot learning curve
    #print('plot KNN learning curve sample vs score ') 
    plt.figure()
    plt.title('Learning curve for KNN')
    plt.xlabel('Training Examples')
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",  label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_KNN_learning_curve.png')

#KNN Model complexity accuracy 
if test_KNN:
    for k in range(1, 11):
        clf_knn = KNeighborsClassifier(n_neighbors=k)
        clf_knn.fit(X_train, y_train)
        y_pred = clf_knn.predict(X_test)
        clf_knn_accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy of kNN with k = %d is %.2f%%' % (k, clf_knn_accuracy * 100))

    f1_test = []
    f1_train = []
    klist = range(1,16,1)
    for i in klist:
        clf = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
        clf.fit(X_train,y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
    
    plt.figure()    
    plt.plot(klist, f1_test, 'o-', color='g', label='Test F1 Score')
    plt.plot(klist, f1_train, 'o-', color = 'r', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Neighbors')
    plt.title('KNN Model # of Neigbors f1 score')
    plt.legend(loc='best')
    plt.savefig('t1_KNN_Model_complex_curve.png')
    

#hyper parm tuning for KNN    
if test_KNN:
    #parameters to search:
    #number of hidden units
    #learning_rate
    #print('KNN hyper parm tuning 1')
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11 ,13, 15],  
              'weights': ['uniform', 'distance'], 
              'metric': ['euclidean','manhattan']} 
    
    knnmodel = GridSearchCV(estimator = KNeighborsClassifier(),
                       param_grid=param_grid, cv=10)
     
    start_time = time.time() 
    knnmodel.fit(X_train, y_train)
    end_time = time.time()
    knn_fit_time = end_time - start_time 
    print(' KNN fit time', knn_fit_time)
    
    print("Per KNN Hyperparameter tuning, best parameters are:")
    print(knnmodel.best_params_)
    print("Per KNN Hyperparameter tuning, best score is :")
    print(knnmodel.best_score_)
    
    start_time = time.time()
    y_pred = knnmodel.predict(X_test)
    end_time = time.time()
    knn_query_time = end_time - start_time
    print(' KNN query time', knn_query_time)
    
    knn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of KNN is %.2f%%' % (knn_accuracy * 100))
    cm_knn = confusion_matrix(y_test, y_pred)
    #print('confusion matrix', cm_knn)

#KNN capture test, training accurcy and fit times for best parms
if test_KNN:   
    #print(' KNN best parm learning curve processing') 
    clf_dt = KNeighborsClassifier(n_neighbors=7, metric='manhattan',weights='uniform')
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) 
    KNN_train_mean = train_scores_mean
    KNN_test_mean = test_scores_mean
    KNN_fit_mean =  fit_times_mean 

"""

Boosting - test decision for learning curve, model complexity, hyper parm tuning

"""
if test_Boosting:
    clf_boosted = AdaBoostClassifier(random_state=rs)
    clf_boosted.fit(X_train, y_train)
    y_pred = clf_boosted.predict(X_test)
    boosted_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of Adaboost is %.2f%%' % (boosted_accuracy * 100))

if test_Boosting:   
    #print(' Boosting learning curve processing') 
    clf_dt = AdaBoostClassifier(random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) 
    boosting_train_mean = train_scores_mean
    boosting_test_mean = test_scores_mean
    boosting_fit_mean =  fit_times_mean 

# plot learning curve
    #print('plot Boosting learning curve sample vs score ') 
    plt.figure()

    plt.title('Learning curve for Boosting')
    plt.xlabel('Training Examples')
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",  label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('t1_Boosting_learning_curve.png')


# gradient boosting 
if test_Boosting:
    f1_test = []
    f1_train = []
    n_estimators = np.linspace(1,150,10).astype('int')
    for i in n_estimators:         
            clf = GradientBoostingClassifier(n_estimators=i, max_depth=10, 
                                             min_samples_leaf=2, random_state=rs)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
    
    plt.figure()  
    plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(n_estimators, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Estimators')    
    plt.title('Gradient Boosting Estimators - f1 score')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('t1_Boosting_Model_complexity.png')

if test_Boosting:
#n_estimators, learning_rate, max_depth, min_samples_leaf
    leafs = np.arange(1,5)
    depth = np.arange(1,5) 
    learning_rates = [0.001, 0.01, 0.05, .1]
    param_grid = {'min_samples_leaf': leafs,
                  'max_depth': depth,
                  'n_estimators': np.linspace(10,25,3).round().astype('int'),
                  'learning_rate': learning_rates}

    boost = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid=param_grid, cv=10)
     
    start_time = time.time()  
    boost.fit(X_train, y_train)	
    end_time = time.time()
    boost_fit_time = end_time - start_time 
    print(' boosting fit time', boost_fit_time)
 	 		  		  		    	 		 	   
    print("Per Hyperparameter tuning, best boosting parameters are:")
    print(boost.best_params_)
    print("Per Hyperparameter tuning, best boostin score is :")
    print(boost.best_score_)

    start_time = time.time()
    y_pred = boost.predict(X_test)
    end_time = time.time()
    boost_query_time = end_time - start_time
    print(' boosting query time', boost_query_time)
   
    boost_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of boosting is %.2f%%' % (boost_accuracy * 100))
    cm_boost = confusion_matrix(y_test, y_pred)
    #print('confusion matrix', cm_boost)

#Boosting capture test, training accurcy and fit times for best parms
if test_Boosting:   
    #print(' Boosting with best parms learning curve processing')     
    clf_dt = GradientBoostingClassifier(min_samples_leaf=1, max_depth=1, n_estimators=55, learning_rate=0.05, random_state=rs)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
                 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) 
    boosting_train_mean = train_scores_mean
    boosting_test_mean = test_scores_mean
    boosting_fit_mean =  fit_times_mean 

#plot confusion matrix for all the models
if cm_plot:

    plt.figure(figsize=(24,12))

    plt.suptitle("Confusion Matrixes",fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

    plt.subplot(2,3,1)
    plt.title("Decision Tree Classifier Confusion Matrix")
    sns.heatmap(cm_dt,annot=True,cmap="Greens",fmt="d",cbar=False, annot_kws={"size": 24})

    plt.subplot(2,3,2)
    plt.title("Neural Network Confusion Matrix")
    sns.heatmap(cm_nn,annot=True,cmap="Greens",fmt="d",cbar=False, annot_kws={"size": 24})

    plt.subplot(2,3,3)
    plt.title("K Nearest Neighbors Confusion Matrix")
    sns.heatmap(cm_knn,annot=True,cmap="Greens",fmt="d",cbar=False, annot_kws={"size": 24})

    plt.subplot(2,3,4)
    plt.title("Support Vector Machine Confusion Matrix")
    sns.heatmap(cm_svm,annot=True,cmap="Greens",fmt="d",cbar=False, annot_kws={"size": 24})

    plt.subplot(2,3,5)
    plt.title("Boosting Confusion Matrix")
    sns.heatmap(cm_boost,annot=True,cmap="Greens",fmt="d",cbar=False, annot_kws={"size": 24})

    plt.savefig('t1_Confusion_matrix.png')

#plot best parm training, test accuracy and fit times for training samples
if times_plot:
    plt.figure()
    plt.title("Model Training accuracy")
    plt.xlabel("Training Examples")
    plt.ylabel("Model Accuracy")
    plt.plot(train_sizes, NN_train_mean, '-', color="r", label="Neural Network")
    plt.plot(train_sizes, SVM_train_mean, '-', color="b", label="SVM")
    plt.plot(train_sizes, KNN_train_mean, '-', color="g", label="kNN")
    plt.plot(train_sizes, DT_train_mean, '-', color="c", label="Decision Tree")
    plt.plot(train_sizes, boosting_train_mean, '-', color="k", label="Boosting")
    plt.legend(loc="best")
    plt.savefig('t1_training_accuracy.png')
    
    plt.figure()
    plt.title("Model Testing accuracy")
    plt.xlabel("Training Examples")
    plt.ylabel("Model Accuracy")
    plt.plot(train_sizes, NN_test_mean, '-', color="r", label="Neural Network")
    plt.plot(train_sizes, SVM_test_mean, '-', color="b", label="SVM")
    plt.plot(train_sizes, KNN_test_mean, '-', color="g", label="kNN")
    plt.plot(train_sizes, DT_test_mean, '-', color="c", label="Decision Tree")
    plt.plot(train_sizes, boosting_test_mean, '-', color="k", label="Boosting")
    plt.legend(loc="best")
    plt.savefig('t1_testing_accuracy.png')


    plt.figure()
    plt.title("Model Learning Speed")
    plt.xlabel("Training Examples")
    plt.ylabel("Fit Times")
    plt.plot(train_sizes, NN_fit_mean, '-', color="r", label="Neural Network")
    plt.plot(train_sizes, SVM_fit_mean, '-', color="b", label="SVM")
    plt.plot(train_sizes, KNN_fit_mean, '-', color="g", label="kNN")
    plt.plot(train_sizes, DT_fit_mean, '-', color="c", label="Decision Tree")
    plt.plot(train_sizes, boosting_fit_mean, '-', color="k", label="Boosting")
    plt.legend(loc="best")
    plt.savefig('t1_training_fit_times.png')
    
print('all tests completed')
pass
