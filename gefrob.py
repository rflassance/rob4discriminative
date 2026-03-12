import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gefs import compute_rob_class
from gefs.sklearn_utils import rf2pc
from sklearn.metrics import accuracy_score


def ratio_robustness(pred_log_prob, correct=False):
    sorted_pred_prob = np.sort(pred_log_prob, axis=1)

    # get the highest and second highest probabilities for each data point
    max_prob = sorted_pred_prob[:,-1]
    second_prob = sorted_pred_prob[:,-2]

    # If the second best has probability 0, correct it
    if correct & (second_prob.min()==-np.inf):
        second_prob[second_prob==-np.inf] = np.log(0.001)
    
    # take elementwise minimum of the two measures
    robust = np.exp((max_prob - second_prob)/2)
    
    return robust


# For the generative forest, getting the joint log-probability
def gef_logprobs(X_test, cats, gef):
    n_cat=len(cats)
    Xy=np.zeros([n_cat*len(X_test),len(X_test[0])+1])
    ## For each line of X_test, repeat it n_cat times and add each unique entry of y
    for i in range(len(X_test)):
        for j in range(n_cat):
            Xy[n_cat*i+j,:]=np.append(X_test[i],cats[j])
    ## Get the log-probabilities
    logprobs=gef.log_likelihood(Xy)
    ## Reshape and return
    return logprobs.reshape([len(X_test),n_cat])


def gef_rob(X, y, cats, model, B=10, test_size=0.3):
    # Turn X into a dataframe in case it is not to infer dtype
    X=pd.DataFrame(X)
    
    ncat = np.ones(X.shape[1]+1) # This assumes all features are continuous
    
    # Changing the features for discrete if necessary
    X_type=np.array(X.dtypes)
    matchings_indices = [ i for i, x in enumerate(X_type) if x not in ['float','int'] ]
    for i in matchings_indices:
        ncat[i]=len(np.unique(X[X.columns[i]]))
    
    # Convert to numpy arrays if not already
    X = np.asarray(X)
    y = np.asarray(y)
    
    ncat[-1] = len(cats)  # The class variable is naturally categorical

    # Initialize lists to store results
    l_rob_list = []
    r_rob_list = []
    class_resul_list = []

    for i in range(B):
        # Step 1: Random sample without replacement of size n (with train/test splitting)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size,random_state= i)
        
        # Step 3: Fit the model on training data and transform it into a GeF
        model.fit(X_train, y_train)
        mod=model.best_estimator_
        gef = rf2pc(mod, X_train, y_train, ncat, learnspn=np.Inf, minstd=1., smoothing=1e-6)
        
        # Step 4: Calculate metrics
        
        ## Robustness(es)
        logprobs=gef_logprobs(X_test, cats, gef)
        _, l_rob = compute_rob_class(gef.root, X_test, X.shape[1], int(ncat[-1]))
        r_rob = ratio_robustness(logprobs)
        l_rob_list.append(l_rob)
        r_rob_list.append(r_rob)
        
        preds, pred_prob = gef.classify(X_test, return_prob=True)
    
        # Step 5: Add classification results
        class_resul_list.append((y_test == preds)*1)        

    return {"local_rob": np.array(l_rob_list, dtype=object),
            "ratio_rob": np.array(r_rob_list, dtype=object),
            "class_resul": np.array(class_resul_list, dtype=object)}