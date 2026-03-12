import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deslib.dcs.mcb import MCB
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES


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

def introduce_label_noise(y_train, classes, noise_fraction=0.05, random_state=42):
    number_of_classes = len(classes)
    np.random.seed(random_state)
    n_samples = len(y_train)
    n_noisy = int(noise_fraction * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    y_train_noisy = y_train.copy()
    for index in noisy_indices:
        current_label = y_train[index]
        possible_labels = np.sort(classes)
        possible_labels = possible_labels[possible_labels != current_label]
        new_label = np.random.choice(possible_labels)
        y_train_noisy[index] = new_label

    return y_train_noisy

def acc_dyn_sel(B, X, y, ptrain, pval, all_mods, noise_frac = 0, k_max = 10, correct=False):
    m=len(all_mods)
    samp_scores = np.zeros([B,m+7])

    for b in range(B):
        Xn=X
        yn=y
        # Data splitting
        X_train, X2, y_train, y2 = train_test_split(Xn, yn, train_size=ptrain, random_state=b, stratify=yn)
        X_val, X_test, y_val, y_test = train_test_split(X2, y2, train_size=pval/(1-ptrain), random_state=b, stratify=y2)

        # Adding noise to training and validation if required
        if noise_frac != 0:
            classes=np.unique(y)
            y_train = introduce_label_noise(y_train, classes, noise_frac, random_state=b)
            y_val = introduce_label_noise(y_val, classes, noise_frac, random_state=b)
        
        #TRAINING DATA
        # Train all models
        [all_mods[j].fit(X_train, y_train) for j in range(m)]

        #VALIDATION DATA
        n_val=len(y_val)

        # DESlib models with selection of k
        acc_mix = np.zeros([k_max-1,4])
        # Evaluating k on validation data
        for k in range(2,k_max+1):
            mix_mods = [
                        MCB(all_mods, k = k),
                        KNORAU(all_mods, k = k),
                        KNORAE(all_mods, k = k),
                        METADES(all_mods, k = k)
                       ]
            [mix_mods[i].fit(X_val, y_val) for i in range(4)]
            acc_mix[k-2,:] = np.array([mix_mods[i].score(X_val, y_val) for i in range(4)])
        # Choosing best k for each strategy
        best_k = acc_mix.argmax(axis=0)+2
        # Applying best k
        mix_mods = [
                    MCB(all_mods, k = int(best_k[0])),
                    KNORAU(all_mods, k = int(best_k[1])),
                    KNORAE(all_mods, k = int(best_k[2])),
                    METADES(all_mods, k = int(best_k[3]))
                   ]
        [mix_mods[i].fit(X_val, y_val) for i in range(4)]
        
        # Get the one with the best accuracy in training
        scores=np.array([accuracy_score(y_val, all_mods[j].predict(X_val)) for j in range(m)])

        # Get the two best models and apply the robustness scheme
        score_order=scores.argsort()[::-1]
        mod_ord=all_mods[score_order[[0,1]]]
        
        # Get robustness metric of the second best model and sort the data
        rob1=ratio_robustness(np.log(mod_ord[0].predict_proba(X_val)), correct)
        rob2=ratio_robustness(np.log(mod_ord[1].predict_proba(X_val)), correct)
        rob_ratio=rob2/rob1
        rob_order_ratio=rob_ratio.argsort()

        # Case 1: best_acc (RS-D)
        # Order the validation data and get the predictions
        X_ord=X_val[rob_order_ratio,:]
        y_ord=y_val[rob_order_ratio]
        M1_res=(mod_ord[0].predict(X_ord)==y_ord)*1
        M2_res=(mod_ord[1].predict(X_ord)==y_ord)*1
        M2_res=np.append(M2_res[1:],0) # First observation removed (never used), add 0 in the end
        # Pick the pct/rob threshold such that overall accuracy is the highest
        idx_M2=(M1_res.cumsum()+M2_res[::-1].cumsum()[::-1]).argmax()
        prop_b=(idx_M2+1)/n_val
        rob_b=rob_ratio[rob_order_ratio[idx_M2]]
        if rob_b==np.max(rob_ratio):
            rob_b=np.inf

        # Case 2: highest ARC jump (RS-I)
        # Order the validation data and get the predictions
        X_ord=X_val[rob_order_ratio,:]
        y_ord=y_val[rob_order_ratio]
        M1_res=(mod_ord[0].predict(X_ord)==y_ord)*1
        M2_res=(mod_ord[1].predict(X_ord)==y_ord)*1
        n_ord=len(M2_res)
        # Checking where the ARC difference between M2 and M1 is the highest
        M2_win=(M2_res[::-1].cumsum()[::-1] - M1_res[::-1].cumsum()[::-1])/np.ones(n_ord).cumsum()[::-1]
        if any(M2_win>0):
            idx_M2=M2_win.argmax()
            prop_h=(idx_M2+1)/n_ord
            rob_h=rob_ratio[rob_order_ratio[idx_M2]]
        else:
            prop_h=1
            rob_h=np.inf
        
        #TEST DATA
        # Order the test data by M2 robustness and predict them with M2 only when over the rejection percentage (or robustness threshold)
        rob1_test=ratio_robustness(np.log(mod_ord[0].predict_proba(X_test)), correct)
        rob2_test=ratio_robustness(np.log(mod_ord[1].predict_proba(X_test)), correct)
        n_test=len(X_test)

        cd_pct_b=(rob2_test/rob1_test).argsort()[np.linspace(1,n_test,num=n_test)/n_test>prop_b]
        cd_rob_b=(rob2_test/rob1_test)>rob_b

        cd_pct_h=(rob2_test/rob1_test).argsort()[np.linspace(1,n_test,num=n_test)/n_test>prop_h]
        cd_rob_h=(rob2_test/rob1_test)>rob_h

        pred_M1=mod_ord[0].predict(X_test)
        pred_M2=mod_ord[1].predict(X_test)

        pred_te=np.zeros([n_test,m+7])
        for i in range(m):
            pred_te[:,i]=all_mods[i].predict(X_test)
        
        # Predictions of chosen model
        pred_te[:,m]=pred_M1

        # Predictions of DESlib models
        for i in range(4):
            pred_te[:,m+i+1] = mix_mods[i].predict(X_test)
        
        # Predictions of robustness strategies
        ## RS-D
        pred_te[:,m+5]=pred_M1
        pred_te[cd_rob_b,m+5]=pred_M2[cd_rob_b]
        ## RS-I
        pred_te[:,m+6]=pred_M1
        pred_te[cd_rob_h,m+6]=pred_M2[cd_rob_h]

        # Compare on test data
        samp_scores[b,:]=[accuracy_score(y_test, pred_te[:,j]) for j in range(len(pred_te[0]))]
    
    return samp_scores