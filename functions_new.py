import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from deslib.dcs.ola import OLA
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES


Colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def get_accaccs_curve(measure, classification_results, get_ordered_metrics=False):
	N = len(measure)
	measure_sorted = np.zeros(N)
	classification_results_sorted = np.zeros(N)

	# sort the data points by robustness measure for each of the models in the list
	arg_sorted_measure = np.argsort(measure)
	classification_results_sorted += classification_results[arg_sorted_measure]
	measure_sorted += measure[arg_sorted_measure]

	accs = np.zeros(N)

	# get the accuracy for each n
	for i in range(N):
		step_acc = classification_results_sorted[i:].sum()/(N-i)
		accs[i] = step_acc

	if get_ordered_metrics:
		return accs, measure_sorted, classification_results_sorted
	return accs


def get_ideal_accaccs(classification_results):
	N = len(classification_results)

	# compute ideal curve: all correct predictions first
	ideal_curve = np.zeros(N)
	ideal_curve[:int(classification_results.sum())] = 1
	ideal_accs = np.cumsum(ideal_curve)/np.arange(1, N+1)

	return ideal_accs


def get_worst_case_accaccs(classification_results):
	N = len(classification_results)

	# compute worst-case curve: all wrong predictions first
	wc_curve = np.zeros(N)
	wc_curve[-int(classification_results.sum()):] = 1
	wc_accs = np.cumsum(wc_curve)/np.arange(1, N+1)

	return wc_accs


def calculate_auc(accs, classification_results, total_auc=True):
	return np.mean(accs)
	ideal_accs = get_ideal_accaccs(classification_results)
	N = len(accs)

	# calculate AUC
	if total_auc:
		total_accuracy = 0
	else:
		total_accuracy = ideal_accs[-1]
	auc = 0
	for i in range(N-1):
		auc += ((accs[i] - total_accuracy)/(ideal_accs[i] - total_accuracy))
	auc /= (N-1)

	return auc


def accuracy_rejection_curve_single(measure, classification_results, ax=None, set_name=None, key=None, color=None):
    """
    Plot the mean accuracy-rejection curve for all models.
    The x-axis is the rejection rate and the y-axis is the accuracy.
    If split is True, the data points in each window are split by class.
    measure is an array of shape (MxN) that contains for each of the M models, the robustness measure for all N samples
    y contains the true labels for the N samples
    classification_results is an array of shape (MxN) that contains for each of the models the clasification result for each of the samples.
    The classification result is 1 if the prediction was right, and 0 otherwise
    """
    N = len(measure)
    accs = get_accaccs_curve(measure, classification_results)

    ### plot the figure ###
    # if no axis is given, create a new figure
    # else plot on the given axis
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))
        if key is not None:
            ax.set_title(key)

    if color is None:
        color = 'green'


    ax.plot(np.linspace(0, 100, N), accs, color=color)
    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy")

    if not ax:
        # save plot
        if set_name:
            plt.savefig(f"plots/{set_name}/{set_name}_acc_rej_rob.png")
        plt.show()



def accuracy_rejection_curve_single_AUC(measure, classification_results, ax=None, set_name=None, key=None, color=None):
    """
    Plot the mean accuracy-rejection curve for all models.
    The x-axis is the rejection rate and the y-axis is the accuracy.
    If split is True, the data points in each window are split by class.
    measure is an array of shape (MxN) that contains for each of the M models, the robustness measure for all N samples
    y contains the true labels for the N samples
    classification_results is an array of shape (MxN) that contains for each of the models the clasification result for each of the samples.
    The classification result is 1 if the prediction was right, and 0 otherwise
    """
    N = len(measure)
    accs = get_accaccs_curve(measure, classification_results)

    auc = calculate_auc(accs, classification_results)
    print("\nAUC: ", auc)

    ### plot the figure ###
    # if no axis is given, create a new figure
    # else plot on the given axis
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))
        if key is not None:
            ax.set_title(key)

    if color is None:
        color = 'green'


    ax.plot(np.linspace(0, 100, N), accs, color=color)
    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy")

    if not ax:
        # save plot
        if set_name:
            plt.savefig(f"plots/{set_name}/{set_name}_acc_rej_rob.png")
        plt.show()

    return auc


def accuracy_rejection_curve(measure, classification_results, percentiles=None, ax=None, set_name=None, mean=False, stddev=False, key=None, color=None, AUC=False):
    """
    Plot the mean accuracy-rejection curve for all models.
    The x-axis is the rejection rate and the y-axis is the accuracy.
    If split is True, the data points in each window are split by class.
    measure is an array of shape (MxN) that contains for each of the M models, the robustness measure for all N samples
    y contains the true labels for the N samples
    classification_results is an array of shape (MxN) that contains for each of the models the clasification result for each of the samples.
    The classification result is 1 if the prediction was right, and 0 otherwise
    """
    if len(measure.shape) == 1:
        if AUC:
            return accuracy_rejection_curve_single_AUC(measure, classification_results, ax=ax, set_name=set_name, key=key, color=color)
        accuracy_rejection_curve_single(measure, classification_results, ax=ax, set_name=set_name, key=key, color=color)
        return

    M = len(measure)
    N = len(measure[0])

    measure_bis = np.zeros(measure.shape)
    classification_results_bis = np.zeros(classification_results.shape)

    # sort the data points by robustness measure for each of the models in the list
    for i in range(M):
        S = sorted(zip(measure[i], classification_results[i]), key=lambda x: x[0])
        measure_bis[i] = np.array([x[0] for x in S])
        classification_results_bis[i] = np.array([x[1] for x in S])


    # initialize array to keep track of the accuracy for each n
    # the length of the array depends on the step size
    accs = np.zeros(N)
    if percentiles is not None:
        percents = np.zeros((N, 2))
    if stddev:
        stddevs = np.zeros(N)

    # get the accuracy for each n
    for i in range(N):
        step_accs = []
        for j in range(M):
            step_acc = classification_results_bis[j][i:].sum()/(N-i)
            step_accs.append(step_acc)
        accs[i] = np.mean(step_accs)
        if stddev:
            stddevs[i] = np.std(step_accs)
        if percentiles is not None:
            percents[i] = np.percentile(step_accs, percentiles)

    if AUC:
        auc = calculate_auc(accs, classification_results)
        print("\nAUC: ", auc)
    
    ### plot the figure ###
    # if no axis is given, create a new figure
    # else plot on the given axis

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))
        if key is not None:
            ax.set_title(key)

    if color is None:
        color = 'green'

    if mean:
        ax.plot(np.linspace(0, 100, N), accs, color=color)
        ax.set_ylabel("Mean of accuracy")
        if stddev:
            ax.fill_between(np.linspace(0, 100, N), accs-2*stddevs, accs+2*stddevs, alpha=0.5, color=color)
    elif stddev:
        ax.plot(np.linspace(0, 100, N), stddevs, color=color)
        ax.set_ylabel("Standard deviation of accuracy")
    else:
        ax.set_ylabel("Accuracy")
    if percentiles is not None:
        ax.plot(np.linspace(0, 100, N), percents[::-1][:, 0], color=color)
        ax.plot(np.linspace(0, 100, N), percents[::-1][:, 1], color=color)
        ax.fill_between(np.linspace(0, 100, N), percents[::-1][:, 0], percents[::-1][:, 1], alpha=0.3, color=color)
    # ax.plot(np.arange(0, 100, 100/N), accs)
    # set axis labels
    ax.set_xlabel("Rejection rate")


    if not ax:
        # save plot
        if set_name:
            plt.savefig(f"plots/{set_name}/{set_name}_acc_rej_rob.png")
        plt.show()


def combine_plots_acc_rej(rob_measures, classification_results, set_name=None, fig_size=None, percentiles=None, mean=False, stddev=False, keys=None, show=True, title_extra="", folder=None, y_min=0.75, y_max=0.35, legend_loc='upper right', colors=Colors, AUC=False):
    if fig_size is None:
        fig_size = (8,2)

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    axs = [axs]

    counter = 0
    for rob_key in rob_measures:
        if keys is not None:
            if rob_key not in keys:
                continue

        accuracy_rejection_curve(rob_measures[rob_key], classification_results,  ax=axs[0], set_name=None, percentiles=percentiles, color=colors[counter], mean=mean, stddev=stddev, AUC=AUC)
        counter += 1

    extra = ""
    if mean:
        extra += " mean"
    if stddev:
        extra += " std dev"
    if percentiles is not None:
        extra += " percentiles"
    # title of the plot
    fig.suptitle("Accuracy Rejection curves" + extra + " " + title_extra)

    # titles and legends of the subplots
    # axs[0].set_title("All points")
    if keys is not None:
        l = keys
    else:
        l = list(rob_measures.keys())

    legend_location = legend_loc
    if not mean and stddev:
        if legend_loc == 'empty':
            legend_location = 'upper right'
        plt.ylim(0, y_max)
    else:
        plt.ylim(y_min, 1.01)

    if legend_loc != 'empty':
        if colors is not None:
            axs[0].legend(l, labelcolor=colors, loc=legend_location)
            leg = axs[0].get_legend()
            for i in range(len(l)):
                leg.legend_handles[i].set_color(colors[i])
        else:
            axs[0].legend(l, loc=legend_location)

    # plt.tight_layout()
    plt.xlim(0, 100)

    if set_name:
        if folder is None:
            folder = 'plots'
        # robs = '_'.join(list(rob_measures.keys()))
        # plt.savefig(f"plots/{set_name}/{set_name}_{robs}_acc_rej.png")
        plt.savefig(f"{folder}/{set_name}.png")

    if show:
        plt.show()
    else:
        plt.close()


def models_acc_rej(rob_measures, classification_results, set_name=None, fig_size=None, percentiles=None, mean=False, stddev=False, keys=None, show=True, title_extra="", folder=None, y_min=0.75, y_max=0.35, legend_loc='upper right', colors=Colors, AUC=False):
    print('')
    if fig_size is None:
        fig_size = (8,2)

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    axs = [axs]

    counter = 0
    for rob_key in rob_measures:
        if keys is not None:
            if rob_key not in keys:
                continue

        accuracy_rejection_curve(rob_measures[rob_key], classification_results[rob_key],  ax=axs[0], set_name=None, percentiles=percentiles, color=colors[counter], mean=mean, stddev=stddev, AUC=AUC)
        counter += 1
            
    
    extra = ""
    if mean:
        extra += " mean"
    if stddev:
        extra += " std dev"
    if percentiles is not None:
        extra += " percentiles"
    # title of the plot
    fig.suptitle("Accuracy Rejection curves" + extra + " " + title_extra)

    # titles and legends of the subplots
    # axs[0].set_title("All points")
    if keys is not None:
        l = keys
    else:
        l = list(rob_measures.keys())

    legend_location = legend_loc
    if not mean and stddev:
        if legend_loc == 'empty':
            legend_location = 'upper right'
        plt.ylim(0, y_max)
    else:
        plt.ylim(y_min, 1.01)

    if legend_loc != 'empty':
        if colors is not None:
            axs[0].legend(l, labelcolor=colors, loc=legend_location)
            leg = axs[0].get_legend()
            for i in range(len(l)):
                leg.legend_handles[i].set_color(colors[i])
        else:
            axs[0].legend(l, loc=legend_location)

    # plt.tight_layout()
    plt.xlim(0, 100)

    if set_name:
        if folder is None:
            folder = 'plots'
        # robs = '_'.join(list(rob_measures.keys()))
        # plt.savefig(f"plots/{set_name}/{set_name}_{robs}_acc_rej.png")
        plt.savefig(f"{folder}/{set_name}.png")

    if show:
        plt.show()
    else:
        plt.close()


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


# For the list of chosen models, return a matrix of robustness and of predictions
def mods2rob_preds(X_test, mod_list, scores, n_mods, rob_method):
    n=len(X_test)
    m=len(mod_list)
    pred_mat=np.zeros([n,m+1])
    for i in range(m):
        pred_mat[:,i]=mod_list[i].predict(X_test)
    chosen_mods=mod_list[np.argsort(scores)[::-1][:n_mods]]
    rob_mat=np.zeros([n, n_mods])
    for i in range(n_mods):
        rob_mat[:,i]=rob_method(np.log(chosen_mods[i].predict_proba(X_test)))
    rob_ind = rob_mat.argsort()[:,n_mods-1]
    pred_mat[:,m]=pred_mat[np.arange(pred_mat.shape[0]), rob_ind]
    return pred_mat


def resamp_metrics(X, y, all_mods, B, n_mods, test_size, rob_method):
    b_metrics=np.zeros([B,len(all_mods)+1])
    for i in range(B):
        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= i)
        # Model training
        [all_mods[j].fit(X_train, y_train) for j in range(len(all_mods))]
        # Model scores
        scores=np.array([all_mods[j].best_score_ for j in range(len(all_mods))])
        # Predictions of all models, most robust in the end
        pred_mat = mods2rob_preds(X_test, all_mods, scores, n_mods, rob_method)
        # Saving the scores (accuracy)
        b_metrics[i,:]=[accuracy_score(y_test, pred_mat[:, j]) for j in range(len(pred_mat[0]))]
    return b_metrics


def acc_bjump_k_n(B, X, y, ptrain, pval, all_mods, n, k_max = 10, correct=False):
    m=len(all_mods)
    samp_scores = np.zeros([B,m+12])

    for b in range(B):
        Xn=X
        yn=y
        if n < len(X):
            Xn, _, yn, _ = train_test_split(Xn, yn, train_size=n, random_state=b, stratify=y)
        # Data splitting
        X_train, X2, y_train, y2 = train_test_split(Xn, yn, train_size=ptrain, random_state=b, stratify=yn)
        X_val, X_test, y_val, y_test = train_test_split(X2, y2, train_size=pval/(1-ptrain), random_state=b, stratify=y2)

        #TRAINING DATA
        # Train all models
        [all_mods[j].fit(X_train, y_train) for j in range(m)]

        #VALIDATION DATA
        n_val=len(y_val)

        # DESlib models with selection of k
        acc_mix = np.zeros([k_max-1,7])
        # Evaluating k on validation data
        for k in range(2,k_max+1):
            mix_mods = [OLA(all_mods, k = k), MCB(all_mods, k = k), APriori(all_mods, k = k),
                        KNORAU(all_mods, k = k), KNORAE(all_mods, k = k), DESP(all_mods, k = k),
                        METADES(all_mods, k = k)
                       ]
            [mix_mods[i].fit(X_val, y_val) for i in range(7)]
            acc_mix[k-2,:] = np.array([mix_mods[i].score(X_val, y_val) for i in range(7)])
        # Choosing best k for each strategy
        best_k = acc_mix.argmax(axis=0)+2
        # Applying best k
        mix_mods = [OLA(all_mods, k = int(best_k[0])), MCB(all_mods, k = int(best_k[1])), APriori(all_mods, k = int(best_k[2])),
                    KNORAU(all_mods, k = int(best_k[3])), KNORAE(all_mods, k = int(best_k[4])), DESP(all_mods, k = int(best_k[5])),
                    METADES(all_mods, k = int(best_k[6]))
                   ]
        [mix_mods[i].fit(X_val, y_val) for i in range(7)]
        
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

        # Case 1: best_acc
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

        # Case 2: highest ARC jump
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

        pred_te=np.zeros([n_test,m+12])
        for i in range(m):
            pred_te[:,i]=all_mods[i].predict(X_test)
        
        # Predictions of chosen model
        pred_te[:,m]=pred_M1

        # Predictions of DESlib models
        for i in range(7):
            pred_te[:,m+i+1] = mix_mods[i].predict(X_test)
        
        # Predictions of robustness strategies
        ## Percentage threshold (best)
        pred_te[:,m+8]=pred_M1
        pred_te[cd_pct_b,m+8]=pred_M2[cd_pct_b]
        ## Robustness threshold (best)
        pred_te[:,m+9]=pred_M1
        pred_te[cd_rob_b,m+9]=pred_M2[cd_rob_b]
        ## Percentage threshold (high)
        pred_te[:,m+10]=pred_M1
        pred_te[cd_pct_h,m+10]=pred_M2[cd_pct_h]
        ## Robustness threshold (high)
        pred_te[:,m+11]=pred_M1
        pred_te[cd_rob_h,m+11]=pred_M2[cd_rob_h]

        # Compare on test data
        samp_scores[b,:]=[accuracy_score(y_test, pred_te[:,j]) for j in range(len(pred_te[0]))]
    
    return samp_scores


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


def acc_bjump_noise(B, X, y, ptrain, pval, all_mods, n, noise_fraction=0.2, k_max = 10, correct=False):
    m=len(all_mods)
    samp_scores = np.zeros([B,m+12])

    for b in range(B):
        Xn=X
        yn=y
        if n < len(X):
            Xn, _, yn, _ = train_test_split(Xn, yn, train_size=n, random_state=b, stratify=y)
        # Data splitting
        X_train, X2, y_train, y2 = train_test_split(Xn, yn, train_size=ptrain, random_state=b, stratify=yn)
        X_val, X_test, y_val, y_test = train_test_split(X2, y2, train_size=pval/(1-ptrain), random_state=b, stratify=y2)

        # Adding noise to training and validation
        classes=np.unique(y)
        y_train = introduce_label_noise(y_train, classes, noise_fraction, random_state=b)
        y_val = introduce_label_noise(y_val, classes, noise_fraction, random_state=b)
        
        #TRAINING DATA
        # Train all models
        [all_mods[j].fit(X_train, y_train) for j in range(m)]

        #VALIDATION DATA
        n_val=len(y_val)

        # DESlib models with selection of k
        acc_mix = np.zeros([k_max-1,7])
        # Evaluating k on validation data
        for k in range(2,k_max+1):
            mix_mods = [OLA(all_mods, k = k), MCB(all_mods, k = k), APriori(all_mods, k = k),
                        KNORAU(all_mods, k = k), KNORAE(all_mods, k = k), DESP(all_mods, k = k),
                        METADES(all_mods, k = k)
                       ]
            [mix_mods[i].fit(X_val, y_val) for i in range(7)]
            acc_mix[k-2,:] = np.array([mix_mods[i].score(X_val, y_val) for i in range(7)])
        # Choosing best k for each strategy
        best_k = acc_mix.argmax(axis=0)+2
        # Applying best k
        mix_mods = [OLA(all_mods, k = int(best_k[0])), MCB(all_mods, k = int(best_k[1])), APriori(all_mods, k = int(best_k[2])),
                    KNORAU(all_mods, k = int(best_k[3])), KNORAE(all_mods, k = int(best_k[4])), DESP(all_mods, k = int(best_k[5])),
                    METADES(all_mods, k = int(best_k[6]))
                   ]
        [mix_mods[i].fit(X_val, y_val) for i in range(7)]
        
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

        # Case 1: best_acc
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

        # Case 2: highest ARC jump
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

        pred_te=np.zeros([n_test,m+12])
        for i in range(m):
            pred_te[:,i]=all_mods[i].predict(X_test)
        
        # Predictions of chosen model
        pred_te[:,m]=pred_M1

        # Predictions of DESlib models
        for i in range(7):
            pred_te[:,m+i+1] = mix_mods[i].predict(X_test)
        
        # Predictions of robustness strategies
        ## Percentage threshold (best)
        pred_te[:,m+8]=pred_M1
        pred_te[cd_pct_b,m+8]=pred_M2[cd_pct_b]
        ## Robustness threshold (best)
        pred_te[:,m+9]=pred_M1
        pred_te[cd_rob_b,m+9]=pred_M2[cd_rob_b]
        ## Percentage threshold (high)
        pred_te[:,m+10]=pred_M1
        pred_te[cd_pct_h,m+10]=pred_M2[cd_pct_h]
        ## Robustness threshold (high)
        pred_te[:,m+11]=pred_M1
        pred_te[cd_rob_h,m+11]=pred_M2[cd_rob_h]

        # Compare on test data
        samp_scores[b,:]=[accuracy_score(y_test, pred_te[:,j]) for j in range(len(pred_te[0]))]
    
    return samp_scores