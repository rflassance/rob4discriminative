import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from gefs import compute_rob_class
from gefs.sklearn_utils import tree2pc, rf2pc
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


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
                leg.legendHandles[i].set_color(colors[i])
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
                leg.legendHandles[i].set_color(colors[i])
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


def gef_rob(X, y, cats, model, M=10, test_size=0.3):
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

    for i in range(M):
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