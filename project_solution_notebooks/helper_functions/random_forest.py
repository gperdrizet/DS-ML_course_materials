'''Helper functions for random forest project solution notebook.'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold

def cross_val(model, features: pd.DataFrame, labels: pd.Series) -> list[float]:
    '''Reusable helper function to run cross-validation on a model. Takes model,
    Pandas data frame of features and Pandas data series of labels. Returns 
    list of cross-validation fold accuracy scores as percents.'''

    # Define the cross-validation strategy
    cross_validation=StratifiedKFold(n_splits=7, shuffle=True, random_state=315)

    # Run the cross-validation, collecting the scores
    scores=cross_val_score(
        model,
        features,
        labels,
        cv=cross_validation,
        n_jobs=-1,
        scoring='accuracy'
    )

    # Print mean and standard deviation of the scores
    print(f'Cross-validation accuracy: {(scores.mean() * 100):.2f} +/- {(scores.std() * 100):.2f}%')

    # Return the scores
    return scores


def plot_cross_validation(title: str, results: dict) -> plt:
    '''Takes a list of dictionary of cross validation results
    plots as horizontal box-and-whiskers plot. Returns plot
    object.'''

    box_plot=sns.boxplot(
        data = pd.DataFrame.from_dict(results),
        orient = 'h'
    )

    medians=[]

    for scores in results.values():
        medians.append(np.median(scores))

    for ytick in box_plot.get_yticks():
        box_plot.text(medians[ytick],ytick,f'{medians[ytick]:.1f}%',
            horizontalalignment='center',size='x-small',color='black',weight='semibold',
            bbox=dict(facecolor='gray', edgecolor='black'))

    plt.title(title)
    plt.xlabel('Accuracy (%)')


    return plt


def plot_hyperparameter_tuning(results: dict) -> plt:
    '''Takes RandomizedSearchCV result object, plots cross-validation
    train and test scores for each fold.'''

    results=pd.DataFrame(results.cv_results_)
    sorted_results=results.sort_values('rank_test_score')

    plt.title('Hyperparameter optimization')
    plt.xlabel('Parameter set rank')
    plt.ylabel('Accuracy')
    plt.gca().invert_xaxis()

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score'] + sorted_results['std_test_score'], # pylint: disable=line-too-long
        sorted_results['mean_test_score'] - sorted_results['std_test_score'], # pylint: disable=line-too-long
        alpha = 0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score'],
        label = 'Validation'
    )

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_train_score'] + sorted_results['std_train_score'], # pylint: disable=line-too-long
        sorted_results['mean_train_score'] - sorted_results['std_train_score'], # pylint: disable=line-too-long
        alpha = 0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_train_score'],
        label = 'Training'
    )

    plt.legend(loc = 'best', fontsize = 'x-small')

    return plt
