'''Helper functions for decision trees project solution notebook.'''

import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer # pylint: disable=unused-import
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import QuantileTransformer


def impute(data_df: pd.DataFrame, missing_data_features: list) -> pd.DataFrame:
    '''Takes pandas dataframe and feature list. Runs
    Scikit-learn's IterativeImputer on specified features.
    Returns an updated dataframe and the fit imputer and
    quantile transformers for later use.'''

    # Save the feature names for later - the imputer will return a numpy array
    # and we might like to get out Pandas dataframe back
    feature_names=list(data_df.columns)

    # Make a copy of the training features dataframe, in case we decide that this
    # is a bad idea
    imputed_training_features=data_df.copy()
    imputed_training_features[missing_data_features]=(
        imputed_training_features[missing_data_features].replace({0:np.nan}))

    # Quantile transform our target features - this is for the imputer, not the decision tree
    qt=QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
    qt.fit(imputed_training_features[missing_data_features])
    imputed_training_features[missing_data_features]=(
        qt.transform(imputed_training_features[missing_data_features]))

    # Run the imputation
    imp=IterativeImputer(
        max_iter=100,
        verbose=True,
        tol=1e-6,
        sample_posterior=True,
        add_indicator=True
    )

    imp.fit(imputed_training_features)
    imputed_training_features=imp.transform(imputed_training_features)

    # Convert back to pandas
    indicator_features=[]
    for feature in missing_data_features:
        indicator_features.append(f'{feature}_indicator')

    feature_names.extend(indicator_features)

    imputed_training_features=pd.DataFrame(data=imputed_training_features, columns=feature_names)

    # Now that zeros have been replaces - undo the quantile transformation.
    # This way we are training the model on un-altered values. This way,
    # during deployment, we don't have to worry about transforming the input data.
    imputed_training_features[missing_data_features]=qt.inverse_transform(
        imputed_training_features[missing_data_features])

    return imputed_training_features, imp, qt


def plot_scatter_matrix(input_df: pd.DataFrame) -> plt:
    '''Plots scatter matrix of features in dataframe'''

    # Copy the input so we don't accidentally make any changes
    features_df=input_df.copy()

    # Get the cross correlation matrix
    correlations = features_df.corr(method = 'spearman')
    correlations = correlations.to_numpy().flatten()

    # Get the feature names from the dataframe
    num_features = len(features_df.columns)
    print(f'Have {num_features} features for plot:')

    # Number and rename the features for plotting
    feature_nums = {}

    for i, feature in enumerate(features_df.columns):
        print(f' {i}: {feature}')
        feature_nums[feature] = i

    features_df.rename(feature_nums, axis=1, inplace=True)

    # Get feature number pairs for each plot using the cartesian product of the feature numbers
    feature_pairs = itertools.product(list(range(num_features)), list(range(num_features)))

    # Assign fraction of plot width to scatter matrix and colorbar
    colorbar_fraction=0.05
    scatter_fraction=1 - colorbar_fraction

    # Set the width of the figure based on number of features
    single_plot_width=0.75
    fig_height=num_features * single_plot_width

    # Now, set the total width such that fraction occupied by the scatter matrix
    # equals the height. This will let us draw a square cross-corelation matrix
    # with the right amount of space left for the colorbar
    fig_width=fig_height / scatter_fraction

    # Set-up two subfigures, one for the scatter matrix and one for the colorbar
    fig=plt.figure(
        figsize=(fig_width, fig_height)
    )

    subfigs=fig.subfigures(
        1,
        2,
        wspace=0,
        hspace=0,
        width_ratios=[scatter_fraction, colorbar_fraction]
    )

    axs1=subfigs[0].subplots(
        num_features,
        num_features,
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )

    # Get the colormap
    cmap=mpl.colormaps['viridis']

    # Construct a normalization function to map correlation values onto the colormap
    norm=mpl.colors.Normalize(vmin=min(correlations), vmax=max(correlations))

    # Counters to keep track of where we are in the grid
    plot_count=0
    row_count=0
    column_count=0

    # Loop to draw each plot
    for feature_pair, correlation, ax in zip(feature_pairs, correlations, axs1.flatten()):

        first_feature=feature_pair[0]
        second_feature=feature_pair[1]

        ax.scatter(
            features_df[first_feature],
            features_df[second_feature],
            s=0.2,
            color=[cmap(norm(correlation))],
            alpha=0.9
        )

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # Set y label for first plot in each row only
        if column_count == 0:
            ax.set_ylabel(first_feature)
        else:
            ax.set_ylabel('')

        # Update grid counters
        plot_count += 1
        column_count += 1

        if column_count == num_features:
            column_count = 0
            row_count += 1

        # Set x label for plots in last row only
        if row_count == num_features:
            ax.set_xlabel(first_feature)
        else:
            ax.set_xlabel('')

        ax.set_xlabel(second_feature)

    plt.tight_layout()

    axs2 = subfigs[1].subplots(1, 1)
    color_bar = mpl.colorbar.ColorbarBase(
        ax = axs2,
        cmap = cmap,
        norm = norm
    )

    color_bar.set_label('Spearman correlation coefficient', size = 18)
    color_bar.ax.tick_params(labelsize = 14)

    return plt


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
