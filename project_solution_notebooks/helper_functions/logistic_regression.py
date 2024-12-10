'''Helper functions for logistic regression project solution notebook.'''

from sklearn.metrics import confusion_matrix
import pandas as pd


def calculate_cashflow(
        training_features,
        training_labels,
        base_training_predictions,
        weighted_training_predictions,
        cost_to_call_customer,
        return_on_deposit
):
    
    '''Uses some made up numbers to calculate mock cash flow data using
    predictions from the base and weighted models.'''

    # Make a dictionary to hold our toy data
    results={
        'Cost': [],
        'Profit': [],
    }

    # List for model types/conditions
    models=[]

    # Baseline: no model - we call everyone
    calls_made=len(training_features)
    total_cost=calls_made * cost_to_call_customer
    deposits_sold=int(sum(training_labels))
    revenue=deposits_sold * return_on_deposit
    profit=revenue - total_cost

    models.append('None')
    results['Cost'].append(total_cost)
    results['Profit'].append(profit)

    # Model without class weighting - we only call the people the model predicts
    # are going to take the deposit
    cm=confusion_matrix(base_training_predictions, training_labels).transpose()

    calls_made=int(cm[0,1] + cm[1,1])
    total_cost=calls_made * cost_to_call_customer
    deposits_sold=int(cm[1,1])
    revenue=deposits_sold * return_on_deposit
    profit=revenue - total_cost

    models.append('Base model')
    results['Cost'].append(total_cost)
    results['Profit'].append(profit)

    # Model with class weighting
    cm=confusion_matrix(weighted_training_predictions, training_labels).transpose()

    calls_made=int(cm[0,1] + cm[1,1])
    total_cost=calls_made * cost_to_call_customer
    deposits_sold=int(cm[1,1])
    revenue=deposits_sold * return_on_deposit
    profit=revenue - total_cost

    models.append('Weighted model')
    results['Cost'].append(total_cost)
    results['Profit'].append(profit)

    return results, models