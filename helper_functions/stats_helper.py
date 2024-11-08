import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
from typing import List, Union

def chow_test(
        data1: pd.DataFrame, 
        data2: pd.DataFrame, 
        features: Union[str, List[str]],
        response: str, ):
    """
    Conducts a Chow test to determine whether two datasets can be represented 
    by a single linear regression model or if they require separate models 
    due to structural differences in the relationships between the response 
    variable and predictors.

    This is an F test where the null hypothesis is that the data can be
    represented as a single linear regression model and the alternative
    hypothesis is that they cannot.

    Args:
        - data1 (pd.DataFrame): The first dataset containing the response and
        predictor variables.
        - data2 (pd.DataFrame): The second dataset containing the response and
        predictor variables.
        - response (str): Name of the response (dependent) variable.
        - predictors (str or List[str]): Name(s) of predictor (independent)
        variable(s).

    Returns:
        - F_stat (float): The F-statistic for the Chow test.
        - p_value (float): The p-value indicating statistical significance of
        the test.

    Raises:
        ValueError: If either data set does not contain the specified response
        or predictor variable.
    """
    if isinstance(features, str):
        features = [features]
    response = [response]
    required_columns = features + response


    
    if not set(required_columns).issubset(set(data1.columns) & set(data2.columns)):
        raise ValueError(
            "Both datasets must contain the features and the response variable"
            )

    data1 = data1[required_columns]
    data2 = data2[required_columns]
    data_pooled = pd.concat([data1, data2])

    X1 = sm.add_constant(data1[features])
    Y1 = data1[response]
    model1 = sm.OLS(Y1, X1).fit()
    rss1 = sum(model1.resid ** 2)

    X2 = sm.add_constant(data2[features])
    Y2 = data2[response]
    model2 = sm.OLS(Y2, X2).fit()
    rss2 = sum(model2.resid ** 2)

    X_pooled = sm.add_constant(data_pooled[features])
    Y_pooled = data_pooled[response]
    model_pooled = sm.OLS(Y_pooled, X_pooled).fit()
    rss_pooled = sum(model_pooled.resid ** 2)

    rss_summed = rss1 + rss2
    n1 = len(data1)
    n2 = len(data2)
    k = len(features) + 1
    degrees_freedom = n1 + n2 - (k * 2)

    F_stat =  ((rss_pooled - rss_summed) / k) / (rss_summed / degrees_freedom)
    p_value = 1 - f.cdf(F_stat, k, degrees_freedom)

    return F_stat, p_value
