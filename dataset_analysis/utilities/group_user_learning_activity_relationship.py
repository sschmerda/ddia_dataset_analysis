from .standard_import import *
from .constants import *
from .config import *
from .functions import *

def avg_seq_len(series: pd.Series, avg_fun):
    """Calculates the average of the value counts of a pandas series.

    Parameters
    ----------
    series : pd.Series
        An input series.
    avg_fun : _type_
        An averaging function.

    Returns
    -------
    float
        The calculated average.
    """    
    vc = series.value_counts()
    return avg_fun(vc)

def print_and_return_interactions_per_user(interactions: pd.DataFrame, user_field: str, user_field_name: str):
    """Print and return interactions stats per user

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    user_field : str
        The user field column
    user_field_name : str
        The name of the user field

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of interactions per user
    """    
    interactions_per_user = interactions.groupby(user_field).size().reset_index().rename(columns={0: NUMBER_OF_INTERACTIONS_FIELD_NAME_STR})

    print(f'mean: number of interactions per {user_field_name}: {interactions_per_user[NUMBER_OF_INTERACTIONS_FIELD_NAME_STR].mean()}')
    print(f'median: number of interactions per {user_field_name}: {interactions_per_user[NUMBER_OF_INTERACTIONS_FIELD_NAME_STR].median()}')
    print(f'max: number of interactions per {user_field_name}: {interactions_per_user[NUMBER_OF_INTERACTIONS_FIELD_NAME_STR].max()}')
    print(f'min: number of interactions per {user_field_name}: {interactions_per_user[NUMBER_OF_INTERACTIONS_FIELD_NAME_STR].min()}')
    print(f'std: number of interactions per {user_field_name}: {interactions_per_user[NUMBER_OF_INTERACTIONS_FIELD_NAME_STR].std()}')
    print(f'iqr: number of interactions per {user_field_name}: {iqr(interactions_per_user[NUMBER_OF_INTERACTIONS_FIELD_NAME_STR])}')

    return interactions_per_user

def print_and_return_unique_col2_per_col1(interactions: pd.DataFrame, col1_field: str, col2_field: str, number_unique_col2_per_col1_name: str):
    """Print and return unique col2 vals per col1 val 

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    col1_field : str
        The col1 field column
    col2_field : str
        The col2 field column
    number_unique_col2_per_col1_name : str
        The name of the number of unique col2 vals per col1 val field

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of unique col2 vals per col1 val
    """
    n_unique_col2_per_col1 = interactions.groupby(col1_field)[col2_field].nunique().reset_index().rename(columns={col2_field: number_unique_col2_per_col1_name})

    print(f'mean: {number_unique_col2_per_col1_name}: {n_unique_col2_per_col1[number_unique_col2_per_col1_name].mean()}')
    print(f'median: {number_unique_col2_per_col1_name}: {n_unique_col2_per_col1[number_unique_col2_per_col1_name].median()}')
    print(f'max: {number_unique_col2_per_col1_name}: {n_unique_col2_per_col1[number_unique_col2_per_col1_name].max()}')
    print(f'min: {number_unique_col2_per_col1_name}: {n_unique_col2_per_col1[number_unique_col2_per_col1_name].min()}')
    print(f'std: {number_unique_col2_per_col1_name}: {n_unique_col2_per_col1[number_unique_col2_per_col1_name].std()}')
    print(f'iqr: {number_unique_col2_per_col1_name}: {iqr(n_unique_col2_per_col1[number_unique_col2_per_col1_name])}')

    return n_unique_col2_per_col1

def print_and_return_avg_num_interactions_over_col2_for_col1(interactions: pd.DataFrame, col1_field: str, col2_field: str, mean_number_interactions_over_col2_for_col1_name: str, median_number_interactions_over_col2_for_col1_name: str):
    """Print and return average number of interactions over col2 vals per col1 val.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    col1_field : str
        The col1 field column
    col2_field : str
        The col2 field column
    mean_number_interactions_over_col2_for_col1_name : str
        The name of the mean number of interactions over col2 vals for col1 val field
    median_number_interactions_over_col2_for_col1_name : str
        The name of the median number of interactions over col2 vals for col1 val field
        _description_

    Returns
    -------
    pd.DataFrame
        Two dataframes containing the mean and median number of interactions over col2 vals per col1 val
    """    
    mean_n_interactions_per_col2 = interactions.groupby(col1_field)[col2_field].agg(avg_seq_len, np.mean).sort_values(ascending=False).reset_index().rename(columns={col2_field: mean_number_interactions_over_col2_for_col1_name})
    median_n_interactions_per_col2 = interactions.groupby(col1_field)[col2_field].agg(avg_seq_len, np.median).sort_values(ascending=False).reset_index().rename(columns={col2_field: median_number_interactions_over_col2_for_col1_name})

    print(f'mean of {mean_number_interactions_over_col2_for_col1_name}: {mean_n_interactions_per_col2[mean_number_interactions_over_col2_for_col1_name].mean()}')
    print(f'median of {mean_number_interactions_over_col2_for_col1_name}: {mean_n_interactions_per_col2[mean_number_interactions_over_col2_for_col1_name].median()}')
    print(f'max of {mean_number_interactions_over_col2_for_col1_name}: {mean_n_interactions_per_col2[mean_number_interactions_over_col2_for_col1_name].max()}')
    print(f'min of {mean_number_interactions_over_col2_for_col1_name}: {mean_n_interactions_per_col2[mean_number_interactions_over_col2_for_col1_name].min()}')
    print(f'std of {mean_number_interactions_over_col2_for_col1_name}: {mean_n_interactions_per_col2[mean_number_interactions_over_col2_for_col1_name].std()}')
    print(f'iqr of {mean_number_interactions_over_col2_for_col1_name}: {iqr(mean_n_interactions_per_col2[mean_number_interactions_over_col2_for_col1_name])}')

    print('')
    print('_'*100)
    print('')

    print(f'mean of {median_number_interactions_over_col2_for_col1_name}: {median_n_interactions_per_col2[median_number_interactions_over_col2_for_col1_name].mean()}')
    print(f'median of {median_number_interactions_over_col2_for_col1_name}: {median_n_interactions_per_col2[median_number_interactions_over_col2_for_col1_name].median()}')
    print(f'max of {median_number_interactions_over_col2_for_col1_name}: {median_n_interactions_per_col2[median_number_interactions_over_col2_for_col1_name].max()}')
    print(f'min of {median_number_interactions_over_col2_for_col1_name}: {median_n_interactions_per_col2[median_number_interactions_over_col2_for_col1_name].min()}')
    print(f'std of {median_number_interactions_over_col2_for_col1_name}: {median_n_interactions_per_col2[median_number_interactions_over_col2_for_col1_name].std()}')
    print(f'iqr of {median_number_interactions_over_col2_for_col1_name}: {iqr(median_n_interactions_per_col2[median_number_interactions_over_col2_for_col1_name])}')

    return mean_n_interactions_per_col2, median_n_interactions_per_col2 

def print_and_return_avg_num_unique_col3_over_col2_for_col1(interactions: pd.DataFrame, col1_field: str, col2_field: str, col3_field: str, mean_number_unique_col3_over_col2_for_col1_name: str, median_number_unique_col3_over_col2_for_col1_name: str):
    """Print and return average number of col3 vals over cols2 vals for col1 val

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactins dataframe
    col1_field : str
        The col1 field column
    col2_field : str
        The col2 field column 
    col3_field : str
        The col3 field column
    mean_number_unique_col3_over_col2_for_col1_name : str
        The name of the mean number of col3 vals over col2 vals for col1 val field
    median_number_unique_col3_over_col2_for_col1_name : str
        The name of the median number of col3 vals over col2 vals for col1 val field

    Returns
    -------
    pd.DataFrame
       Two dataframes containing the mean and median number of unique col3 vals over cols2 vals for col1 val 
    """

    mean_n_unique_col3_per_col2 = interactions.groupby([col1_field, col2_field])[col3_field].nunique().reset_index().rename(columns={col3_field: 'n_unique_col3'}).sort_values(by=col1_field).groupby(col1_field)['n_unique_col3'].agg(np.mean).sort_values(ascending=False).reset_index().rename(columns={'n_unique_col3': mean_number_unique_col3_over_col2_for_col1_name}) 
    median_n_unique_col3_per_col2 = interactions.groupby([col1_field, col2_field])[col3_field].nunique().reset_index().rename(columns={col3_field: 'n_unique_col3'}).sort_values(by=col1_field).groupby(col1_field)['n_unique_col3'].agg(np.median).sort_values(ascending=False).reset_index().rename(columns={'n_unique_col3': median_number_unique_col3_over_col2_for_col1_name})

    print(f'mean of {mean_number_unique_col3_over_col2_for_col1_name}: {mean_n_unique_col3_per_col2[mean_number_unique_col3_over_col2_for_col1_name].mean()}')
    print(f'median of {mean_number_unique_col3_over_col2_for_col1_name}: {mean_n_unique_col3_per_col2[mean_number_unique_col3_over_col2_for_col1_name].median()}')
    print(f'max of {mean_number_unique_col3_over_col2_for_col1_name}: {mean_n_unique_col3_per_col2[mean_number_unique_col3_over_col2_for_col1_name].max()}')
    print(f'min of {mean_number_unique_col3_over_col2_for_col1_name}: {mean_n_unique_col3_per_col2[mean_number_unique_col3_over_col2_for_col1_name].min()}')


    print('')
    print('_'*100)
    print('')


    print(f'mean of {median_number_unique_col3_over_col2_for_col1_name}: {median_n_unique_col3_per_col2[median_number_unique_col3_over_col2_for_col1_name].mean()}')
    print(f'median of {median_number_unique_col3_over_col2_for_col1_name}: {median_n_unique_col3_per_col2[median_number_unique_col3_over_col2_for_col1_name].median()}')
    print(f'max of {median_number_unique_col3_over_col2_for_col1_name}: {median_n_unique_col3_per_col2[median_number_unique_col3_over_col2_for_col1_name].max()}')
    print(f'min of {median_number_unique_col3_over_col2_for_col1_name}: {median_n_unique_col3_per_col2[median_number_unique_col3_over_col2_for_col1_name].min()}')

    return mean_n_unique_col3_per_col2, median_n_unique_col3_per_col2

# function that prints and plots all of the group_user_learning_activity_relationship stats
def print_and_plot_group_user_learning_activity_relationship(interactions: pd.DataFrame,
                                                             group_field: str,
                                                             log_scale: bool):

    if not group_field:
        interactions[GROUP_FIELD_NAME_STR] = '0'

    # interactions per user
    print('*'*100)
    print('*'*100)
    print(' ')
    print('-'*100)
    print(f'Interactions per {USER_FIELD_NAME_STR}:')
    print('-'*100)
    interactions_per_user = print_and_return_interactions_per_user(interactions, 
                                                                   USER_FIELD_NAME_STR, 
                                                                   USER_FIELD_NAME_STR)
    print('\n')
    print('Plots:')
    plot_distribution(interactions_per_user,
                      NUMBER_OF_INTERACTIONS_FIELD_NAME_STR,
                      NUMBER_OF_INTERACTIONS_PER_USER_STR,
                      log_scale)
    print('*'*100)
    print('*'*100)

    # group_user_learning_activity_relationship
    parameter_list = [(GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, NUMBER_UNIQUE_GROUPS_PER_USER_STR),
                      (LEARNING_ACTIVITY_FIELD_NAME_STR, USER_FIELD_NAME_STR, NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_USER_STR),
                      (USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, NUMBER_UNIQUE_USERS_PER_GROUP_STR),
                      (LEARNING_ACTIVITY_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_STR),
                      (USER_FIELD_NAME_STR, LEARNING_ACTIVITY_FIELD_NAME_STR, NUMBER_UNIQUE_USERS_PER_LEARNING_ACTIVITY_STR),
                      (GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_FIELD_NAME_STR, NUMBER_UNIQUE_GROUPS_PER_LEARNING_ACTIVITY_STR)]

    for var, grouper, label in parameter_list: 

        print('-'*100)
        print(f'Number of Unique {var}s per {grouper}:')
        print('-'*100)
        n_unique_var_per_group = print_and_return_unique_col2_per_col1(interactions, 
                                                                       grouper, 
                                                                       var, 
                                                                       label)
        print('\n')
        print('Plots:')
        plot_distribution(n_unique_var_per_group,
                          label,
                          label,
                          log_scale)
        print('*'*100)
        print('*'*100)
        print(' ')
        
    # avg number interactions
    # 1. per group for user
    # 2. per user for group
    parameter_list = [(GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, MEAN_NUMBER_INTERACTIONS_PER_GROUP_FOR_USER_STR, MEDIAN_NUMBER_INTERACTIONS_PER_GROUP_FOR_USER_STR),
                      (USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, MEAN_NUMBER_INTERACTIONS_PER_USER_FOR_GROUP_STR, MEDIAN_NUMBER_INTERACTIONS_PER_USER_FOR_GROUP_STR)]

    for var, grouper, label_mean, label_median in parameter_list:

        print('-'*100)
        print(f'Mean/Median Number of Interactions per {var} for a {grouper}:')
        print('-'*100)
        mean_n_interactions_per_var_for_group,\
        median_n_interactions_per_var_for_group = print_and_return_avg_num_interactions_over_col2_for_col1(interactions, 
                                                                                                           grouper, 
                                                                                                           var, 
                                                                                                           label_mean, 
                                                                                                           label_median)
        print('\n')
        print('Plots:')
        print('Mean:')
        plot_distribution(mean_n_interactions_per_var_for_group,
                          label_mean,
                          label_mean,
                          log_scale)

        print('\n')
        print('Median:')
        plot_distribution(median_n_interactions_per_var_for_group,
                          label_median,
                          label_median,
                          log_scale)
        print('*'*100)
        print('*'*100)
        print(' ')

    # avg number learning activities
    # 1. per group for user
    # 2. per user for group
    parameter_list = [(GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, LEARNING_ACTIVITY_FIELD_NAME_STR, MEAN_NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_FOR_USER_STR, MEDIAN_NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_FOR_USER_STR),
                      (USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_FIELD_NAME_STR, MEAN_NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_USER_FOR_GROUP_STR, MEDIAN_NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_USER_FOR_GROUP_STR)]

    for var, grouper, val, label_mean, label_median in parameter_list:

        print('-'*100)
        print(f'Mean/Median Number of Unique {val}s per {var} for a {grouper}:')
        print('-'*100)
        mean_n_unique_vals_per_var_for_group,\
        median_n_unique_vals_per_var_for_group = print_and_return_avg_num_unique_col3_over_col2_for_col1(interactions, 
                                                                                                         grouper, 
                                                                                                         var, 
                                                                                                         val, 
                                                                                                         label_mean, 
                                                                                                         label_median)
        print('\n')
        print('Plots:')
        print('Mean:')
        plot_distribution(mean_n_unique_vals_per_var_for_group,
                          label_mean,
                          label_mean,
                          log_scale)

        print('\n')
        print('Median:')
        plot_distribution(median_n_unique_vals_per_var_for_group,
                          label_median,
                          label_median,
                          log_scale)
        print('*'*100)
        print('*'*100)
        print(' ')
