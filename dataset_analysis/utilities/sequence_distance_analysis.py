from .standard_import import *
from .constants import *
from .config import *
from .functions import *
from .sequence_distance import *
from .sequence_distance_no_group import *

def calculate_sequence_distances(interactions: pd.DataFrame,
                                 user_field: str,
                                 group_field: str,
                                 learning_activity_field: str,
                                 sequence_id_field: str):
    """For each group calculates the (learning activity-) sequence distances between each possible user combination pair.
    If a interactions dataframe does not contain a grouping field, the sequence distance results will be treated as if
    they belong to a single group(group '0') ranging over the entire length of the interactions dataframe.


    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    user_field : str
        The user field column
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field
    learning_activity_field : str
        The learning_activity field column
    sequence_id_field : str
        The sequence_id field column

    Returns
    -------
    dict
        A dictionary containing for every group(group '0' only if ther is no grouping field in the interactions dataframe)\
        a ndarray of sequence distances between user combinations, a ndarray of lengths of the longer\
        of two compared sequences, a ndarray of users id combinations used for sequence distance calculation,\
        a ndarray of user ids per group, a ndarray of sequence lengths for every user per group, a ndarray of sequence\
        ids and a ndarray of tuples containing the sequence of learning activities the sequence ids map to.
    """
    if group_field:
        print('-'*20)
        print(f'{GROUP_FIELD_NAME_STR}-Field Available:')
        print(f'Calulate {SEQUENCE_STR} Distances for each {GROUP_FIELD_NAME_STR}')
        print('-'*20)
        seq_sim = SeqDist(interactions, 
                          user_field, 
                          group_field, 
                          learning_activity_field, 
                          sequence_id_field)
        seq_distances = seq_sim.get_user_sequence_distances_per_group(distance)
    else:
        print('-'*20)
        print(f'{GROUP_FIELD_NAME_STR}-Field NOT Available:')
        print(f'Calulate {SEQUENCE_STR} Distances')
        print('-'*20)
        seq_sim = SeqDistNoGroup(interactions,
                                 user_field,
                                 learning_activity_field,
                                 sequence_id_field)
        seq_distances = seq_sim.get_user_sequence_distances(distance)

    return seq_distances

def plot_sequence_distances(seq_distances: dict):
    """Plot sequence distances(a boxplot per group) and average sequence distances(single boxplot) per group

    Parameters
    ----------
    seq_distances : dict
        The sequence distance dictionary
    """
    # sequence distances per group
    group_list = []
    distances_list = []
    normalized_distances_list = []
    max_sequence_length_list = []
    for group, subdict in tqdm(seq_distances.items()):

        # extract data from dictionary
        distances = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR])
        max_sequence_len_per_distance = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])
        normalized_distances = distances / max_sequence_len_per_distance 

        group_list.extend([group]*len(distances))
        distances_list.extend(distances)
        normalized_distances_list.extend(normalized_distances)
        max_sequence_length_list.extend(max_sequence_len_per_distance)

    seq_dist_per_group_dict = {GROUP_FIELD_NAME_STR: group_list,
                               LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: distances_list,
                               LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR: normalized_distances_list,
                               LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_sequence_length_list}

    seq_dist_per_group_df = pd.DataFrame(seq_dist_per_group_dict)


    # avg sequence distances per group
    avg_seq_dist_per_group_df = get_avg_seq_dist_per_group_df(seq_distances)

    # plot sequence distance per group
    seq_dist_x_var_list = [LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR, 
                           LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR]
    print('*'*100)
    print('*'*100)
    print(' ')
    for x_var in seq_dist_x_var_list:
        print('-'*100)
        print(f'{x_var} per {GROUP_FIELD_NAME_STR}:')
        print(f'Base: All {USER_FIELD_NAME_STR}-{SEQUENCE_STR} Combinations')
        print('-'*100)
        print('\n')
        print('Plots:')
        g = sns.boxplot(data=seq_dist_per_group_df, 
                        x=x_var, 
                        y=GROUP_FIELD_NAME_STR,
                        showmeans=True, 
                        meanprops=marker_config);

        for patch in g.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.5))

        g = sns.stripplot(data=seq_dist_per_group_df, 
                        x=x_var, 
                        y=GROUP_FIELD_NAME_STR,
                        size=2, 
                        color="red",
                        alpha=0.1)
        g.set(xlabel=x_var);
        plt.show()
        print('*'*100)
        print('*'*100)
        print(' ')

    # plot avg sequence distance per group
    seq_x_var_list = [LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_NAME_STR,
                      LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_NAME_STR,
                      LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR,
                      LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR]
    for var in seq_x_var_list:
        print('-'*100)
        print(f'{var} per {GROUP_FIELD_NAME_STR}:')
        print('-'*100)
        print('\n')
        print('Plots:')
        # sequence distance distribution
        plot_distribution(avg_seq_dist_per_group_df, 
                          var,
                          var,
                          False)

        # avg sequence length vs avg sequence distance
        g = sns.regplot(data=avg_seq_dist_per_group_df, 
                        x=var, 
                        y=LEARNING_ACTIVITY_MEAN_SEQUENCE_LENGTH_NAME_STR)
        g.set(xlabel=f'{var} per {GROUP_FIELD_NAME_STR}', 
              ylabel=f'{LEARNING_ACTIVITY_MEAN_SEQUENCE_LENGTH_NAME_STR} per {GROUP_FIELD_NAME_STR}');
        plt.show()
        g = sns.regplot(data=avg_seq_dist_per_group_df, 
                        x=var, 
                        y=LEARNING_ACTIVITY_MEDIAN_SEQUENCE_LENGTH_NAME_STR)
        g.set(xlabel=f'{var} per {GROUP_FIELD_NAME_STR}', 
              ylabel=f'{LEARNING_ACTIVITY_MEDIAN_SEQUENCE_LENGTH_NAME_STR} per {GROUP_FIELD_NAME_STR}');
        plt.show()
        print('*'*100)
        print('*'*100)
        print(' ')

def plot_sequence_distance_matrix_by_group(seq_distances: dict,
                                           normalize_distance: dict,
                                           group_str: dict):
    """Plot the siquence distance matrix for a specific group.

    Parameters
    ----------
    seq_distances : dict
        The sequence distance dictionary returned by the calculate_sequence_distances function
    normalize_distance : bool
        A boolean indicating whether the sequence distances are being normalized between 0 and 1
    group_str : str
        A string indicating for which group the evaluation metric distribution per cluster will be displayed
    """
    distances = np.array(seq_distances[group_str][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR])
    if normalize_distance:
        max_sequence_len_per_distance = np.array(seq_distances[group_str][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])
        distances = distances / max_sequence_len_per_distance 

    square_matrix = squareform(distances)

    print('*'*100)
    print('*'*100)
    print(' ')
    print('-'*100)
    if normalize_distance:
        print(f'Normalized {SEQUENCE_STR} Distance Matrix for {GROUP_FIELD_NAME_STR} {group_str}:')
    else:
        print(f'{SEQUENCE_STR} Distance Matrix for {GROUP_FIELD_NAME_STR} {group_str}:')
    print('-'*100)
    g = sns.heatmap(square_matrix, 
                    cbar=True, 
                    square=True)
    g.set(xlabel=SEQUENCE_STR, 
          ylabel=SEQUENCE_STR)
    plt.show()

def plot_sequence_distance_matrix_all_group(seq_distances: dict,
                                            normalize_distance: bool,
                                            height: int):
    """Plot the siquence distance matrix for all groups.

    Parameters
    ----------
    seq_distances : dict
        The sequence distance dictionary returned by the calculate_sequence_distances function
    normalize_distance : bool
        A boolean indicating whether the sequence distances are being normalized between 0 and 1
    height : int
        The height of the subplots
    """
    square_matrices_per_group_df = pd.DataFrame()
    for group, subdict in seq_distances.items():

        distances = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR])
        if normalize_distance:
            max_sequence_len_per_distance = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])
            distances = distances / max_sequence_len_per_distance 

        square_matrix = squareform(distances)
        
        square_matrix_df = pd.DataFrame(square_matrix) 
        square_matrix_df = pd.melt(square_matrix_df, 
                                   ignore_index=False, 
                                   var_name='row', 
                                   value_name='value')\
                             .reset_index()\
                             .rename(columns={'index': 'column'})
        square_matrix_df[GROUP_FIELD_NAME_STR] = group
        square_matrices_per_group_df = pd.concat([square_matrices_per_group_df, square_matrix_df])

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    print('*'*100)
    print('*'*100)
    print(' ')
    print('-'*100)
    if normalize_distance:
        print(f'Normalized {SEQUENCE_STR} Distance Matrix per {GROUP_FIELD_NAME_STR}:')
    else:
        print(f'{SEQUENCE_STR} Distance Matrix per {GROUP_FIELD_NAME_STR}:')
    print('-'*100)
    g = sns.FacetGrid(square_matrices_per_group_df, 
                      col=GROUP_FIELD_NAME_STR,
                      col_wrap=6, 
                      sharex=False,
                      sharey=False,
                      height=height, 
                      aspect= 1)
    if normalize_distance:
        g.map_dataframe(draw_heatmap, 
                        'column', 
                        'row', 
                        'value',
                        vmin=0,
                        vmax=1,
                        cbar=True, 
                        square = True)
    else:
        g.map_dataframe(draw_heatmap, 
                        'column', 
                        'row', 
                        'value', 
                        cbar=True, 
                        square = True)
    g.set(xlabel=SEQUENCE_STR, 
          ylabel=SEQUENCE_STR)
    # get figure background color
    facecolor=plt.gcf().get_facecolor()
    for ax in g.axes.flat:
        # set aspect of all axis
        ax.set_aspect('equal','box')
        # set background color of axis instance
        ax.set_facecolor(facecolor)
    plt.show()