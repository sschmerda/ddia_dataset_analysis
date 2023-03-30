from .standard_import import *
from .constants import *
from .config import *

def get_nas_in_data(interactions: pd.DataFrame):
    """Calculate the number of NAs in a dataframe

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactins dataframe

    Returns
    -------
    pd.Series
        A pd.Series containing the percentages of NAs in the input dataframe
    """    
    pct_na = interactions.isna().sum() / len(interactions) * 100
    pct_na = pct_na.apply(str) + ' %'

    return pct_na

def map_new_to_old_values(interactions: pd.DataFrame,
                          group_field: str,
                          user_field: str,
                          learning_activity_field: str):
    """Map new column values to old ones for the group, user and learning_activity fields(categorical variables). New values range from [0] to [#unique values in the respective fields -1].
    New values are of type string.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    learning_activity_field : str
        The learning_activity field column

    Returns
    -------
    tuple
        A tuple containing the input interactions dataframe with mapped values in the specified categorical varible fields and a remapping dataframe which can be used in remapping the new to the original values.
    """
    field_dict = {group_field: GROUP_FIELD_NAME_STR, user_field: USER_FIELD_NAME_STR, learning_activity_field: LEARNING_ACTIVITY_FIELD_NAME_STR}
    field_dict = {k:v for k,v in field_dict.items() if k} 
    mapping_dict_all_fields = {} 

    for field in field_dict.keys():
        values = enumerate(interactions[field].dropna().unique())
        mapping_dict = {v: str(n) for n,v in values}
        interactions[field] = interactions[field].map(mapping_dict)
        mapping_dict_all_fields[field] = mapping_dict

    for k,v in field_dict.items():
        mapping_dict_all_fields[v] = mapping_dict_all_fields.pop(k)

    old_new_mapping_dict = {}
    for k, v in mapping_dict_all_fields.items():
        old_new_mapping_dict[f'{k} {NEW_VALUE_STR}'] = pd.Series(v.values()) 
        old_new_mapping_dict[f'{k} {ORIGINAL_VALUE_STR}'] = pd.Series(v.keys()) 

    value_mapping_df = pd.DataFrame(old_new_mapping_dict)

    return interactions, value_mapping_df

def drop_na_by_fields(interactions: pd.DataFrame, field_list=[]):
    """Drops rows of the interactins dataframe that havn NAs in any of the fields specified in field_list

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    field_list : list, optional
        A list of fields in the interactions dataframe which are used for detecting NAs, by default []

    Returns
    -------
    pd.DataFrame
        The dataframe with rows removed which habe NAs in any of the fields specified in field_list 
    """    
    field_list = [i for i in field_list if i]
    input_len = interactions.shape[0]
    interactions = interactions.dropna(subset=field_list)
    output_len = interactions.shape[0]
    n_removed = input_len - output_len
    pct_removed = int(round((input_len - output_len) / input_len * 100))

    print(f'Input length: {input_len}')
    print(f'Outpunt length: {output_len}')
    print(f'Number of rows removed: {n_removed}')
    print(f'Percentage of rows removed: {pct_removed}%')

    return interactions

def drop_learning_activity_sequence_if_contains_na_in_field(interactions: pd.DataFrame, 
                                                            group_field: str, 
                                                            user_field: str, 
                                                            field_list=[], 
                                                            field_value_tuple_filter_list=[]):
    """Drops sequences groupwise from a dataframe that contain NAs in fields specified in field_list. Certain values in fields ((field, value) tuple in field_value_tuple_filter_list) can be ommited such that the sequences they are contained in will not be dropped. 

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    field_list : list, optional
        A list of fields in the interactions dataframe which are used for detecting NAs, by default []
    field_value_tuple_filter_list : list, optional
        A list of field-value tuples which will not be used fo detecting NAs, by default []

    Returns
    -------
    pd.DataFrame
        The interactions dataframe without sequences which contain NAs in fields specified in field_list
    list
        A list containing row indices of entries that have NAs in either group or user field. These entries cannot
        be exactly matched to a particular sequence
    """    
    group_list = [group_field, user_field]
    group_list = [i for i in group_list if i]
    total_number_of_sequences = interactions.groupby(group_list).ngroups
    input_len = interactions.shape[0]

    interactions_to_filter = interactions
    for field, value in field_value_tuple_filter_list:
        interactions_to_filter = interactions_to_filter.loc[~(interactions_to_filter[field]==value), :]

    index_user_na = list(interactions_to_filter[user_field][interactions_to_filter[user_field].isna()].index)
    index_group_na = []
    if group_field:
        index_group_na = list(interactions_to_filter[group_field][interactions_to_filter[group_field].isna()].index)

    na_indices_list = list(set(index_user_na + index_group_na))

    n_user_na = len(index_user_na)
    n_group_na = len(index_group_na)
    
    if n_user_na:
        if n_user_na == 1:
            print(f'Warning: There is {n_user_na} NA in the {USER_FIELD_NAME_STR} field!')
        else:
            print(f'Warning: There are {n_user_na} NAs in the {USER_FIELD_NAME_STR} field!')

    if n_group_na:
        if n_group_na == 1:
            print(f'Warning: There is {n_group_na} NA in the {GROUP_FIELD_NAME_STR} field!')
        else:
            print(f'Warning: There are {n_group_na} NAs in the {GROUP_FIELD_NAME_STR} field!')
    print('='*50)
    print('\n')

    field_list = [i for i in field_list if i not in group_list]
    seq_filter = interactions_to_filter.groupby(group_list)[field_list]\
                              .agg(lambda x: x.isna().any())\
                              .reset_index()
    field_any_na = seq_filter[field_list].any(axis=1)
    seq_filter = seq_filter[field_any_na]
    seq_filter = seq_filter[group_list].apply(tuple, axis=1)
    interactions_sequences_tuple = interactions[group_list].apply(tuple, axis=1)\
                                                           .isin(seq_filter)
    interactions = interactions.loc[~interactions_sequences_tuple]

    output_len = interactions.shape[0]
    n_removed = input_len - output_len
    pct_removed = float(round((input_len - output_len) / input_len * 100, 4))
    n_sequences_removed = len(seq_filter)
    pct_sequences_removed = float(round(n_sequences_removed / total_number_of_sequences * 100, 4))


    print(f'Input length: {input_len}')
    print(f'Outpunt length: {output_len}')
    print(f'Number of rows removed: {n_removed}')
    print(f'Percentage of rows removed: {pct_removed}%')
    print('-' * 50)
    print(f'Input number of sequences: {total_number_of_sequences}')
    print(f'Output number of sequences: {total_number_of_sequences - n_sequences_removed}')
    print(f'Number of sequences removed: {n_sequences_removed}')
    print(f'Percentage of sequences removed: {pct_sequences_removed}%')

    return interactions, na_indices_list

def sort_by_timestamp(interactions: pd.DataFrame, timestamp_field: str, higher_level_sort_list=[]):
    """Sorts the input dataframe by fields in higher_level_sort_list and the timestamp field

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    timestamp_field : str
        The timestamp field column
    higher_level_sort_list : list, optional
        A list containing fields of the interactions dataframe used for sorting before timestamp is taken in consideration, by default []

    Returns
    -------
    pd.DataFrame
        The sorted interactions dataframe
    """
    higher_level_sort_list.append(timestamp_field)
    interactions = interactions.sort_values(by=higher_level_sort_list)
    
    return interactions

def keep_last_repeated_learning_activities(interactions: pd.DataFrame, group_field: str, user_field: str, learning_activity_field: str, timestamp_field: str):
    """Filters out all but the last of repeated learnig activities in the interactions dataframe per user-group sequence

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
    user_field : str
        The user field column
    learning_activity_field : str
        The learning_activity field column
    timestamp_field : str
        The timestamp field column

    Returns
    -------
    pd.DataFrame
        The filtered dataframe
    """
    interactions = interactions.reset_index(drop=True)
    interactions = interactions.sort_values(by=[group_field, user_field, timestamp_field])
    initial_len = interactions.shape[0]

    keep_index_list = []

    for _, df in tqdm(interactions.groupby([group_field, user_field])):

        la_prev = None
        
        for index, learning_activity in df[learning_activity_field].iloc[::-1].items():

            if learning_activity == la_prev:
                continue
            else:
                keep_index_list.append(index)
            
            la_prev = learning_activity
        
    interactions = interactions.loc[keep_index_list]
    interactions = interactions.sort_values(by=[group_field, user_field, timestamp_field])

    final_len = interactions.shape[0]
    n_removed_interactions = initial_len - final_len
    n_removed_interactions_pct = (initial_len - final_len) / initial_len * 100

    print('= Repeated Interactions Removal =')
    print(f'Initial number of interactions: {initial_len}')
    print(f'Final number of interactions: {final_len}')
    print(f'Removed number of interactions: {n_removed_interactions}')
    print(f'Removed percentage of interactions: {n_removed_interactions_pct}%')

    return interactions

def keep_last_repeated_learning_activities_no_group(interactions: pd.DataFrame, user_field: str, learning_activity_field: str, timestamp_field: str):
    """Filters out all but the last of repeated learnig activities in the interactions dataframe per user sequence

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    user_field : str
        The user field column
    learning_activity_field : str
        The learning_activity field column
    timestamp_field : str
        The timestamp field column

    Returns
    -------
    pd.DataFrame
        The filtered dataframe
    """
    interactions = interactions.reset_index(drop=True)
    interactions = interactions.sort_values(by=[user_field, timestamp_field])
    initial_len = interactions.shape[0]

    keep_index_list = []

    for _, df in tqdm(interactions.groupby([user_field])):

        la_prev = None
        
        for index, learning_activity in df[learning_activity_field].iloc[::-1].items():

            if learning_activity == la_prev:
                continue
            else:
                keep_index_list.append(index)
            
            la_prev = learning_activity
        
    interactions = interactions.loc[keep_index_list]
    interactions = interactions.sort_values(by=[user_field, timestamp_field])
    final_len = interactions.shape[0]

    n_removed_interactions = initial_len - final_len
    n_removed_interactions_pct = (initial_len - final_len) / initial_len * 100

    print('= Repeated Interactions Removal =')
    print(f'Initial number of interactions: {initial_len}')
    print(f'Final number of interactions: {final_len}')
    print(f'Removed number of interactions: {n_removed_interactions}')
    print(f'Removed percentage of interactions: {n_removed_interactions_pct}%')

    return interactions

def add_sequence_id_field(interactions: pd.DataFrame,
                          group_field: str,
                          user_field: str,
                          learning_activity_field: str):
    """Adds a sequence_id field to the interactions dataframe. An unique id is mapped to each unique sequence of learning
    activities, indicatig to which sequence a (group,user,learning_activity) entry belongs to.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    learning_activity_field : str
        The learning_activity field column

    Returns
    -------
    pd.DatafRame
        The interactions dataframe with added sequence_id field
    """     
    grouping_list = [group_field, user_field]
    grouping_list = [i for i in grouping_list if i]

    unique_sequences = interactions.groupby(grouping_list)[learning_activity_field].agg(tuple).rename(SEQUENCE_ID_FIELD_NAME_STR)

    sequence_sequence_id_mapping_dict = {seq:str(seq_id) for seq_id, seq in enumerate(unique_sequences.unique())}
    unique_sequences = unique_sequences.apply(lambda x: sequence_sequence_id_mapping_dict[x]).reset_index()

    interactions = interactions.merge(unique_sequences, how='inner', on=grouping_list)

    sequence_id_column_index_positions = list(interactions.columns).index(learning_activity_field) + 1
    sequenc_id_column = interactions.pop(SEQUENCE_ID_FIELD_NAME_STR)
    interactions.insert(sequence_id_column_index_positions, SEQUENCE_ID_FIELD_NAME_STR, sequenc_id_column)
    
    return interactions

def rename_fields(interactions: pd.DataFrame,
                  group_field: str,
                  user_field: str,
                  learning_activity_field: str,
                  timestamp_field: str):
    """Renames the fields of the interactions dataframe according to the preset field name strings and generates a remapping dataframe containing a mapping between new and old field names.
       If a field argument is set to None(in the case it is missing in the interactions dataframe), a column with the corresponding name containing None entries will be created.  

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    learning_activity_field : str
        The learning activity field column
    timestamp_field : str
        The timestamp field column

    Returns
    -------
    tuple
        The interactions dataframe with renamed fields and a remapping dataframe
    """
    interactions_column_names = interactions.columns
    old_field_names = [timestamp_field, user_field, group_field, learning_activity_field]
    new_field_names = [TIMESTAMP_FIELD_NAME_STR, USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_FIELD_NAME_STR]
    evaluation_field_names = [i for i in chain(EVALUATION_LEARNING_ACTIVITY_FIELD_LIST,
                                               EVALUATION_GROUP_FIELD_LIST,
                                               EVALUATION_GROUP_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_LIST,
                                               EVALUATION_COURSE_FIELD_LIST,
                                               EVALUATION_COURSE_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_LIST,
                                               EVALUATION_COURSE_ALL_GROUPS_MEAN_FIELD_LIST)]
    old_new_mapping = list(zip(old_field_names, new_field_names))
    rename_dict = {k:v for k,v in old_new_mapping if k in interactions_column_names}

    # rename fields
    interactions = interactions.rename(columns=rename_dict)

    # add non-available fields containing np.nan values 
    for old, new in old_new_mapping:
        if not old:
            interactions[new] = np.nan

    # keep necessary columns
    interactions = interactions[new_field_names + evaluation_field_names]
    interactions = interactions.reset_index(drop=True)


    # mapping dataframe
    fields_mapping_df = pd.DataFrame({NEW_FIELDNAME_FIELD_NAME_STR: new_field_names, 
                                      ORIGINAL_FIELDNAME_FIELD_NAME_STR: old_field_names})

    return interactions, fields_mapping_df

def typecast_fields(interactions: pd.DataFrame,
                    timestamp_field: str,
                    group_field: str,
                    user_field: str,
                    learning_activity_field: str,
                    **kwargs):
    """Change the datatype of columns to the appropriate one.
    (categorical variables into str type, timestamp variable into datetime)

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    timestamp_field : str
        The timestamp field column
        Can be ommited if the interactions dataframe does not have a timestamp field
    group_field : str
        The group field column
        Can be ommited if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
        Can be ommited if the interactions dataframe does not have a user_field
    learning_activity_field : str
        The learning_activity field column
        Can be ommited if the interactions dataframe does not have a learning_activity_field

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with typecast fields
    """
    # typecast catergorical variable as strings
    cat_field_list = [group_field, user_field, learning_activity_field]
    cat_typecast_dict = {i: 'str' for i in cat_field_list if i}
    interactions = interactions.astype(cat_typecast_dict)

    # typecast timestamp as datetime field
    if timestamp_field:
        interactions[timestamp_field] = pd.to_datetime(interactions[timestamp_field], errors='coerce')

    return interactions

def save_interaction_and_mapping_df(interactions: pd.DataFrame,
                                    field_mapping_dataframe: pd.DataFrame,
                                    value_mapping_dataframe: pd.DataFrame,
                                    path_to_dataset_folder: str,
                                    dataset_name: str):
    """Saves the interactions and fields_mapping dataframes

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    fields_mapping_dataframe : pd.DataFrame
        The fields mapping dataframe
    path_to_dataset_folder : str
        The directory in which the the datasets are being saved
    dataset_name : str
        The name used for saving the interactions dataset
    """
    # interactions datafram
    interactions = interactions.reset_index(drop=True)
    interactions.to_csv(path_to_dataset_folder + dataset_name + '.csv', index=False)

    # field mapping dataframe
    field_mapping_dataframe.to_csv(path_to_dataset_folder + dataset_name + FIELD_MAPPING_DATAFRAME_NAME_STR + '.csv', index=False)

    # value mapping dataframe
    value_mapping_dataframe.to_csv(path_to_dataset_folder + dataset_name + VALUE_MAPPING_DATAFRAME_NAME_STR + '.csv', index=False)

def pickle_write(object_to_pickle,
                 path_within_pickle_directory: str,
                 filename: str):
    """Serializes a python object and stores it in the specified location

    Parameters
    ----------
    object_to_pickle : 
        A python object to be serialized
    path_within_pickle_directory : str
        A path to a subfolder of the pickle directory indicating where the serialized object is being saved 
    filename : str
        The name given to the serialized python object
    """
    with open(PATH_TO_PICKLED_OBJECTS_FOLDER + path_within_pickle_directory + filename + '.pickle', 'wb') as f:
        pickle.dump(object_to_pickle, f, pickle.HIGHEST_PROTOCOL)

def pickle_read(path_within_pickle_directory,
                filename):
    """Reads and returns a serialized python object located in the specified directory 

    Parameters
    ----------
    path_within_pickle_directory : str
        A path to a subfolder of the pickle directory indicating where the serialized object is being located 
    filename : str
        The name given to the serialized python object

    Returns
    -------
        The deserialized python object
    """        
    with open(PATH_TO_PICKLED_OBJECTS_FOLDER + path_within_pickle_directory + filename + '.pickle' , 'rb') as f:
        pickled_object = pickle.load(f)

    return pickled_object

def print_summary_stats(interactions: pd.DataFrame, 
                        user_field: str, 
                        group_field: str, 
                        learning_activity_field: str):
    """Print summary statistics of the interactions dataframe

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
        The learning_activity column
    """    
    print(f'Number of interactions: {interactions.shape[0]}')
    if group_field:
        print(f'Number of unique {group_field}s: {interactions[group_field].nunique()}')
    print(f'Number of unique {user_field}s: {interactions[user_field].nunique()}')
    print(f'Number of unique {learning_activity_field}s: {interactions[learning_activity_field].nunique()}')

def calculate_sparsity(index: pd.Series, column: pd.Series):
    """Calculate sparsity for a matrix 

    Parameters
    ----------
    index : pd.Series
        A series whose values represent the rows of the sparsity matrix.
    column : pd.Series
        A series whose values represent the columns of the sparsity matrix.

    Returns
    -------
    float
        The calculated sparsity in percentages.
    """    
    (index_values, column_values), count_matrix = crosstab(index, column)
    number_non_zero_elements = (count_matrix != 0).sum()
    number_elements = count_matrix.size
    sparsity = 100 - (number_non_zero_elements / number_elements * 100)

    return sparsity

def get_avg_seq_dist_per_group_df(seq_dist_per_group_dict: dict):
    """Generate a DataFrame containing avg user sequence distances per grouping variable.

    Parameters
    ----------
    seq_dist_per_group_dict : dict
        A sequence distance per group dict calculated via SeqSim

    Returns
    -------
    pd.DataFrame
        A DataFrame containing avg user sequence distances per grouping variable.
    """    
    group_names = list(seq_dist_per_group_dict.keys())
    mean_seq_distances_per_group =  [np.mean(data[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]) for data in seq_dist_per_group_dict.values()]
    median_seq_distances_per_group = [np.median(data[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]) for data in seq_dist_per_group_dict.values()]
    mean_norm_seq_distances_per_group =  [np.mean(np.array(data[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]) / np.array(data[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])) for data in seq_dist_per_group_dict.values()]
    median_norm_seq_distances_per_group =  [np.median(np.array(data[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]) / np.array(data[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])) for data in seq_dist_per_group_dict.values()]
    mean_user_seq_length_per_group = [np.mean(data[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]) for data in seq_dist_per_group_dict.values()]
    median_user_seq_length_per_group = [np.median(data[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]) for data in seq_dist_per_group_dict.values()]
    mean_max_seq_length_per_group = [np.mean(data[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]) for data in seq_dist_per_group_dict.values()]
    median_max_seq_length_per_group = [np.median(data[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]) for data in seq_dist_per_group_dict.values()]

    avg_seq_dist_per_group_df = pd.DataFrame({GROUP_FIELD_NAME_STR: group_names,
                                              LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_NAME_STR: mean_seq_distances_per_group,
                                              LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_NAME_STR: median_seq_distances_per_group,
                                              LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR: mean_norm_seq_distances_per_group,
                                              LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR: median_norm_seq_distances_per_group,
                                              LEARNING_ACTIVITY_MEAN_SEQUENCE_LENGTH_NAME_STR: mean_user_seq_length_per_group,
                                              LEARNING_ACTIVITY_MEDIAN_SEQUENCE_LENGTH_NAME_STR: median_user_seq_length_per_group,
                                              LEARNING_ACTIVITY_MEAN_MAX_SEQUENCE_LENGTH_NAME_STR: mean_max_seq_length_per_group,
                                              LEARNING_ACTIVITY_MEDIAN_MAX_SEQUENCE_LENGTH_NAME_STR: median_max_seq_length_per_group})
    
    return avg_seq_dist_per_group_df


def get_seq_dist_df(seq_dist_dict: dict):
    """Generate a DataFrame containing sequence distances per user combination.

    Parameters
    ----------
    seq_dist_dict : dict
        A sequence distance per user dict calculated via SeqSim

    Returns
    -------
    pd.DataFrame
        A DataFrame containing sequence distances per user combination.
    """    
    user_names_combinations = list(seq_dist_dict['user_combinations'])
    seq_distances_per_user_name_combination = np.array(seq_dist_dict['distances'])
    norm_seq_distances_per_user_name_combination = np.array(seq_dist_dict['distances']) / np.array(seq_dist_dict['max_seq_lengths'])
    max_user_seq_length_per_combination = seq_dist_dict['max_seq_lengths'] 
    mean_user_seq_length_per_combination = seq_dist_dict['mean_seq_lengths'] 
    mean_user_seq_length = np.mean(seq_dist_dict['user_sequence_length'])
    median_user_seq_length = np.median(seq_dist_dict['user_sequence_length'])

    seq_distance_df = pd.DataFrame({'user_name_combination': user_names_combinations,
                                                'distance': seq_distances_per_user_name_combination,
                                                'normalized_distance': norm_seq_distances_per_user_name_combination,
                                                'max_user_sequence_length_per_combination': max_user_seq_length_per_combination,
                                                'mean_user_sequence_length_per_combination': mean_user_seq_length_per_combination,
                                                'mean_user_sequence_length': mean_user_seq_length,
                                                'median_user_sequence_length': median_user_seq_length})
    
    return seq_distance_df

def return_learning_activity_sequence_stats_over_user_per_group(interactions: pd.DataFrame,
                                                                dataset_name: str,
                                                                group_field: str, 
                                                                user_field: str, 
                                                                learning_activity_field: str, 
                                                                timestamp_field: str):
    """Return a dataframe which contains statistics (frequencies and lengths) of unique learning_activity sequences over user entities grouped by group entities

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    dataset_name : str
        The name of the dataset.
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    learning_activity_field : str
        The learning_activity field column
    timestamp_field : str
        The timestamp field column used for sequence sorting

    Returns
    -------
    pd.DataFrame
        A dataframe containing statistics (frequencies and lengths) of unique learning_activity sequences over user entities grouped by group entities
    """

    # helper functions
    def calc_n_repeated(seq_tuple):
        length = len(seq_tuple) 
        n_uniuqe_elements = len(set(seq_tuple))
        number_repeated_elements = length - n_uniuqe_elements

        return number_repeated_elements

    def calc_pct_repeated(seq_tuple):
        length = len(seq_tuple) 
        n_uniuqe_elements = len(set(seq_tuple))
        number_repeated_elements = length - n_uniuqe_elements
        percentage_repeated_elements = number_repeated_elements / length * 100

        return percentage_repeated_elements

    if not group_field:
        interactions[GROUP_FIELD_NAME_STR] = '0'
        group_field = GROUP_FIELD_NAME_STR
    
    group_data = []
    group_code_data = []
    seq_count_per_group_data = []
    unique_seq_count_per_group_data = []
    unique_seq_data = []
    seq_freq_data = []
    seq_freq_pct_data = []
    seq_len_data = []
    n_repeated_learning_activity_in_seq_data = []
    pct_repeated_learning_activity_in_seq_data = []
    n_unique_learning_activities_per_sequence_data = []
    pct_unique_learning_activties_per_group_in_sequence_data = []

    for n, (group, df_1) in enumerate(interactions.groupby(group_field)):
        learning_activity_seq_frequency_over_user_dict = defaultdict(int) 
        df_1 = df_1.sort_values(by=[user_field, timestamp_field])
        for user, df_2 in df_1.groupby(user_field):
            seq_col_3 = df_2[learning_activity_field].to_list()
            seq_col_3 = tuple(seq_col_3)

            learning_activity_seq_frequency_over_user_dict[seq_col_3] += 1
        
        seq_count = sum(list(learning_activity_seq_frequency_over_user_dict.values()))
        unique_seq_count = len(list(learning_activity_seq_frequency_over_user_dict.keys()))

        unique_seq_list = list(learning_activity_seq_frequency_over_user_dict.keys())
        seq_freq_list = list(learning_activity_seq_frequency_over_user_dict.values())
        seq_freq_pct_list = [freq/seq_count*100 for freq in learning_activity_seq_frequency_over_user_dict.values()]
        seq_len_list = list(map(len, learning_activity_seq_frequency_over_user_dict.keys()))
        n_repeated_learning_activity_in_seq_list = list(map(calc_n_repeated, unique_seq_list)) 
        pct_repeated_learning_activity_in_seq_list = list(map(calc_pct_repeated, unique_seq_list)) 
        n_unique_learning_activies_per_sequence_list = [len(set(tup)) for tup in unique_seq_list]
        n_unique_learning_activities_in_group = len({value for tup in unique_seq_list for value in tup})
        pct_unique_learning_activties_per_group_in_sequence_list = [i/n_unique_learning_activities_in_group*100 for i in n_unique_learning_activies_per_sequence_list]


        group_data.extend([str(group)] * len(unique_seq_list))
        group_code_data.extend([n] * len(unique_seq_list))
        seq_count_per_group_data.extend([seq_count] * len(unique_seq_list))
        unique_seq_count_per_group_data.extend([unique_seq_count] * len(unique_seq_list))
        unique_seq_data.extend(unique_seq_list)
        seq_freq_data.extend(seq_freq_list)
        seq_freq_pct_data.extend(seq_freq_pct_list)
        seq_len_data.extend(seq_len_list)
        n_repeated_learning_activity_in_seq_data.extend(n_repeated_learning_activity_in_seq_list)
        pct_repeated_learning_activity_in_seq_data.extend(pct_repeated_learning_activity_in_seq_list)
        n_unique_learning_activities_per_sequence_data.extend(n_unique_learning_activies_per_sequence_list)
        pct_unique_learning_activties_per_group_in_sequence_data.extend(pct_unique_learning_activties_per_group_in_sequence_list)



    seq_stats_dict = {DATASET_NAME_FIELD_NAME_STR: dataset_name,
                      GROUP_FIELD_NAME_STR: group_data,
                      GROUP_CODE_FIELD_NAME_STR: group_code_data,
                      LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR: seq_count_per_group_data,
                      UNIQUE_LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR: unique_seq_count_per_group_data,
                      LEARNING_ACTIVITY_SEQUENCE_NAME_STR: unique_seq_data,
                      LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR: seq_freq_data,
                      LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR: seq_freq_pct_data,
                      LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: seq_len_data,
                      LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_NAME_STR: n_repeated_learning_activity_in_seq_data,
                      LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR: pct_repeated_learning_activity_in_seq_data,
                      LEARNING_ACTIVITY_SEQUENCE_NUMBER_UNIQUE_LEARNING_ACTIVITIES_NAME_STR: n_unique_learning_activities_per_sequence_data,
                      LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR: pct_unique_learning_activties_per_group_in_sequence_data} 

    learning_activity_seq_stats_over_user_per_group = pd.DataFrame(seq_stats_dict)
    learning_activity_seq_stats_over_user_per_group = learning_activity_seq_stats_over_user_per_group.sort_values(by=[GROUP_CODE_FIELD_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR], ascending=[True, False])

    return learning_activity_seq_stats_over_user_per_group

def plot_sequence_stats_per_group(learning_activity_sequence_stats_per_group: pd.DataFrame,
                                  group_field: str):
    """Plot unique sequence statistics per grouping variable

    Parameters
    ----------
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A learning activity sequence statistics per group dataframe created by return_col3_sequence_stats_over_col2_per_col1 
    group_field_name_str : str
        The grouping field column
    """
    
    sequence_frequency_stats_per_group = learning_activity_sequence_stats_per_group\
                                         .groupby(group_field)[LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR]\
                                         .agg([min, max, np.median])\
                                         .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                          'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                          'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR})\
                                         .reset_index()
    
    sequence_frequency_stats_per_group_long = pd.melt(sequence_frequency_stats_per_group[[group_field, 
                                                                                          LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                          LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                          LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                      id_vars=group_field,
                                                      var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                      value_name=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR)

    sequence_length_stats_per_group = learning_activity_sequence_stats_per_group\
                                         .groupby(group_field)[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]\
                                         .agg([min, max, np.median])\
                                         .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                          'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                          'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR})\
                                         .reset_index()
    
    sequence_length_stats_per_group_long = pd.melt(sequence_length_stats_per_group[[group_field, 
                                                                                    LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                    LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                    LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                      id_vars=group_field,
                                                      var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                      value_name=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR)

    repeated_learning_activities_stats_per_group = learning_activity_sequence_stats_per_group\
                                                   .groupby(group_field)[LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR]\
                                                   .agg([min, max, np.median])\
                                                   .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                   'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                   'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR})\
                                                   .reset_index()
    
    repeated_learning_activities_stats_per_group_long = pd.melt(repeated_learning_activities_stats_per_group[[group_field, 
                                                                                                    LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                                    LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                                    LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                                id_vars=group_field,
                                                                var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                                value_name=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR)

    pct_learning_activities_per_group_stats_per_group = learning_activity_sequence_stats_per_group\
                                                        .groupby(group_field)[LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR]\
                                                        .agg([min, max, np.median])\
                                                        .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                        'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                        'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR})\
                                                        .reset_index()
    
    pct_learning_activities_per_group_stats_per_group_long = pd.melt(pct_learning_activities_per_group_stats_per_group[[group_field, 
                                                                                                    LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                                    LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                                    LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                                     id_vars=group_field,
                                                                     var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                                     value_name=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR)

    # all groups in one figure - unique seq count vs seq count
    count_df = learning_activity_sequence_stats_per_group.groupby(group_field).head(1)
    ylim = count_df[LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR].max()
    g = sns.scatterplot(data=count_df,
                        x=LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR, 
                        y=UNIQUE_LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR, 
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR, 
          ylabel=UNIQUE_LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR,
          ylim=(-5,ylim))
    g.set_title(LEARNING_ACTIVITY_UNIQUE_VS_TOTAL_NUMBER_OF_SEQUENCES_PER_GROUP_TITLE_NAME_STR, 
                fontsize=20)
    g.axline(xy1=(0,0), slope=1, color='r', linewidth=3);
    plt.show(g)

    # all groups in one figure - seq freq stats
    g = sns.scatterplot(data=sequence_frequency_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_FREQUENCY_PCT_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_FREQUENCY_PCT_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=sequence_frequency_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                      hue=group_field)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # all groups in one figure - seq len stats
    g = sns.scatterplot(data=sequence_length_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_LENGTH_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_LENGTH_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=sequence_length_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                      hue=group_field)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_LENGTH_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # all groups in one figure - repeated learning activities stats
    g = sns.scatterplot(data=repeated_learning_activities_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=repeated_learning_activities_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                      hue=group_field)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # all groups in one figure - pct of unique learning activities per group in seq stats
    g = sns.scatterplot(data=pct_learning_activities_per_group_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=pct_learning_activities_per_group_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                      hue=group_field)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # per group figures
    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=group_field, 
                      col_wrap=6, 
                      sharex=False, 
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=(-10,110), 
          ylim=(0))
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=group_field, 
                      col_wrap=6, 
                      sharex=False, 
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=(0), 
          ylim=(0))
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=group_field, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR, 
          xlim=(-10,110), 
          ylim=(-10, 110))
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=group_field, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR, 
          xlim=(0), 
          ylim=(-10, 110))
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=group_field, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR, 
          xlim=(-10,110), 
          ylim=(-10, 110))
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=group_field, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR, 
          xlim=(0), 
          ylim=(-10, 110))
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

def plot_sequence_stats(learning_activity_sequence_stats_per_group: pd.DataFrame):
    """Plot unique sequence statistics

    Parameters
    ----------
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A learning activity sequence statistics per group dataframe created by return_col3_sequence_stats_over_col2_per_col1 
    """
    
    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=(-5,105), 
          ylim=(0))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=(-5), 
          ylim=(0))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
          xlim=(-5,105), 
          ylim=(-5))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
          xlim=(-5), 
          ylim=(-5))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
          xlim=(-5,105), 
          ylim=(-5))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
          xlim=(-5), 
          ylim=(-5))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)


def plot_distribution(data: pd.DataFrame,
                      x_var: str,
                      label: str,
                      log_scale: bool,
                      pointsize=5):
    """Plot the distribution of a variable via boxplot-stripplot, kernel-density and histogram.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing the variable for which the distribution plots are being plotted
    x_var : str
        The dataframe field name of the variable for which the distribution plots are being plotted
    label : str
        The axis label of the variable for which the distribution plots are being plotted
    log_scale : bool
        A boolean indicating whether logarithmized axis should be applied
    pointsize : float
        The pointsize of the stripplot
    """
    # box and stripplot
    g = sns.boxplot(data=data, 
                    x=x_var, 
                    showmeans=True, 
                    meanprops=marker_config);
    
    for patch in g.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))

    g = sns.stripplot(data=data, 
                      x=x_var, 
                      size=pointsize, 
                      color="red");
    g.set(xlabel=label);
    if log_scale:
        plt.xscale('log')
    plt.show(g)

    # kernel density plot
    g = sns.displot(data=data, 
                    x=x_var, 
                    log_scale=log_scale, 
                    kind='kde', 
                    rug=True)
    g.set(xlabel=label);
    plt.show(g)

    # histogram
    g = sns.displot(data=data, 
                    x=x_var, 
                    log_scale=log_scale, 
                    kind='hist', 
                    rug=True)
    g.set(xlabel=label);
    plt.show(g)