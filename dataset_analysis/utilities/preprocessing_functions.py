from .standard_import import *
from .constants import *
from .config import *
from .sequence_statistics_functions import *
from .plotting_functions import *

class ImportInteractions(ABC):
    """Imports the interaction dataframe and filters it by interactions of a specified type. 

    Attributes
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    interaction_types_df : pd.DataFrame
        A dataframe counting the occurrences of all interactions types
    """    
     
    def __init__(self,
                 dataset_name: str,
                 interactions_type_field: Union[str, None],
                 interaction_types_to_keep: Union[list[str], None],
                 result_tables: Type[Any]) -> None:
        """ 
        Parameters
        ----------
        dataset_name : str
            The name of the dataset
        interactions_type_field : str
            The name of interaction type field
        interaction_types_to_keep : list[str]
            A list containing the interaction type values to keep
        result_tables : Type[Any]
            The ResultTables object
        """    

        self.dataset_name = dataset_name
        self.interactions_type_field = interactions_type_field
        self.interaction_types_to_keep = interaction_types_to_keep
        self.interactions = self._create_interactions_df()
        self._interactions_n_rows_before_filter = self.interactions.shape[0]
    
        if interactions_type_field and interaction_types_to_keep:
            self.interaction_types_df = self._create_interaction_type_df()
            self.interactions = self._filter_by_interaction_type()
        else:
            self.interaction_types_df = self._create_empty_interaction_type_df()

        self._interactions_n_rows_after_filter = self.interactions.shape[0]
    
        # add data to results_table
        result_tables.interaction_types = self.interaction_types_df.copy()
        
    def print_filter_results(self) -> None:
        """Prints the interaction type filter results. 
        """    

        n_removed = self._interactions_n_rows_before_filter - self._interactions_n_rows_after_filter
        pct_removed = int(round((n_removed) / self._interactions_n_rows_before_filter * 100))

        print(STAR_STRING)
        print('\n')
        print(f'== {INTERACTIONS_TYPE_FILTER_PRINT_OUTPUT_NAME_STR} ==')
        print('\n')
        print(f'{INTERACTIONS_TYPE_NAME_STR}s: ')
        print(DASH_STRING)
        if self.interactions_type_field:
            print(self.interaction_types_df)
        else:
            print(f'{INTERACTIONS_TYPE_FILTER_NOT_AVAILABLE_NAME_STR}')
        print(DASH_STRING)
        print('\n')
        print(LEARNING_ACTIVITY_SEQUENCE_FILTER_INTERACTIONS_DATAFRAME_NAME_STR)
        print(DASH_STRING)
        print(f'Input length: {self._interactions_n_rows_before_filter}')
        print(f'Output length: {self._interactions_n_rows_after_filter}')
        print(f'Number of rows removed: {n_removed}')
        print(f'Percentage of rows removed: {pct_removed}%')
        print(DASH_STRING)
        print('\n')
        print(STAR_STRING)
    
    def _create_interactions_df(self) -> pd.DataFrame:

        interactions = pd.DataFrame()    

        return interactions
    
    def _create_interaction_type_df(self) -> pd.DataFrame:

        interaction_types_df = (self.interactions[self.interactions_type_field]
                                .value_counts()
                                .reset_index(name=INTERACTIONS_COUNT_NAME_STR)
                                .rename(columns={'index': INTERACTIONS_TYPE_NAME_STR}))

        interaction_types_df[INTERACTIONS_TYPE_USED_NAME_STR] = (interaction_types_df[INTERACTIONS_TYPE_NAME_STR]
                                                                .isin(self.interaction_types_to_keep))

        interaction_types_df = (interaction_types_df
                                .sort_values(by=[INTERACTIONS_TYPE_USED_NAME_STR, INTERACTIONS_COUNT_NAME_STR], ascending=False)
                                .reset_index(drop=True))
        
        interaction_types_df.insert(0, DATASET_NAME_STR, self.dataset_name)
    
        return interaction_types_df

    def _create_empty_interaction_type_df(self) -> pd.DataFrame:

        interaction_types_df = pd.DataFrame({DATASET_NAME_STR: self.dataset_name,
                                             INTERACTIONS_TYPE_NAME_STR: None,
                                             INTERACTIONS_COUNT_NAME_STR: None,
                                             INTERACTIONS_TYPE_USED_NAME_STR: None},
                                             index=[0])

        return interaction_types_df
    
    def _filter_by_interaction_type(self) -> pd.DataFrame:

        interactions_filtered = self.interactions.loc[self.interactions[self.interactions_type_field].isin(self.interaction_types_to_keep), :]
        
        return interactions_filtered

class TransformInteractions(ABC):
    """Transforms the interaction dataframe. 

    Attributes
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    """    
     
    def __init__(self,
                 interactions: pd.DataFrame):
        """ 
        Parameters
        ----------
        interactions : pd.DataFrame
            The interactions dataframe
        """    

        self.interactions = self._transform_interactions_df(interactions.copy())  
    
    def _transform_interactions_df(self,
                                   interactions: pd.DataFrame) -> pd.DataFrame:

        return interactions

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

    print(STAR_STRING)
    print('\n')
    print(pct_na)
    print('\n')
    print(STAR_STRING)

def map_new_to_old_values(interactions: pd.DataFrame,
                          group_field: str,
                          user_field: str,
                          learning_activity_field: str) -> tuple[pd.DataFrame]:
    """Map new column values to old ones for the group, user and learning_activity fields(categorical variables). New values range from [0] to [#unique values in the respective fields -1].
    New values are of type int.

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
        A tuple containing the input interactions dataframe with mapped values in the specified categorical variable fields and a remapping dataframe which can be used in remapping the new to the original values.
    """
    field_dict = {user_field: USER_FIELD_NAME_STR, group_field: GROUP_FIELD_NAME_STR, learning_activity_field: LEARNING_ACTIVITY_FIELD_NAME_STR}
    mapping_dict_all_fields = {} 

    for field in field_dict.keys():
        mapping_dict = {}
        if field:
            values = enumerate(interactions[field].dropna().unique())
            mapping_dict = {v: n for n,v in values}
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

def drop_na_by_fields(interactions: pd.DataFrame, 
                      field_list=[]) -> pd.DataFrame:
    """Drops rows of the interactions dataframe that have NAs in any of the fields specified in field_list

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    field_list : list, optional
        A list of fields in the interactions dataframe which are used for detecting NAs, by default []

    Returns
    -------
    pd.DataFrame
        The dataframe with rows removed which have NAs in any of the fields specified in field_list 
    """    
    field_list = [i for i in field_list if i]
    input_len = interactions.shape[0]
    interactions = interactions.dropna(subset=field_list)
    output_len = interactions.shape[0]
    n_removed = input_len - output_len
    pct_removed = int(round((input_len - output_len) / input_len * 100))

    print(STAR_STRING)
    print('\n')
    print(LEARNING_ACTIVITY_SEQUENCE_FILTER_INTERACTIONS_DATAFRAME_NAME_STR)
    print(DASH_STRING)
    print(f'Input length: {input_len}')
    print(f'Output length: {output_len}')
    print(f'Number of rows removed: {n_removed}')
    print(f'Percentage of rows removed: {pct_removed}%')
    print(DASH_STRING)
    print('\n')
    print(STAR_STRING)

    return interactions

def drop_learning_activity_sequence_if_contains_na_in_field(interactions: pd.DataFrame, 
                                                            group_field: str, 
                                                            user_field: str, 
                                                            field_list=[], 
                                                            field_value_tuple_filter_list=[]) -> tuple:
    """Drops sequences groupwise from a dataframe that contain NAs in fields specified in field_list. Certain values in fields ((field, value) tuple in field_value_tuple_filter_list) can be ommited such that the sequences they are contained in will not be dropped. 

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column.
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
    print(STAR_STRING)
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


    print(LEARNING_ACTIVITY_SEQUENCE_FILTER_INTERACTIONS_DATAFRAME_NAME_STR)
    print(DASH_STRING)
    print(f'Input length: {input_len}')
    print(f'Outpunt length: {output_len}')
    print(f'Number of rows removed: {n_removed}')
    print(f'Percentage of rows removed: {pct_removed}%')
    print(DASH_STRING)
    print(f'Input number of sequences: {total_number_of_sequences}')
    print(f'Output number of sequences: {total_number_of_sequences - n_sequences_removed}')
    print(f'Number of sequences removed: {n_sequences_removed}')
    print(f'Percentage of sequences removed: {pct_sequences_removed}%')
    print(DASH_STRING)
    print('\n')
    print(STAR_STRING)

    return interactions, na_indices_list

def sort_by_timestamp(interactions: pd.DataFrame,
                      group_field: str,
                      user_field: str,
                      timestamp_field: str,
                      order_field: str):
    """Sorts the input dataframe by fields 

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    timestamp_field : str
        The timestamp field column
    order_field : str
        The order field column
        Can be set to None if the interactions dataframe does not have an order field

    Returns
    -------
    pd.DataFrame
        The sorted interactions dataframe
    """
    sort_list = [group_field, user_field, timestamp_field, order_field]
    sort_list = [i for i in sort_list if i]
    interactions = interactions.sort_values(by=sort_list)
    
    return interactions

def add_sequence_id_field(interactions: pd.DataFrame,
                          group_field: str) -> pd.DataFrame:
    """Adds a sequence_id field to the interactions dataframe. An unique id is mapped to each unique sequence of learning
    activities, indicating to which sequence a (group,user,learning_activity) entry belongs to. The sequence_id is of type int.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added sequence_id field
    """     
    if group_field:
        group_field = GROUP_FIELD_NAME_STR

    grouping_list = [group_field, USER_FIELD_NAME_STR]
    grouping_list = [i for i in grouping_list if i]

    sequences = interactions.groupby(grouping_list)[LEARNING_ACTIVITY_FIELD_NAME_STR].agg(tuple).rename(SEQUENCE_ID_FIELD_NAME_STR)
    unique_sequences = sequences.unique()

    sequence_id_mapping_dict = {seq: seq_id for seq_id, seq in enumerate(unique_sequences)}
    sequences = sequences.apply(lambda x: sequence_id_mapping_dict[x]).reset_index()

    interactions = interactions.merge(sequences, how='inner', on=grouping_list)

    sequence_id_column_index_positions = list(interactions.columns).index(LEARNING_ACTIVITY_FIELD_NAME_STR) + 1
    sequence_id_column = interactions.pop(SEQUENCE_ID_FIELD_NAME_STR)
    interactions.insert(sequence_id_column_index_positions, SEQUENCE_ID_FIELD_NAME_STR, sequence_id_column)
    
    return interactions

def rename_fields(interactions: pd.DataFrame,
                  group_field: str,
                  user_field: str,
                  learning_activity_field: str,
                  timestamp_field: str) -> tuple[pd.DataFrame]:
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
                                               EVALUATION_COURSE_FIELD_LIST)]
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
        Can be set to None if the interactions dataframe does not have a timestamp field
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
        Can be set to None if the interactions dataframe does not have a user_field
    learning_activity_field : str
        The learning_activity field column
        Can be set to None if the interactions dataframe does not have a learning_activity_field

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with typecast fields
    """
    # typecast categorical variable as strings
    cat_field_list = [group_field, user_field, learning_activity_field]
    cat_typecast_dict = {i: 'str' for i in cat_field_list if i}
    interactions = interactions.astype(cat_typecast_dict)

    # typecast timestamp as datetime field
    if timestamp_field:
        interactions[timestamp_field] = pd.to_datetime(interactions[timestamp_field], errors='coerce')

    return interactions

def change_data_type(dataframe: pd.DataFrame,
                     data_type_mapping_dict: dict[str, str]) -> pd.DataFrame:
    """Changes the datatypes of the fields of a dataframe according to a mapping dictionary

    Parameters
    ----------
    dataframe : pd.DataFrame
        A pandas DataFrame
    data_type_mapping_dict : dict[str, str]
        A dictionary containing mapping old datatypes to new ones

    Returns
    -------
    pd.DataFrame
        A new dataframe with changed datatypes
    """
    col_data_type_dict_list = []
    for old_dt, new_dt in data_type_mapping_dict.items():
        dataframe_cols = dataframe.select_dtypes(include=old_dt).columns
        n_cols = len(dataframe_cols)
        col_data_type_dict = dict(zip(dataframe_cols, [new_dt]*n_cols))
        col_data_type_dict_list.append(col_data_type_dict)

    col_data_type_remapping_dict = {k: v for dictionary in col_data_type_dict_list for k, v in dictionary.items()}
    dataframe = dataframe.astype(col_data_type_remapping_dict)
    
    return dataframe 

def return_adjusted_boxplot_outlier_threshold(data: np.array) -> tuple[float]:
    """Returns the lower and upper outlier thresholds using the adjusted boxplot algorithm proposed in
    (Hubert M. and Vandervieren E. 2008 - An adjusted boxplot for skewed distributions) which accounts for
    skewed distribution

    Parameters
    ----------
    data : np.array
        An array of values for which the adjusted boxplot outlier thresholds will be calculated

    Returns
    -------
    tuple
        A tuple containing the lower and upper outlier threshold 
    """
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    q2 = iqr(data)
    medcouple = sm.stats.stattools.medcouple(y=data)

    if medcouple >= 0:
        lower_threshold = q1 - 1.5 * np.exp(-4 * medcouple) * q2
        upper_threshold = q3 + 1.5 * np.exp(3 * medcouple) * q2
    else:
        lower_threshold = q1 - 1.5 * np.exp(-3 * medcouple) * q2
        upper_threshold = q3 + 1.5 * np.exp(4 * medcouple) * q2

    return (lower_threshold, upper_threshold)


class SequenceFilter():
    """A class used to filter out sequences and groups which do not conform to criteria specified in the parameters 

    Attributes
    ----------
    interactions : pd.DataFrame
        The filtered interaction dataframe
    unique_learning_activity_sequence_stats_per_group : pd.DataFrame
        A dataframe containing statistics for unique sequences per group 
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A dataframe containing statistics for all sequences per group 
    sequence_count_per_group : pd.DataFrame
        A dataframe containing sequence counts and unique sequence counts per group for each preprocessing step
    group_count : pd.DataFrame
        A dataframe containing group counts for each preprocessing step
    seq_filter_parameters : pd.DataFrame
        A dataframe containing the parameter values used for filtering sequences and groups

    Methods
    -------
    filter_sequences
        Filters sequences and groups according to the specified parameters and plots thresholds and sequence/group counts
        for the respective preprocessing steps
    """

    def __init__(self,
                 dataset_name: str, 
                 interactions: pd.DataFrame,
                 group_field,
                 min_pct_unique_learning_activities_per_group_in_seq_threshold: int,
                 max_pct_repeated_learning_activities_in_seq_threshold: int,
                 min_sequence_number_per_group_threshold: int,
                 min_unique_sequence_number_per_group_threshold: int):

        self.dataset_name = dataset_name
        self.interactions = interactions
        self.group_field = group_field
        self.min_pct_unique_learning_activities_per_group_in_seq = min_pct_unique_learning_activities_per_group_in_seq_threshold
        self.max_pct_repeated_learning_activities_in_seq = max_pct_repeated_learning_activities_in_seq_threshold
        self.min_sequence_number_per_group_threshold = min_sequence_number_per_group_threshold
        self.min_unique_sequence_number_per_group_threshold = min_unique_sequence_number_per_group_threshold

        @property
        def interactions(self):
            return self._interactions.copy()
    
        @interactions.setter
        def interactions(self, 
                         interactions: pd.DataFrame):

            if isinstance(interactions, pd.DataFrame):
                self._interactions = interactions.copy() 
            else:
                self._interactions = None 

        # initial calculations
        # TODO: adapt for datasets without groups
        self.interactions[GROUP_FIELD_NAME_STR] = self.interactions[GROUP_FIELD_NAME_STR].astype(int)

        self.unique_learning_activity_sequence_stats_per_group = SequenceStatistics.return_unique_learning_activity_sequence_stats_per_group(self.interactions,
                                                                                                                                             self.dataset_name,
                                                                                                                                             self.group_field)
        self.learning_activity_sequence_stats_per_group = SequenceStatistics.return_learning_activity_sequence_stats_per_group(self.unique_learning_activity_sequence_stats_per_group)
        
        self.sequence_count_per_group = self._return_seq_count_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                         LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_ORIGINAL_NAME_STR)

        self.group_count = self._return_group_count(self.unique_learning_activity_sequence_stats_per_group,
                                                    LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_ORIGINAL_NAME_STR)
        
        self.seq_filter_parameters = self._create_parameter_df()

    def filter_sequences(self,
                         result_tables: Type[Any]):
        """Filter out sequences and groups which do not conform to criteria specified in the objects parameters 

        Parameters
        ----------
        result_tables : Type[Any]
            The ResultTables object
        """
        # filtering and plotting
        self._filter_sequences_by_min_unique_max_repeated_learning_activities()
        self._filter_sequences_by_length_outliers()
        self._plot_seq_count_change_per_group(self.sequence_count_per_group)
        self._filter_groups_by_sequence_count()
        self._plot_group_count_change(self.group_count)

        # add data to results_table
        result_tables.seq_filter_parameters = self.seq_filter_parameters.copy()
        result_tables.seq_filter_sequence_count_per_group = self.sequence_count_per_group.copy()
        result_tables.seq_filter_group_count = self.group_count.copy()

    @staticmethod
    def _plot_sequence_filter_thresholds(learning_activity_sequence_stats_per_group: pd.DataFrame,
                                         sequence_statistic: str,
                                         sequence_statistic_label: str,
                                         threshold: Union[int, list[tuple[float]]]) -> None:
        if isinstance(threshold, int):
            title_str = f'{sequence_statistic}{LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_FILTER_TITLE_STR}{threshold}%'
            share_x_axis = True
            axis_xlim = return_axis_limits(learning_activity_sequence_stats_per_group[sequence_statistic],
                                          True)
        elif isinstance(threshold, list):
            title_str = f'{sequence_statistic}{LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_LENGTH_OUTLIER_TITLE_STR}'
            share_x_axis = True
            axis_xlim = None
        else:
            raise TypeError('threshold needs to be int or list!')
        
        n_cols = set_facet_grid_column_number(learning_activity_sequence_stats_per_group[GROUP_FIELD_NAME_STR],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

        g=sns.displot(learning_activity_sequence_stats_per_group,
                      x=sequence_statistic,
                      col=GROUP_FIELD_NAME_STR,
                      kind='hist',
                      stat='count',
                      col_wrap=n_cols,
                      height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                      aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                      bins=SEABORN_HISTOGRAM_BIN_CALC_METHOD,
                      alpha=SEABORN_PLOT_OBJECT_ALPHA,
                      facet_kws=dict(sharex=share_x_axis,
                                     sharey=False,
                                     xlim=axis_xlim))
        g.map_dataframe(sns.rugplot,
                        sequence_statistic,
                        height=SEABORN_RUG_PLOT_HEIGHT_PROPORTION_FACET,
                        color=SEABORN_RUG_PLOT_COLOR,
                        expand_margins=True,
                        alpha=SEABORN_RUG_PLOT_ALPHA_FACET)

        # vertical line
        axes = g.figure.axes
        if isinstance(threshold, int):
            for ax in axes:
                ax.axvline(x=threshold, 
                           ymin=0, 
                           ymax=1, 
                           color=SEABORN_LINE_COLOR_RED, 
                           linewidth=SEABORN_LINE_WIDTH_FACET);
        elif isinstance(threshold, list):
            for ax, threshold_tuple in zip(axes, threshold):
                if threshold_tuple[0] >= 0 or threshold_tuple[1] >= 0: 
                    ax.axvline(x=threshold_tuple[0], 
                               ymin=0, 
                               ymax=1, 
                               color=SEABORN_LINE_COLOR_RED, 
                               linewidth=SEABORN_LINE_WIDTH_FACET);
                    ax.axvline(x=threshold_tuple[1], 
                               ymin=0, 
                               ymax=1, 
                               color=SEABORN_LINE_COLOR_RED, 
                               linewidth=SEABORN_LINE_WIDTH_FACET);
                else:
                    continue
        else:
            raise TypeError('threshold needs to be int or list!')

        g.set(xlabel=sequence_statistic_label)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.tight_layout()
        y_loc = calculate_suptitle_position(g,
                                            SEABORN_SUPTITLE_HEIGHT_CM)
        g.figure.suptitle(title_str, 
                          fontsize=SEABORN_TITLE_FONT_SIZE,
                          y=y_loc)
        plt.show(g)

    @staticmethod
    def _plot_seq_count_change_per_group(seq_count_df: pd.DataFrame) -> None:

        seq_count_change_df_long = pd.melt(seq_count_df, 
                                           id_vars=[DATASET_NAME_FIELD_NAME_STR, 
                                                    GROUP_FIELD_NAME_STR, 
                                                    LEARNING_ACTIVITY_SEQUENCE_COUNT_TYPE_NAME_STR],
                                           var_name=LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_STEP_NAME_STR,
                                           value_name=LEARNING_ACTIVITY_SEQUENCE_COUNT_VALUE_NAME_STR)

        n_cols = set_facet_grid_column_number(seq_count_df[GROUP_FIELD_NAME_STR],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

        g=sns.catplot(seq_count_change_df_long,
                      x=LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_STEP_NAME_STR,
                      y=LEARNING_ACTIVITY_SEQUENCE_COUNT_VALUE_NAME_STR,
                      hue=LEARNING_ACTIVITY_SEQUENCE_COUNT_TYPE_NAME_STR,
                      col=GROUP_FIELD_NAME_STR,
                      kind='point',
                      col_wrap=n_cols,
                      height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                      aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,         
                      sharex=True, 
                      sharey=True)

        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g._legend.set_frame_on(True)
        y_loc = calculate_suptitle_position(g,
                                            SEABORN_SUPTITLE_HEIGHT_CM)
        g.figure.suptitle(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_COUNT_CHANGE_TITLE_NAME_STR, 
                          fontsize=SEABORN_TITLE_FONT_SIZE,
                          y=y_loc)
        # g.add_legend()
        plt.show();

    @staticmethod
    def _plot_sequence_count_per_group(unique_learning_activity_sequence_stats_per_group: pd.DataFrame,
                                       min_sequence_number_per_group_threshold: int,
                                       min_unique_sequence_number_per_group_threshold: int) -> None:

        count_df = unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR).head(1)

        # all groups in one figure - unique seq count vs seq count scatter 
        ylim = return_axis_limits(count_df[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR],
                                  False)
        title_string_scatter = LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_UNIQUE_SEQUENCE_VS_SEQUENCE_COUNT_SCATTER_TITLE_NAME_STR
        seq_count_threshold_str = f'\n{LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_COUNT_THRESHOLD_TITLE_NAME_STR}{min_sequence_number_per_group_threshold}'
        unique_seq_count_threshold_str = f'\n{LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_UNIQUE_SEQUENCE_COUNT_THRESHOLD_TITLE_NAME_STR}{min_unique_sequence_number_per_group_threshold}'
        title_string_scatter = title_string_scatter + seq_count_threshold_str + unique_seq_count_threshold_str

        # jointplot
        g = sns.jointplot(data=count_df,
                          x=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
                          y=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
                          height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_SINGLE,
                          s=SEABORN_POINT_SIZE_JOINTPLOT,
                          alpha=SEABORN_POINT_ALPHA_JOINTPLOT_PREPROCESSING,
                          edgecolor=SEABORN_POINT_EDGECOLOR,
                          linewidth=SEABORN_POINT_LINEWIDTH,
                          marginal_kws={'alpha': SEABORN_PLOT_OBJECT_ALPHA,
                                        'bins': SEABORN_HISTOGRAM_BIN_CALC_METHOD})

        g.plot_marginals(sns.rugplot, 
                         color=SEABORN_RUG_PLOT_COLOR, 
                         height=SEABORN_RUG_PLOT_HEIGHT_PROPORTION_JOINTPLOT,
                         alpha=SEABORN_RUG_PLOT_ALPHA_JOINTPLOT)
        g.ax_joint.set_ylim(ylim)
        g.ax_joint.set_xlim(ylim)
        plt.tight_layout()
        y_loc = calculate_suptitle_position(g,
                                            SEABORN_SUPTITLE_HEIGHT_CM)
        g.figure.suptitle(title_string_scatter, 
                          fontsize=SEABORN_TITLE_FONT_SIZE,
                          y=y_loc)
        g.ax_joint.axline(xy1=(0,0), 
                          slope=1, 
                          color=SEABORN_LINE_COLOR_ORANGE, 
                          linewidth=SEABORN_LINE_WIDTH_JOINTPLOT, 
                          zorder=0)
        if min_sequence_number_per_group_threshold:
            g.ax_joint.axvline(x=min_sequence_number_per_group_threshold, 
                               ymin=0, 
                               ymax=1, 
                               color=SEABORN_LINE_COLOR_RED, 
                               linewidth=SEABORN_LINE_WIDTH_JOINTPLOT)
        if min_unique_sequence_number_per_group_threshold:
            g.ax_joint.axhline(y=min_unique_sequence_number_per_group_threshold, 
                               xmin=0, 
                               xmax=1, 
                               color=SEABORN_LINE_COLOR_RED, 
                               linewidth=SEABORN_LINE_WIDTH_JOINTPLOT)
        plt.show(g);

    @staticmethod
    def _plot_group_count_change(group_count_df: pd.DataFrame) -> None:

        group_count_df = pd.melt(group_count_df, 
                                 id_vars=[DATASET_NAME_FIELD_NAME_STR],
                                 var_name=LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_STEP_NAME_STR,
                                 value_name=LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_GROUP_COUNT_NAME_STR)

        g=sns.catplot(group_count_df,
                      x=LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_STEP_NAME_STR,
                      y=LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_GROUP_COUNT_NAME_STR,
                      kind='point',
                      height=SEABORN_FIGURE_LEVEL_HEIGHT_WIDE_SINGLE,
                      aspect=SEABORN_FIGURE_LEVEL_ASPECT_WIDE)

        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.tight_layout()
        y_loc = calculate_suptitle_position(g,
                                            SEABORN_SUPTITLE_HEIGHT_CM)
        g.figure.suptitle(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_GROUP_COUNT_CHANGE_TITLE_NAME_STR, 
                          fontsize=SEABORN_TITLE_FONT_SIZE,
                          y=y_loc)
        plt.show(g);

    def _preprocessing_step_print_output(self,
                                         preprocessing_step_full_name: str,
                                         preprocessing_base: str,
                                         filtering_threshold: Union[str, list[str]]) -> None:
        print('')
        print(STAR_STRING)
        print(STAR_STRING)
        print('')
        print(f'!!! {preprocessing_step_full_name} !!!')
        print('')
        print(preprocessing_base)
        print('')
        if isinstance(filtering_threshold, list):
            for i in filtering_threshold:
                print(i)
        elif isinstance(filtering_threshold, str):
            print(filtering_threshold)
        else:
            raise TypeError(f'filtering_threshold needs to be either str or list')
        print('')

    def _print_seq_preprocessing_stats(self,
                                       df_before_filter: pd.DataFrame,
                                       df_after_filter: pd.DataFrame) -> None:

        n_rows_original = df_before_filter.shape[0]
        n_rows_after_filter = df_after_filter.shape[0]
        number_rows_removed_after_filter = n_rows_original - n_rows_after_filter
        pct_rows_removed_after_filter = (n_rows_original - n_rows_after_filter) / n_rows_original * 100

        datatype_mapping_dict = {'int64': 'Int64', 'float64': 'Int64'}
        sequence_count_per_group = change_data_type(self.sequence_count_per_group,
                                                    datatype_mapping_dict)
        group_count = change_data_type(self.group_count,
                                       datatype_mapping_dict)

        print('')
        print(f'== {LEARNING_ACTIVITY_SEQUENCE_FILTER_INTERACTIONS_DATAFRAME_NAME_STR} ==')
        print(DASH_STRING)
        print(f'{LEARNING_ACTIVITY_SEQUENCE_FILTER_ROWS_BEFORE_FILTER_NAME_STR}{n_rows_original}')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_FILTER_ROWS_AFTER_FILTER_NAME_STR}{n_rows_after_filter}')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_FILTER_ROWS_REMOVED_NAME_STR}{number_rows_removed_after_filter}')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_FILTER_ROWS_REMOVED_PCT_NAME_STR}{pct_rows_removed_after_filter}%')
        print(DASH_STRING)
        print('')
        print(f'== {LEARNING_ACTIVITY_SEQUENCE_FILTER_SEQUENCE_COUNT_DATAFRAME_NAME_STR} ==')
        print(DASH_STRING)
        print(sequence_count_per_group)
        print(DASH_STRING)
        print('')
        print(f'== {LEARNING_ACTIVITY_SEQUENCE_FILTER_GROUP_COUNT_DATAFRAME_NAME_STR} ==')
        print(DASH_STRING)
        print(group_count)
        print(DASH_STRING)

    def _return_seq_count_per_group(self,
                                    unique_learning_activity_sequence_stats_per_group: pd.DataFrame,
                                    preprocessing_step: str) -> pd.DataFrame:
    
        seq_count_series = unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR].sum()
        seq_count_series.name = LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR

        unique_seq_count_series = unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR).size()
        unique_seq_count_series.name = LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_NAME_STR

        seq_count_df = pd.concat([seq_count_series, unique_seq_count_series], axis=1).reset_index()
        seq_count_df = pd.melt(seq_count_df,
                               id_vars=GROUP_FIELD_NAME_STR,
                               var_name=LEARNING_ACTIVITY_SEQUENCE_COUNT_TYPE_NAME_STR,
                               value_name=preprocessing_step)
        
        seq_count_df.insert(0, 
                            DATASET_NAME_FIELD_NAME_STR,
                            self.dataset_name)

        return seq_count_df

    def _merge_seq_count_dataframes(self,
                                    df_1: pd.DataFrame,
                                    df_2: pd.DataFrame) -> pd.DataFrame:

        seq_count_df = pd.merge(df_1, 
                                df_2, 
                                on=[DATASET_NAME_FIELD_NAME_STR,
                                    GROUP_FIELD_NAME_STR, 
                                    LEARNING_ACTIVITY_SEQUENCE_COUNT_TYPE_NAME_STR],
                                how='left')

        return seq_count_df

    def _return_group_count(self,
                            unique_learning_activity_sequence_stats_per_group: pd.DataFrame,
                            preprocessing_step: str) -> pd.DataFrame:

        group_count = unique_learning_activity_sequence_stats_per_group[GROUP_FIELD_NAME_STR].nunique()
        group_count_dict = {DATASET_NAME_FIELD_NAME_STR: self.dataset_name,
                            preprocessing_step: group_count}

        group_count_df = pd.DataFrame(group_count_dict, index=[0])

        return group_count_df

    def _merge_group_count_dataframes(self,
                                      df_1: pd.DataFrame,
                                      df_2: pd.DataFrame) -> pd.DataFrame:

        group_count_df = pd.merge(df_1, 
                                  df_2,
                                  on=[DATASET_NAME_FIELD_NAME_STR])
        
        return group_count_df

    def _filter_sequences_per_group(self,
                                    interactions: pd.DataFrame,
                                    sequences_to_keep_dict: dict[int, np.array]) -> pd.DataFrame: 

        filtered_df_list = []
        for group, df in interactions.groupby(GROUP_FIELD_NAME_STR):
            filtered_df_list.append(df.loc[df[SEQUENCE_ID_FIELD_NAME_STR].isin(sequences_to_keep_dict.get(group, [])), :])

        filtered_df = pd.concat(filtered_df_list)

        return filtered_df

    def _filter_groups(self,
                       interactions: pd.DataFrame,
                       groups_to_keep: np.array) -> pd.DataFrame:

        filtered_df = interactions.loc[interactions[GROUP_FIELD_NAME_STR].isin(groups_to_keep), :]

        return filtered_df

    def _update_data_after_filtering(self,
                                     filter_type: Literal['sequence', 'group'],
                                     filter_var: Union[list, dict[int, np.array]],
                                     preprocessing_step: str) -> None:

        interactions_before = self.interactions.copy() 

        if filter_type == LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_SEQUENCE_NAME_STR:
            self.interactions = self._filter_sequences_per_group(self.interactions,
                                                                 filter_var)
        elif filter_type == LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_GROUP_NAME_STR:
            self.interactions = self._filter_groups(self.interactions,
                                                    filter_var)
        else:
            raise TypeError(f'filter_type needs to be either "{LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_SEQUENCE_NAME_STR}" or "{LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_GROUP_NAME_STR}"')

        self.unique_learning_activity_sequence_stats_per_group = SequenceStatistics.return_unique_learning_activity_sequence_stats_per_group(self.interactions,
                                                                                                                          self.dataset_name,
                                                                                                                          self.group_field)
        self.learning_activity_sequence_stats_per_group = SequenceStatistics.return_learning_activity_sequence_stats_per_group(self.unique_learning_activity_sequence_stats_per_group)

        sequence_count_per_group_new = self._return_seq_count_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                        preprocessing_step)
        
        self.sequence_count_per_group = self._merge_seq_count_dataframes(self.sequence_count_per_group,
                                                                         sequence_count_per_group_new)

        group_count_new = self._return_group_count(self.unique_learning_activity_sequence_stats_per_group,
                                                   preprocessing_step)
        self.group_count = self._merge_group_count_dataframes(self.group_count,
                                                              group_count_new)

        self._print_seq_preprocessing_stats(interactions_before,
                                            self.interactions)

    def _filter_sequences_by_min_unique_max_repeated_learning_activities(self) -> None:
        # remove sequences which do not conform to the threshold values in 'min_pct_unique_learning_activities_per_group_in_seq' and 'max_pct_repeated_learning_activities_in_seq'
        
        threshold_min_unique_str = LEARNING_ACTIVITY_SEQUENCE_FILTER_STEP_MIN_UNIQUE_NAME_STR + str(self.min_pct_unique_learning_activities_per_group_in_seq) + '%'
        threshold_max_repeated_str = LEARNING_ACTIVITY_SEQUENCE_FILTER_STEP_MAX_REPEATED_NAME_STR + str(self.max_pct_repeated_learning_activities_in_seq) + '%'
        threshold_max_repeated_str = threshold_max_repeated_str.replace('\n', '')
        self._preprocessing_step_print_output(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FIRST_STEP_FULL_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_BASE_SEQUENCE_FILTER_NAME_STR,
                                              [threshold_min_unique_str, threshold_max_repeated_str])

        self._plot_sequence_filter_thresholds(self.learning_activity_sequence_stats_per_group,
                                              LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
                                              self.min_pct_unique_learning_activities_per_group_in_seq)
        self._plot_sequence_filter_thresholds(self.learning_activity_sequence_stats_per_group,
                                              LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
                                              self.max_pct_repeated_learning_activities_in_seq)

        min_pct_unique_learning_activities_filter = self.unique_learning_activity_sequence_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR] >= self.min_pct_unique_learning_activities_per_group_in_seq
        max_pct_repeated_learning_activities_filter = self.unique_learning_activity_sequence_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR] <= self.max_pct_repeated_learning_activities_in_seq
        row_filter = (min_pct_unique_learning_activities_filter & max_pct_repeated_learning_activities_filter)

        self.unique_learning_activity_sequence_stats_per_group = self.unique_learning_activity_sequence_stats_per_group.loc[row_filter, :]
        seq_to_keep_per_group_dict = self.unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR)[SEQUENCE_ID_FIELD_NAME_STR].agg(list).to_dict()

        self._update_data_after_filtering(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_SEQUENCE_NAME_STR,
                                          seq_to_keep_per_group_dict,
                                          LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_MIN_UNIQUE_MAX_REPEATED_LEARNING_ACTIVITY_NAME_STR)

    def _filter_sequences_by_length_outliers(self) -> None:
        # remove outlier sequences by sequence length

        self._preprocessing_step_print_output(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SECOND_STEP_FULL_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_BASE_SEQUENCE_FILTER_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_FILTER_STEP_SEQUENCE_LENGTH_OUTLIER_NAME_STR)
        seq_to_keep_per_group_dict = {}
        threshold_tuples_list = []
        for group, df in self.learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR):

            seq_len_values = df[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR].values
            if len(seq_len_values) > 1:
                min_threshold, max_threshold = return_adjusted_boxplot_outlier_threshold(seq_len_values)
            else:
                min_threshold, max_threshold = -1, -1
                 
            threshold_tuples_list.append((min_threshold, max_threshold))

            min_filter = self.unique_learning_activity_sequence_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR] >= min_threshold
            max_filter = self.unique_learning_activity_sequence_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR] <= max_threshold
            row_filter = (min_filter & max_filter)
            if len(seq_len_values) > 1:
                seq_to_keep_per_group_dict[group] = self.unique_learning_activity_sequence_stats_per_group[SEQUENCE_ID_FIELD_NAME_STR].loc[row_filter].values
            else:
                seq_to_keep_per_group_dict[group] = self.unique_learning_activity_sequence_stats_per_group[SEQUENCE_ID_FIELD_NAME_STR].values

        self._plot_sequence_filter_thresholds(self.learning_activity_sequence_stats_per_group,
                                              LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                              threshold_tuples_list)

        self._update_data_after_filtering(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_SEQUENCE_NAME_STR,
                                          seq_to_keep_per_group_dict,
                                          LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_LENGTH_OUTLIER_NAME_STR)

    def _filter_groups_by_sequence_count(self) -> None:

        threshold_sequence_count = LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_COUNT_THRESHOLD_TITLE_NAME_STR + str(self.min_sequence_number_per_group_threshold)
        threshold_unique_sequence_count = LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_UNIQUE_SEQUENCE_COUNT_THRESHOLD_TITLE_NAME_STR + str(self.min_unique_sequence_number_per_group_threshold) 
        self._preprocessing_step_print_output(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_THIRD_STEP_FULL_NAME_STR,
                                              LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_BASE_GROUP_FILTER_NAME_STR,
                                              [threshold_sequence_count, threshold_unique_sequence_count])

        self._plot_sequence_count_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                            self.min_sequence_number_per_group_threshold,
                                            self.min_unique_sequence_number_per_group_threshold)

        seq_count_str = LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR
        unique_seq_count_str = LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR
        seq_count_field_list = [seq_count_str, unique_seq_count_str]
        count_df = self.unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR)[seq_count_field_list].first().reset_index()

        seq_count_filter = count_df[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR] >= self.min_sequence_number_per_group_threshold 
        unique_seq_count_filter = count_df[LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR] >= self.min_unique_sequence_number_per_group_threshold 
        row_filter = (seq_count_filter & unique_seq_count_filter)
        
        groups_to_keep_array = count_df.loc[row_filter, GROUP_FIELD_NAME_STR].values

        self._update_data_after_filtering(LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_FILTER_TYPE_GROUP_NAME_STR,
                                          groups_to_keep_array,
                                          LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_SEQUENCE_COUNT_PER_GROUP_NAME_STR)

    def _create_parameter_df(self):
        
        parameter_df = pd.DataFrame({DATASET_NAME_STR: self.dataset_name,
                                     LEARNING_ACTIVITY_SEQUENCE_FILTER_STEP_MIN_UNIQUE_NAME_STR.replace('"', '').replace('\n', ''): self.min_pct_unique_learning_activities_per_group_in_seq,
                                     LEARNING_ACTIVITY_SEQUENCE_FILTER_STEP_MAX_REPEATED_NAME_STR.replace('"', '').replace('\n', ''): self.max_pct_repeated_learning_activities_in_seq,
                                     LEARNING_ACTIVITY_MINIMUM_SEQUENCE_COUNT_PER_GROUP_NAME_STR: self.min_sequence_number_per_group_threshold,
                                     LEARNING_ACTIVITY_MINIMUM_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR: self.min_unique_sequence_number_per_group_threshold},
                                     index=[0])
        
        return parameter_df
        