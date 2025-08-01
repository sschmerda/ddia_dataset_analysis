from .standard_import import *
from .constants.constants import *
from .preprocessing_functions import *
from .html_style_functions import *

def calculate_sparsity(index: pd.Series, 
                       column: pd.Series) -> float:
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

class ResultTables():
    """A class which upon initialization holds data about the available fields, the score is_correct field relationship and summary statistics of 
    the input interactions dataframe.
    This data can be displayed as html tables with the corresponding display_* methods.

    Parameters
    ----------
    dataset_name : str
        The name of the input dataframe
    group_field : str
        The group field name
    eval_score_ranges_dict : dict
        A list of data used for evaluation score ranges calculations
    interactions : pd.DataFrame
        The interactions dataframe
    interaction_types : pd.DataFrame
        The interaction types dataframe
    evaluation_score_range_list : list
        A list containing data about evaluation fields at the learning activity, group and course level.
    seq_filter_parameters : pd.DataFrame
        The seq_filter_parameters dataframe
    seq_filter_sequence_count_per_group : pd.DataFrame
        A dataframe containing sequence counts and unique sequence counts per group for each preprocessing step
    seq_filter_group_count : pd.DataFrame
        A dataframe containing group counts for each preprocessing step
    seq_stats_sequence_count_per_group : pd.DataFrame
        A dataframe containing sequence counts and unique sequence counts per group
    seq_stat_conf_int_df : pd.DataFrame
        A dataframe containing confidence intervals for sequence statistics per group
    unique_learning_activity_sequence_stats_per_group : pd.DataFrame
        A dataframe containing statistics for unique sequences per group 
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A dataframe containing statistics for all sequences per group 
    cluster_results_per_group_df : pd.DataFrame
        A dataframe containing parameters and cluster statistics for the optimal cluster results per group.
    sequence_cluster_per_group_df : pd.DataFrame
        A dataframe containing sequence-cluster relation information per group 
    omnibus_test_result_df : pd.DataFrame
        A dataframe containing the results of the omnibus tests for differences in the evaluation metric between clusters 
    measure_association_conf_int_bootstrap_failures_df : pd.DataFrame
        A dataframe containing information about the number of failures in the calculation of the bootstrap confidence intervals
        of the measures of association
    object_size_df: pd.DataFrame
        A dataframe containing informations about the in-memory sizes of objects relevant for preprocessing and the analysis
    """    
    # available fields class vars
    is_available_str = IS_AVAILABLE_STR
    field_str = FIELD_STR

    # score is_correct relationship class vars
    has_field_str = HAS_FIELD_STR
    has_score_field_str = HAS_SCORE_FIELD_STR
    has_is_correct_field_str = HAS_IS_CORRECT_FIELD_STR
    chosen_score_correct_threshold_str = CHOSEN_SCORE_CORRECT_THRESHOLD_STR
    score_minimum_docu_str = SCORE_MINIMUM_DOCU_STR
    score_maximum_docu_str = SCORE_MAXIMUM_DOCU_STR
    are_equal_all_score_minima_str = ARE_EQUAL_ALL_SCORE_MINIMA_STR 
    are_equal_all_score_maxima_str = ARE_EQUAL_ALL_SCORE_MAXIMA_STR 
    score_minimum_data_str = SCORE_MINIMUM_DATA_STR
    score_maximum_data_str = SCORE_MAXIMUM_DATA_STR

    # summary statistics class vars
    n_rows_str = N_ROWS_STR
    n_unique_users_str = N_UNIQUE_USERS_STR
    n_unique_learning_activities_str = N_UNIQUE_LEARNING_ACTIVITIES_STR
    n_unique_groups_str = N_UNIQUE_GROUPS_STR
    sparsity_user_learning_activity_matrix_str = SPARSITY_USER_LEARNING_ACTIVITY_MATRIX_STR
    sparsity_user_group_matrix_str = SPARSITY_USER_GROUP_MATRIX_STR

    # sequence statistics
    n_sequences_str = N_SEQUENCES_STR
    n_unique_sequences_str = N_UNIQUE_SEQUENCES_STR
    mean_sequence_length_str = MEAN_SEQUENCE_LENGTH_STR
    median_sequence_length_str = MEDIAN_SEQUENCE_LENGTH_STR
    std_sequence_length_str = STD_SEQUENCE_LENGTH_STR
    iqr_sequence_length_str = IQR_SEQUENCE_LENGTH_STR

    def __init__(self, 
                 dataset_name: str, 
                 group_field: str,
                 evaluation_score_ranges_data_list: list,
                 interactions: pd.DataFrame,
                 interaction_types: pd.DataFrame,
                 eval_score_ranges_dict: dict,
                 seq_filter_parameters: pd.DataFrame,
                 seq_filter_sequence_count_per_group: pd.DataFrame,
                 seq_filter_group_count: pd.DataFrame,
                 seq_stats_sequence_count_per_group: pd.DataFrame,
                 seq_stat_conf_int_df: pd.DataFrame,
                 unique_learning_activity_sequence_stats_per_group: pd.DataFrame,
                 learning_activity_sequence_stats_per_group: pd.DataFrame,
                 best_cluster_results_per_group_df: pd.DataFrame,
                 sequence_cluster_per_group_df: pd.DataFrame,
                 omnibus_test_result_df: pd.DataFrame,
                 measure_association_conf_int_bootstrap_failures_df: pd.DataFrame,
                 object_size_df: pd.DataFrame) -> None: 

        # data to be calculated
        self.group_list = None
        self.available_fields_df = None
        self.summary_statistics_df = None
        self.sequence_statistics_df = None
        self.eval_score_ranges_list = None
        self.score_is_correct_rel_df = None

        # set upon object initialization 
        self.dataset_name = dataset_name
        self.group_field = group_field
        self.evaluation_score_ranges_data_list = evaluation_score_ranges_data_list

        # set upon initialization or later as attribute
        self.interactions = interactions
        self.interaction_types = interaction_types
        self.eval_score_ranges_dict = eval_score_ranges_dict
        self.seq_filter_parameters = seq_filter_parameters
        self.seq_filter_sequence_count_per_group = seq_filter_sequence_count_per_group
        self.seq_filter_group_count = seq_filter_group_count
        self.seq_stats_sequence_count_per_group = seq_stats_sequence_count_per_group
        self.seq_stat_conf_int_df = seq_stat_conf_int_df
        self.unique_learning_activity_sequence_stats_per_group = unique_learning_activity_sequence_stats_per_group
        self.learning_activity_sequence_stats_per_group = learning_activity_sequence_stats_per_group
        self.best_cluster_results_per_group_df = best_cluster_results_per_group_df
        self.sequence_cluster_per_group_df = sequence_cluster_per_group_df
        self.omnibus_test_result_df = omnibus_test_result_df
        self.measure_association_conf_int_bootstrap_failures_df = measure_association_conf_int_bootstrap_failures_df
        self.object_size_df = object_size_df

    @property
    def interactions(self):
        return self._interactions
    
    @interactions.setter
    def interactions(self, 
                     interactions: pd.DataFrame):

        if isinstance(interactions, pd.DataFrame):
            self._interactions = interactions.copy() 

            # calculated upon interactions initialization
            self.group_list = self._gen_group_list()
            self.available_fields_df = self._gen_available_fields_df()
            self.summary_statistics_df = self._gen_summary_statistics_df()
            self.sequence_statistics_df = self._gen_sequence_statistics_df()

        else:
            self._interactions = None 

            self.group_list = None
            self.available_fields_df = None
            self.summary_statistics_df = None
            self.sequence_statistics_df = None

    @property
    def interaction_types(self):
        return self._interaction_types
    
    @interaction_types.setter
    def interaction_types(self, 
                          interaction_types: pd.DataFrame):

        if isinstance(interaction_types, pd.DataFrame):
            self._interaction_types = interaction_types.copy() 

        else:
            self._interaction_types = None 

    @property
    def eval_score_ranges_dict(self):
        return self._eval_score_ranges_dict
    
    @eval_score_ranges_dict.setter
    def eval_score_ranges_dict(self, 
                               eval_score_ranges_dict: dict):

        if isinstance(eval_score_ranges_dict, dict):

            self._eval_score_ranges_dict = eval_score_ranges_dict.copy() 
            self.eval_score_ranges_list = list(zip(self._eval_score_ranges_dict.items(), 
                                                   self.evaluation_score_ranges_data_list))

            # calculated upon evaluation_score_range_dict initialization
            self.score_is_correct_rel_df = self._gen_score_is_correct_rel_df()
        else:
            self._eval_score_ranges_dict = None 

            self.eval_score_ranges_list = None
            self.score_is_correct_rel_df = None

    @property
    def seq_filter_parameters(self):
        return self._seq_filter_parameters
    
    @seq_filter_parameters.setter
    def seq_filter_parameters(self, 
                              seq_filter_parameters: pd.DataFrame):

        if isinstance(seq_filter_parameters, pd.DataFrame):
            self._seq_filter_parameters = seq_filter_parameters.copy()

        else:
            self._seq_filter_parameters = None 

    @property
    def seq_filter_sequence_count_per_group(self):
        return self._seq_filter_sequence_count_per_group
    
    @seq_filter_sequence_count_per_group.setter
    def seq_filter_sequence_count_per_group(self, 
                                            seq_filter_sequence_count_per_group: pd.DataFrame):

        if isinstance(seq_filter_sequence_count_per_group, pd.DataFrame):
            self._seq_filter_sequence_count_per_group = seq_filter_sequence_count_per_group.copy()

        else:
            self._seq_filter_sequence_count_per_group = None 

    @property
    def seq_filter_group_count(self):
        return self._seq_filter_group_count
    
    @seq_filter_group_count.setter
    def seq_filter_group_count(self, 
                               seq_filter_group_count: pd.DataFrame):

        if isinstance(seq_filter_group_count, pd.DataFrame):
            self._seq_filter_group_count = seq_filter_group_count.copy()

        else:
            self._seq_filter_group_count = None 

    @property
    def seq_stats_sequence_count_per_group(self):
        return self._seq_stats_sequence_count_per_group
    
    @seq_stats_sequence_count_per_group.setter
    def seq_stats_sequence_count_per_group(self, 
                                           seq_stats_sequence_count_per_group: pd.DataFrame):

        if isinstance(seq_stats_sequence_count_per_group, pd.DataFrame):
            self._seq_stats_sequence_count_per_group = seq_stats_sequence_count_per_group.copy()

        else:
            self._seq_stats_sequence_count_per_group = None 
    
    @property
    def seq_stat_conf_int_df(self):
        return self._seq_stat_conf_int_df
    
    @seq_stat_conf_int_df.setter
    def seq_stat_conf_int_df(self, 
                             seq_stat_conf_int_df: pd.DataFrame):

        if isinstance(seq_stat_conf_int_df, pd.DataFrame):
            self._seq_stat_conf_int_df = seq_stat_conf_int_df.copy()

        else:
            self._seq_stat_conf_int_df = None 

    @property
    def unique_learning_activity_sequence_stats_per_group(self):
        return self._unique_learning_activity_sequence_stats_per_group
    
    @unique_learning_activity_sequence_stats_per_group.setter
    def unique_learning_activity_sequence_stats_per_group(self, 
                                                          unique_learning_activity_sequence_stats_per_group: pd.DataFrame):

        if isinstance(unique_learning_activity_sequence_stats_per_group, pd.DataFrame):
            self._unique_learning_activity_sequence_stats_per_group = unique_learning_activity_sequence_stats_per_group.copy()

        else:
            self._unique_learning_activity_sequence_stats_per_group = None 

    @property
    def learning_activity_sequence_stats_per_group(self):
        return self._learning_activity_sequence_stats_per_group
    
    @learning_activity_sequence_stats_per_group.setter
    def learning_activity_sequence_stats_per_group(self, 
                                                   learning_activity_sequence_stats_per_group: pd.DataFrame):

        if isinstance(learning_activity_sequence_stats_per_group, pd.DataFrame):
            self._learning_activity_sequence_stats_per_group = learning_activity_sequence_stats_per_group.copy()

        else:
            self._learning_activity_sequence_stats_per_group = None 

    @property
    def best_cluster_results_per_group_df(self):
        return self._best_cluster_results_per_group_df
    
    @best_cluster_results_per_group_df.setter
    def best_cluster_results_per_group_df(self, 
                                          best_cluster_results_per_group_df: pd.DataFrame):

        if isinstance(best_cluster_results_per_group_df, pd.DataFrame):
            self._best_cluster_results_per_group_df = best_cluster_results_per_group_df.copy()

        else:
            self._best_cluster_results_per_group_df = None 

    @property
    def sequence_cluster_per_group_df(self):
        return self._sequence_cluster_per_group_df
    
    @sequence_cluster_per_group_df.setter
    def sequence_cluster_per_group_df(self, 
                                      sequence_cluster_per_group_df: pd.DataFrame):

        if isinstance(sequence_cluster_per_group_df, pd.DataFrame):
            self._sequence_cluster_per_group_df = sequence_cluster_per_group_df.copy()

        else:
            self._sequence_cluster_per_group_df = None 

    @property
    def omnibus_test_result_df(self):
        return self._omnibus_test_result_df
    
    @omnibus_test_result_df.setter
    def omnibus_test_result_df(self, 
                               omnibus_test_result_df: pd.DataFrame):

        if isinstance(omnibus_test_result_df, pd.DataFrame):
            self._omnibus_test_result_df = omnibus_test_result_df.copy()

        else:
            self._omnibus_test_result_df = None 

    @property
    def measure_association_conf_int_bootstrap_failures_df(self):
        return self._measure_association_conf_int_bootstrap_failures_df
    
    @measure_association_conf_int_bootstrap_failures_df.setter
    def measure_association_conf_int_bootstrap_failures_df(self, 
                                                           measure_association_conf_int_bootstrap_failures_df: pd.DataFrame):

        if isinstance(measure_association_conf_int_bootstrap_failures_df, pd.DataFrame):
            self._measure_association_conf_int_bootstrap_failures_df = measure_association_conf_int_bootstrap_failures_df.copy()

        else:
            self._measure_association_conf_int_bootstrap_failures_df = None 

    @property
    def object_size_df(self):
        return self._object_size_df
    
    @object_size_df.setter
    def object_size_df(self, 
                       object_size_df: pd.DataFrame):

        if isinstance(object_size_df, pd.DataFrame):
            self._object_size_df = self._transform_object_size_df(object_size_df.copy())

        else:
            self._object_size_df = None 

    def _gen_group_list(self) -> List[int]:
        """Returns a list containing the group labels which are integers.

        Returns
        -------
        List
            A list containing the group labels which are integers
        """        
        group_list = np.unique(self.interactions[GROUP_FIELD_NAME_STR]).tolist()

        return group_list

    def _gen_available_fields_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains information about what fields are available in the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing field availability information about the input interactions dataframe
        """        
        available_fields_df = self.interactions.head(1).notna().transpose().rename(columns={0:  self.dataset_name})
        available_fields_df.columns = pd.MultiIndex.from_product([[ResultTables.is_available_str], available_fields_df.columns])
        available_fields_df = available_fields_df.reset_index().rename(columns={'index': ResultTables.field_str})

        return available_fields_df

    def _gen_summary_statistics_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains summary statistics of the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing summary statistics of the input interactions dataframe
        """        
        
        n_rows = int(self.interactions.shape[0])
        n_unique_users = int(self.interactions[USER_FIELD_NAME_STR].nunique())
        n_unique_learning_activities = int(self.interactions[LEARNING_ACTIVITY_FIELD_NAME_STR].nunique())

        if self.group_field:
            n_unique_groups = int(self.interactions[GROUP_FIELD_NAME_STR].nunique())
            sparsity_user_learning_activity_matrix = round(calculate_sparsity(self.interactions[USER_FIELD_NAME_STR],
                                                                              self.interactions[LEARNING_ACTIVITY_FIELD_NAME_STR]), 
                                                           2)
            sparsity_user_group_matrix = round(calculate_sparsity(self.interactions[USER_FIELD_NAME_STR],
                                                                  self.interactions[GROUP_FIELD_NAME_STR]),
                                                2)

        else:
            n_unique_groups = None
            sparsity_user_learning_activity_matrix = round(calculate_sparsity(self.interactions[USER_FIELD_NAME_STR],
                                                                              self.interactions[LEARNING_ACTIVITY_FIELD_NAME_STR]),
                                                           2)
            sparsity_user_group_matrix = None

        idx = [ResultTables.n_rows_str, 
               ResultTables.n_unique_users_str, 
               ResultTables.n_unique_groups_str, 
               ResultTables.n_unique_learning_activities_str, 
               ResultTables.sparsity_user_learning_activity_matrix_str, 
               ResultTables.sparsity_user_group_matrix_str]
        data = [n_rows, 
                n_unique_users, 
                n_unique_groups, 
                n_unique_learning_activities, 
                sparsity_user_learning_activity_matrix, 
                sparsity_user_group_matrix]
        data_dict = {self.dataset_name: data}
        summary_statistics_df = pd.DataFrame(data_dict, index=idx)

        return summary_statistics_df

    def _gen_sequence_statistics_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains sequence statistics of the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing sequence statistics of the input interactions dataframe
        """        

        if self.group_field:
            n_sequences = int(self.interactions.groupby([USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR]).ngroups)
            n_unique_sequences = int(self.interactions.groupby([USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])\
                                                      [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                       .agg(lambda x: tuple(x.to_list())).nunique())
            mean_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).mean(), 2)
            median_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).median(), 2)
            std_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).std(), 2)
            iqr_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).quantile(0.5), 2)

        else:
            n_sequences = int(self.interactions.groupby([USER_FIELD_NAME_STR]).ngroups)
            n_unique_sequences = int(self.interactions.groupby([USER_FIELD_NAME_STR])\
                                                      [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                      .agg(lambda x: tuple(x.to_list())).nunique())
            mean_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).mean(), 2)
            median_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).median(), 2)
            std_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).std(), 2)
            iqr_sequence_length = round(self.interactions.groupby([USER_FIELD_NAME_STR])\
                                                    [LEARNING_ACTIVITY_FIELD_NAME_STR]\
                                                    .agg(lambda x: len(x)).quantile(0.5), 2)
        idx = [ResultTables.n_sequences_str, 
               ResultTables.n_unique_sequences_str, 
               ResultTables.mean_sequence_length_str,
               ResultTables.median_sequence_length_str,
               ResultTables.std_sequence_length_str,
               ResultTables.iqr_sequence_length_str]
        data = [n_sequences, 
                n_unique_sequences, 
                mean_sequence_length,
                median_sequence_length,
                std_sequence_length,
                iqr_sequence_length]
        data_dict = {self.dataset_name: data}
        sequence_statistics_df = pd.DataFrame(data_dict, index=idx)

        return sequence_statistics_df

    def _gen_score_is_correct_rel_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains information about the relationship between the score and the is_correct fields in the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing information about the relationship between the score and the is_correct fields in the input interactions dataframe
        """
        # helper function
        def change_int_to_float(x):
            if type(x)==int:
                x = float(x)
            return x

        data_dict = {}
        for i,j in self.eval_score_ranges_list:

            has_field = j[0] != None
            has_score_field = j[1] != None
            has_is_correct_field = j[2] != None
            chosen_score_correct_threshold = j[3]
            score_minimum_docu = j[4]
            score_maximum_docu = j[5]
            if i[1]['eval_score_ranges'] is not None:
                score_minimum_data = float(i[1]['eval_score_ranges']['score_minimum'].min())
                score_maximum_data = float(i[1]['eval_score_ranges']['score_maximum'].max())
                are_equal_all_score_minima = i[1]['eval_score_ranges']['score_minimum'].nunique()==1
                are_equal_all_score_maxima = i[1]['eval_score_ranges']['score_maximum'].nunique()==1
                
                field_value_list = [has_field, 
                                    has_score_field, 
                                    has_is_correct_field, 
                                    chosen_score_correct_threshold, 
                                    score_minimum_docu, 
                                    score_maximum_docu, 
                                    are_equal_all_score_minima, 
                                    are_equal_all_score_maxima, 
                                    score_minimum_data, 
                                    score_maximum_data]
                field_value_list = [change_int_to_float(i) for i in field_value_list]
                data_dict[i[0]] = field_value_list 
            else:
                field_value_list = [has_field, 
                                    has_score_field, 
                                    has_is_correct_field, 
                                    chosen_score_correct_threshold, 
                                    score_minimum_docu, 
                                    score_maximum_docu, 
                                    None, 
                                    None, 
                                    None, 
                                    None]
                field_value_list = [change_int_to_float(i) for i in field_value_list]
                data_dict[i[0]] = field_value_list

        idx = [ResultTables.has_field_str, 
               ResultTables.has_score_field_str, 
               ResultTables.has_is_correct_field_str, 
               ResultTables.chosen_score_correct_threshold_str, 
               ResultTables.score_minimum_docu_str, 
               ResultTables.score_maximum_docu_str, 
               ResultTables.are_equal_all_score_minima_str, 
               ResultTables.are_equal_all_score_maxima_str, 
               ResultTables.score_minimum_data_str, 
               ResultTables.score_maximum_data_str]
        score_is_correct_rel_df = pd.DataFrame(data_dict, index=idx).fillna(np.nan)
        score_is_correct_rel_df.columns = pd.MultiIndex.from_product([[self.dataset_name], score_is_correct_rel_df.columns])

        return score_is_correct_rel_df

    def _transform_object_size_df(self,
                                  object_size_df) -> pd.DataFrame:
        """Generate the object size dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing object size information
        """        
        object_size_df = object_size_df.set_index(MONITORING_OBJECT_NAME_FIELD_NAME_STR)
        object_size_df.index.name = None
        object_size_df = object_size_df.pivot(columns=DATASET_NAME_FIELD_NAME_STR,
                                              values=MONITORING_MEGABYTE_FIELD_NAME_STR)
        object_size_df.columns.name = None
        object_size_df.columns = pd.MultiIndex.from_product([[MONITORING_MEGABYTE_FIELD_NAME_STR], object_size_df.columns])
        object_size_df = object_size_df.loc[MONITORING_RESULT_TABLES_LIST]
        object_size_df = object_size_df.reset_index().rename(columns={'index': MONITORING_OBJECT_NAME_FIELD_NAME_STR})

        return object_size_df

    ############################################################ 
    # result information
    ############################################################ 

    def display_available_fields(self) -> None:
        """Displays the available fields html table.
        """
        try:
            available_fields_df = self.available_fields_df.copy()
            available_fields_df = available_fields_df.set_index(FIELD_STR)

            soup = BeautifulSoup(available_fields_df.to_html(), "html.parser")

            # move Field header value to correct position
            th_elements = soup.select('thead th')
            for th in th_elements[::-1]:
                if th.string == FIELD_STR:
                    th.string = ''
            th = soup.select_one('thead th')
            if th:
                th.append(FIELD_STR)
            
            remove_empty_html_table_row(soup)
            
            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_INTERACTIONS_ERROR_STR)
    
    def display_summary_statistics(self) -> None:
        """Displays the summary statistics html table.
        """
        try:
            # typecast fields to int
            summary_statistics_df = self.summary_statistics_df.copy().transpose()

            idx = [ResultTables.n_rows_str, 
                   ResultTables.n_unique_users_str, 
                   ResultTables.n_unique_groups_str, 
                   ResultTables.n_unique_learning_activities_str]

            typecast_dict = {i: 'int' for i in idx if summary_statistics_df[i].notna()[0]}
            summary_statistics_df = summary_statistics_df.astype(typecast_dict)

            soup = BeautifulSoup(summary_statistics_df.to_html(), "html.parser")

            # standardized table formatting
            apply_html_table_formatting(soup)
            
            display(Markdown(soup.prettify()))
        except:
            print(MISSING_INTERACTIONS_ERROR_STR)

    def display_sequence_statistics(self) -> None:
        """Displays the sequence statistics html table.
        """
        try:
            # typecast fields to int
            sequence_statistics_df = self.sequence_statistics_df.transpose()

            idx = [ResultTables.n_sequences_str, 
                   ResultTables.n_unique_sequences_str]

            typecast_dict = {i: 'int' for i in idx if sequence_statistics_df[i].notna()[0]}
            sequence_statistics_df = sequence_statistics_df.astype(typecast_dict)

            soup = BeautifulSoup(sequence_statistics_df.to_html(), "html.parser")

            # standardized table formatting
            apply_html_table_formatting(soup)
            
            display(Markdown(soup.prettify()))
        except:
            print(MISSING_INTERACTIONS_ERROR_STR)
    
    ############################################################ 
    # preprocessing information
    ############################################################ 

    def display_interaction_types(self) -> None:
        """Displays the interaction_type html table.
        """
        try:
            interaction_types = self.interaction_types.copy()
            interaction_types.index = pd.MultiIndex.from_product([[self.dataset_name], ['']*len(interaction_types)])
            interaction_types = interaction_types.drop(labels=[DATASET_NAME_STR], axis=1)

            soup = BeautifulSoup(interaction_types.to_html(), "html.parser")

            # remove empty cells
            tr_elements = soup.find_all('tr')
            for tr in tr_elements:
                th = tr.find('th', rowspan=None) 
                if th:
                    th.extract()

            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_INTERACTION_TYPES_ERROR_STR)

    def display_score_is_correct_relationship(self) -> None:
        """Displays the score_is_correct_relationship html table.
        """
        try:
            score_is_correct_rel_df = self.score_is_correct_rel_df.copy().transpose()

            soup = BeautifulSoup(score_is_correct_rel_df.to_html(), "html.parser")

            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_EVALUATION_SCORE_RANGE_LIST_ERROR_STR)

    def display_seq_filter_parameters(self) -> None:
        """Displays the seq_filter_parameters html table.
        """
        try:
            seq_filter_parameters = self.seq_filter_parameters.copy()
            seq_filter_parameters = seq_filter_parameters.set_index(DATASET_NAME_FIELD_NAME_STR)
            seq_filter_parameters.index.name = None

            soup = BeautifulSoup(seq_filter_parameters.to_html(), "html.parser")

            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_SEQ_FILTER_PARAMETERS_ERROR_STR)

    def display_seq_filter_seq_count_changes(self) -> None:
        """Displays the seq_filter_seq_count html table.
        """
        try:
            # typecast fields to int
            seq_filter_sequence_count_per_group = self.seq_filter_sequence_count_per_group.copy()
            datatype_mapping_dict = {'int64': 'Int64', 'float64': 'Int64'}
            seq_filter_sequence_count_per_group = change_data_type(seq_filter_sequence_count_per_group,
                                                                  datatype_mapping_dict)

            for group, df in seq_filter_sequence_count_per_group.groupby(LEARNING_ACTIVITY_SEQUENCE_COUNT_TYPE_NAME_STR):

                df.index = pd.MultiIndex.from_product([[group], [self.dataset_name], ['']*len(df)])
                df = df.drop(labels=[DATASET_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_COUNT_TYPE_NAME_STR], axis=1)

                soup = BeautifulSoup(df.to_html(), "html.parser")

                # remove empty cells
                remove_empty_html_table_column(soup)
                            
                # standardized table formatting
                apply_html_table_formatting(soup)

                display(Markdown(soup.prettify()))
        except:
            print(MISSING_SEQ_FILTER_SEQ_CHANGES_ERROR_STR)

    def display_seq_filter_group_count_changes(self) -> None:
        """Displays the seq_filter_group_count html table.
        """
        try:
            seq_filter_group_count = self.seq_filter_group_count.copy()
            seq_filter_group_count.index = pd.MultiIndex.from_product([[LEARNING_ACTIVITY_SEQUENCE_PREPROCESS_GROUP_COUNT_NAME_STR], [self.dataset_name], ['']*len(seq_filter_group_count)])
            seq_filter_group_count = seq_filter_group_count.drop(labels=[DATASET_NAME_STR], axis=1)

            soup = BeautifulSoup(seq_filter_group_count.to_html(), "html.parser")

            # remove empty cells
            remove_empty_html_table_column(soup)

            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_SEQ_FILTER_GROUP_CHANGES_ERROR_STR)

    def display_seq_stats_sequence_count_per_group(self) -> None:
        """Displays the seq_stats_sequence_count_per_group html table.
        """
        try:
            seq_stats_sequence_count_per_group = self.seq_stats_sequence_count_per_group.copy()
            seq_stats_sequence_count_per_group.index = pd.MultiIndex.from_product([[self.dataset_name], ['']*len(seq_stats_sequence_count_per_group)])
            seq_stats_sequence_count_per_group = seq_stats_sequence_count_per_group.drop(labels=[DATASET_NAME_STR], axis=1)

            soup = BeautifulSoup(seq_stats_sequence_count_per_group.to_html(), "html.parser")

            # remove empty cells
            remove_empty_html_table_column(soup)
                        
            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_SEQ_STATS_SEQ_COUNT_ERROR_STR)

    ############################################################ 
    # monitoring information
    ############################################################ 

    def display_object_sizes(self) -> None:
        """Displays the object size html table.
        """
        try:
            object_size_df = self.object_size_df.copy()
            object_size_df = object_size_df.set_index(MONITORING_OBJECT_NAME_FIELD_NAME_STR)

            soup = BeautifulSoup(object_size_df.to_html(), "html.parser")

            # move Field header value to correct position
            th_elements = soup.select('thead th')
            for th in th_elements[::-1]:
                if th.string == MONITORING_OBJECT_NAME_FIELD_NAME_STR:
                    th.string = ''
            th = soup.select_one('thead th')
            if th:
                th.append(MONITORING_OBJECT_NAME_FIELD_NAME_STR)
            
            remove_empty_html_table_row(soup)
            
            # standardized table formatting
            apply_html_table_formatting(soup)

            display(Markdown(soup.prettify()))
        except:
            print(MISSING_OBJECT_SIZE_ERROR_STR)