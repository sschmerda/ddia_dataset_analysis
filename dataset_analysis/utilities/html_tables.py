from .standard_import import *
from .functions import *
from .constants import *

class HtmlTables():
    """A class which upon initialization holds data about the available fields, the score is_correct field relationship and summary statistics of 
    the input interactions dataframe.
    This data can be displayed as html tables with the corresponding display_* methods.

    Parameters
    ----------
    dataset_name : str
        The name of the input dataframe
    interactions : pd.DataFrame
        The interactions dataframe
    evaluation_score_range_dict : dict
        A dictionary containing score ranges of evaluation elements on the learning_activity, group and course level.
    html_tables_data_list : list
        A list containing data about evaluation fields at the learning activity, group and course level.
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
                 dataset_name, 
                 interactions,
                 evaluation_score_range_dict,
                 html_tables_data_list): 

        self.dataset_name = dataset_name
        self.interactions = interactions
        self.evaluation_score_range_dict = evaluation_score_range_dict
        self.html_tables_data_list = html_tables_data_list

        # calculated upon initialization
        self.available_fields_df = self._gen_available_fields_df()
        self.score_is_correct_rel_df = self._gen_score_is_correct_rel_df()
        self.summary_statistics_df = self._gen_summary_statistics_df()
        self.sequence_statistics_df = self._gen_sequence_statistics_df()

        # set interactions to None in order to reduce object size
        self.interactions = None

    def _gen_available_fields_df(self):
        """Returns a dataframe which contains information about what fields are available in the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing field availability information about the input interactions dataframe
        """        
        available_fields_df = self.interactions.head(1).notna().transpose().rename(columns={0:  self.dataset_name})
        available_fields_df.columns = pd.MultiIndex.from_product([[HtmlTables.is_available_str], available_fields_df.columns])
        available_fields_df = available_fields_df.reset_index().rename(columns={'index': HtmlTables.field_str})

        return available_fields_df

    def _gen_score_is_correct_rel_df(self):
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
        for i,j in zip(self.evaluation_score_range_dict.items(), self.html_tables_data_list):

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

        idx = [HtmlTables.has_field_str, 
               HtmlTables.has_score_field_str, 
               HtmlTables.has_is_correct_field_str, 
               HtmlTables.chosen_score_correct_threshold_str, 
               HtmlTables.score_minimum_docu_str, 
               HtmlTables.score_maximum_docu_str, 
               HtmlTables.are_equal_all_score_minima_str, 
               HtmlTables.are_equal_all_score_maxima_str, 
               HtmlTables.score_minimum_data_str, 
               HtmlTables.score_maximum_data_str]
        score_is_correct_rel_df = pd.DataFrame(data_dict, index=idx).fillna(np.nan)
        score_is_correct_rel_df.columns = pd.MultiIndex.from_product([[self.dataset_name], score_is_correct_rel_df.columns])

        return score_is_correct_rel_df

    def _gen_summary_statistics_df(self):
        """Returns a dataframe which contains summary statistics of the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing summary statistics of the input interactions dataframe
        """        
        
        n_rows = int(self.interactions.shape[0])
        n_unique_users = int(self.interactions[USER_FIELD_NAME_STR].nunique())
        n_unique_learning_activities = int(self.interactions[LEARNING_ACTIVITY_FIELD_NAME_STR].nunique())

        if self.html_tables_data_list[1][0]:
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

        idx = [HtmlTables.n_rows_str, 
               HtmlTables.n_unique_users_str, 
               HtmlTables.n_unique_groups_str, 
               HtmlTables.n_unique_learning_activities_str, 
               HtmlTables.sparsity_user_learning_activity_matrix_str, 
               HtmlTables.sparsity_user_group_matrix_str]
        data = [n_rows, 
                n_unique_users, 
                n_unique_groups, 
                n_unique_learning_activities, 
                sparsity_user_learning_activity_matrix, 
                sparsity_user_group_matrix]
        data_dict = {self.dataset_name: data}
        summary_statistics_df = pd.DataFrame(data_dict, index=idx)

        return summary_statistics_df

    def _gen_sequence_statistics_df(self):
        """Returns a dataframe which contains sequence statistics of the input interactions dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing sequence statistics of the input interactions dataframe
        """        

        if self.html_tables_data_list[1][0]:
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
        idx = [HtmlTables.n_sequences_str, 
               HtmlTables.n_unique_sequences_str, 
               HtmlTables.mean_sequence_length_str,
               HtmlTables.median_sequence_length_str,
               HtmlTables.std_sequence_length_str,
               HtmlTables.iqr_sequence_length_str]
        data = [n_sequences, 
                n_unique_sequences, 
                mean_sequence_length,
                median_sequence_length,
                std_sequence_length,
                iqr_sequence_length]
        data_dict = {self.dataset_name: data}
        sequence_statistics_df = pd.DataFrame(data_dict, index=idx)

        return sequence_statistics_df

    def display_available_fields(self):
        """Displays the available fields html table.
        """
        available_fields_df = self.available_fields_df.to_html(notebook=False, index=False, sparsify=True)
        for i in self.available_fields_df.iloc[:, 0]:
            available_fields_df = available_fields_df.replace(f'<td>{i}</td>', f'<td style = "background-color: rgb(80, 80, 80); color: white">{i}</td>')

        # available_fields_df = available_fields_df.replace(f'<th>{HtmlTables.field_str}</th>', '<th> </th>')
        # available_fields_df = available_fields_df.replace('<th></th>', f'<th>{HtmlTables.field_str}</th>')
        available_fields_df = available_fields_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        available_fields_df = available_fields_df.replace('<td>True</td>', '<td style = "background-color: green; color: white; text-align:center">True</td>')
        available_fields_df = available_fields_df.replace('<td>False</td>', '<td style = "background-color: red; color: white; text-align:center">False</td>')
        available_fields_df = available_fields_df.replace(f'<th colspan="3" halign="left">{HtmlTables.is_available_str}</th>', f'<th colspan="3" halign="left", style = "background-color: royalblue; color: white; text-align:center">{HtmlTables.is_available_str}</th>')

        display(Markdown(available_fields_df))
    
    def display_score_is_correct_relationship(self):
        """Displays the score_is_correct_relationship html table.
        """
        score_is_correct_rel_df = self.score_is_correct_rel_df.transpose().to_html(notebook=False, index=True, sparsify=True)
        score_is_correct_rel_df = score_is_correct_rel_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        score_is_correct_rel_df = score_is_correct_rel_df.replace(f'<th rowspan="3" valign="top">{self.dataset_name}</th>', f'<th rowspan="3" valign="top" style = "background-color: rgb(80, 80, 80); color: white">{self.dataset_name}</th>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace('<td>True</td>', '<td style = "background-color: green; color: white; text-align:center">True</td>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace('<td>False</td>', '<td style = "background-color: red; color: white; text-align:center">False</td>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace(f'<td>NaN</td>', '<td style = "text-align:center">-</td>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace('<td>', '<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">')
        score_is_correct_rel_df = score_is_correct_rel_df.replace(f'<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">{np.nan}</td>', f'<td>{np.nan}</td>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{LEARNING_ACTIVITY_FIELD_NAME_STR}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{LEARNING_ACTIVITY_FIELD_NAME_STR}</th>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{GROUP_FIELD_NAME_STR}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{GROUP_FIELD_NAME_STR}</th>')
        score_is_correct_rel_df = score_is_correct_rel_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{COURSE_FIELD_NAME_STR}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{COURSE_FIELD_NAME_STR}</th>')

        display(Markdown(score_is_correct_rel_df))
    
    def display_summary_statistics(self):
        """Displays the summary statistics html table.
        """
        # typecast fields to int
        summary_statistics_df = self.summary_statistics_df.transpose()

        idx = [HtmlTables.n_rows_str, 
               HtmlTables.n_unique_users_str, 
               HtmlTables.n_unique_groups_str, 
               HtmlTables.n_unique_learning_activities_str]

        typecast_dict = {i: 'int' for i in idx if summary_statistics_df[i].notna()[0]}
        summary_statistics_df = summary_statistics_df.astype(typecast_dict)

        summary_statistics_df = summary_statistics_df.to_html(notebook=False, index=True, sparsify=True)
        summary_statistics_df = summary_statistics_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        summary_statistics_df = summary_statistics_df.replace('<td>', '<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">')
        summary_statistics_df = summary_statistics_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{self.dataset_name}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{self.dataset_name}</th>')
        summary_statistics_df = summary_statistics_df.replace(f'NaN', f'-')

        display(Markdown(summary_statistics_df))

    def display_sequence_statistics(self):
        """Displays the sequence statistics html table.
        """
        # typecast fields to int
        sequence_statistics_df = self.sequence_statistics_df.transpose()

        idx = [HtmlTables.n_sequences_str, 
               HtmlTables.n_unique_sequences_str]

        typecast_dict = {i: 'int' for i in idx if sequence_statistics_df[i].notna()[0]}
        sequence_statistics_df = sequence_statistics_df.astype(typecast_dict)

        sequence_statistics_df = sequence_statistics_df.to_html(notebook=False, index=True, sparsify=True)
        sequence_statistics_df = sequence_statistics_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        sequence_statistics_df = sequence_statistics_df.replace('<td>', '<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">')
        sequence_statistics_df = sequence_statistics_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{self.dataset_name}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{self.dataset_name}</th>')
        sequence_statistics_df = sequence_statistics_df.replace(f'NaN', f'-')

        display(Markdown(sequence_statistics_df))