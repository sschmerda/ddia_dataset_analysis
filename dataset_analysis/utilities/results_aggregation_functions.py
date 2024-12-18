from .standard_import import *
from .constants.constants import *
from .configs.result_aggregation_config import *
from .plotting_functions import *

class AggregatedResultTables():
    """A class which upon initialization holds data about summary statistics, sequence summary statistics, the available fields and the score is_correct relationship of 
    the analysed datasets.
    """    
    # class vars
    path_to_html_table_dir = PATH_TO_PICKLED_OBJECTS_FOLDER + PATH_TO_RESULT_TABLES_PICKLE_FOLDER
    n_rows_str = N_ROWS_STR
    n_unique_users_str = N_UNIQUE_USERS_STR
    n_unique_learning_activities_str = N_UNIQUE_LEARNING_ACTIVITIES_STR
    n_unique_groups_str = N_UNIQUE_GROUPS_STR
    sparsity_user_learning_activity_matrix_str = SPARSITY_USER_LEARNING_ACTIVITY_MATRIX_STR
    sparsity_user_group_matrix_str = SPARSITY_USER_GROUP_MATRIX_STR
    n_sequences_str = N_SEQUENCES_STR
    n_unique_sequences_str = N_UNIQUE_SEQUENCES_STR
    is_available_str = IS_AVAILABLE_STR
    field_str = FIELD_STR
    dataset_name_str = DATASET_NAME_STR

    path_to_sequence_distances_analytics_dir = PATH_TO_PICKLED_OBJECTS_FOLDER + PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_PICKLE_FOLDER

    def __init__(self):

        # calculated upon initialization
        # html table for each dataset
        self.html_tables_dict = self._load_html_tables()
        self.avg_sequence_distance_per_group_agg_df, self.avg_unique_sequence_distance_per_group_agg_df  = self._load_sequence_distance_analytics()

        # result dataframes
        self.available_fields_result_df = self._gen_available_fields_result_df()
        self.summary_statistics_result_df = self._gen_summary_statistics_result_df()
        self.sequence_summary_statistics_result_df = self._gen_sequence_summary_statistics_result_df()
        self.score_is_correct_relationship_result_df = self._gen_score_is_correct_relationship_result_df()


    
    def _load_html_tables(self):
        """Returns a dictionary containing the html table objects of the analysed datasets

        Returns
        -------
        pd.DataFrame
            A dictionary containing the html table objects of the analysed datasets
        """        
        html_table_dir = AggregatedResultTables.path_to_html_table_dir
        html_tables_list = glob.glob(html_table_dir + '*.pickle')

        html_tables_dict = {}
        for i in html_tables_list:
            with open(i, 'rb') as f:
                html_table = pickle.load(f)
                dataset_name = html_table.dataset_name
            html_tables_dict[dataset_name] = html_table
        
        return html_tables_dict

    def _load_sequence_distance_analytics(self):
        sequence_distances_analytics_dir = AggregatedResultTables.path_to_sequence_distances_analytics_dir
        sequence_distances_analytics_list = glob.glob(sequence_distances_analytics_dir + '*.pickle')

        avg_sequence_distance_per_group_agg_df = pd.DataFrame()
        avg_unique_sequence_distance_per_group_agg_df = pd.DataFrame()
        for i in sequence_distances_analytics_list:
            with open(i, 'rb') as f:
                sequence_distance_analytics = pickle.load(f)
            
            avg_sequence_distance_per_group_df = sequence_distance_analytics.avg_sequence_distance_per_group_df
            avg_unique_sequence_distance_per_group_df = sequence_distance_analytics.avg_unique_sequence_distance_per_group_df
        
        avg_sequence_distance_per_group_agg_df = pd.concat([avg_sequence_distance_per_group_agg_df, 
                                                            avg_sequence_distance_per_group_df])
        avg_unique_sequence_distance_per_group_agg_df = pd.concat([avg_unique_sequence_distance_per_group_agg_df, 
                                                                   avg_unique_sequence_distance_per_group_df])
        
        return avg_sequence_distance_per_group_agg_df, avg_unique_sequence_distance_per_group_agg_df
            
    def _gen_summary_statistics_result_df(self):
        """Returns a dataframe which contains summary statistics of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing summary statistics of the analysed datasets
        """        
        summary_statistics_dict = {}
        for dataset_name, html_table in self.html_tables_dict.items():
            summary_statistics_df = html_table.summary_statistics_df
            summary_statistics_dict[dataset_name] = summary_statistics_df

        summary_statistics_dfs = [v.transpose() for k,v in sorted(summary_statistics_dict.items(), key=lambda x: x[0].lower())]
        summary_statistics_result_df = pd.concat(summary_statistics_dfs, join='inner', axis=0)

        # typecast to integer type which also can take on NAs
        idx = [AggregatedResultTables.n_rows_str,
               AggregatedResultTables.n_unique_users_str, 
               AggregatedResultTables.n_unique_groups_str, 
               AggregatedResultTables.n_unique_learning_activities_str]
        typecast_dict = {i: 'Int64' for i in idx}
        summary_statistics_result_df = summary_statistics_result_df.astype(typecast_dict)
        summary_statistics_result_df = summary_statistics_result_df.reset_index(names=AggregatedResultTables.dataset_name_str)

        return summary_statistics_result_df

    def _gen_sequence_summary_statistics_result_df(self):
        """Returns a dataframe which contains sequence summary statistics of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing sequence summary statistics of the analysed datasets
        """        
        sequence_summary_statistics_dict = {}
        for dataset_name, html_table in self.html_tables_dict.items():
            sequence_summary_statistics_df = html_table.sequence_statistics_df
            sequence_summary_statistics_dict[dataset_name] = sequence_summary_statistics_df

        summary_statistics_dfs = [v.transpose() for k,v in sorted(sequence_summary_statistics_dict.items(), key=lambda x: x[0].lower())]
        sequence_summary_statistics_result_df = pd.concat(summary_statistics_dfs, join='inner', axis=0)

        # typecast to integer type which also can take on NAs
        idx = [AggregatedResultTables.n_sequences_str,
               AggregatedResultTables.n_unique_sequences_str]
        typecast_dict = {i: 'Int64' for i in idx}
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.astype(typecast_dict)
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.reset_index(names=self.dataset_name_str)

        return sequence_summary_statistics_result_df

    def _gen_available_fields_result_df(self):
        """Returns a dataframe which contains information about field availability of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing field availability information about the analysed datasets
        """        
        available_fields_dict = {}
        for dataset_name, html_table in self.html_tables_dict.items():
            available_fields_result_df = html_table.available_fields_df
            available_fields_dict[dataset_name] = available_fields_result_df

        def set_index(df: pd.DataFrame) -> pd.DataFrame:
            df = df.set_index(self.field_str)
            df.columns = df.columns.droplevel(0)
            return df

        available_fields_dfs = [v for k,v in sorted(available_fields_dict.items(), key=lambda x: x[0].lower())]
        available_fields_dfs = map(set_index, available_fields_dfs)
        available_fields_result_df = pd.concat(available_fields_dfs, join='inner', axis=1)
        available_fields_result_df.columns = pd.MultiIndex.from_product([[AggregatedResultTables.is_available_str], available_fields_result_df.columns])
        available_fields_result_df = available_fields_result_df.reset_index()

        return available_fields_result_df

    def _gen_score_is_correct_relationship_result_df(self):
        """Returns a dataframe which contains information about the relationship between the score and the is_correct fields of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing information about the relationship between the score and the is_correct fields of the analysed datasets.
        """        
        score_is_correct_relationship_dict = {}
        for dataset_name, html_table in self.html_tables_dict.items():
            score_is_correct_relationship_df = html_table.score_is_correct_rel_df
            score_is_correct_relationship_dict[dataset_name] = score_is_correct_relationship_df

        score_is_correct_relationship_dfs = [v.transpose() for k,v in sorted(score_is_correct_relationship_dict.items(), key=lambda x: x[0].lower())]
        score_is_correct_relationship_result_df = pd.concat(score_is_correct_relationship_dfs, join='inner', axis=0)

        score_is_correct_relationship_result_df = score_is_correct_relationship_result_df.reset_index(names=[self.dataset_name_str, self.field_str])

        return score_is_correct_relationship_result_df

    def display_summary_statistics_result(self):
        """Displays the summary statistics result html table.
        """
        summary_statistics_result_df = self.summary_statistics_result_df
        summary_statistics_result_df = summary_statistics_result_df.set_index(self.dataset_name_str)
        summary_statistics_result_df.index.name = None
        dataset_names = summary_statistics_result_df.index
        summary_statistics_result_df = summary_statistics_result_df.to_html(notebook=False, 
                                                                            index=True, 
                                                                            sparsify=True)
        summary_statistics_result_df = summary_statistics_result_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        summary_statistics_result_df = summary_statistics_result_df.replace('<td>', '<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">')
        for dataset_name in dataset_names:
            summary_statistics_result_df = summary_statistics_result_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{dataset_name}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{dataset_name}</th>')
        summary_statistics_result_df = summary_statistics_result_df.replace(f'<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">&lt;NA&gt;</td>', f'<td style = "color: white; text-align:center">-</td>')
        summary_statistics_result_df = summary_statistics_result_df.replace(f'<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">NaN</td>', f'<td style = "color: white; text-align:center">-</td>')

        display(Markdown(summary_statistics_result_df))

    def display_sequence_summary_statistics_result(self):
        """Displays the sequence summary statistics result html table.
        """
        sequence_summary_statistics_result_df = self.sequence_summary_statistics_result_df
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.set_index(self.dataset_name_str)
        sequence_summary_statistics_result_df.index.name = None
        dataset_names = sequence_summary_statistics_result_df.index
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.to_html(notebook=False, 
                                                                            index=True, 
                                                                            sparsify=True)
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.replace('<td>', '<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">')
        for dataset_name in dataset_names:
            sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{dataset_name}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{dataset_name}</th>')
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.replace(f'<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">&lt;NA&gt;</td>', f'<td style = "color: white; text-align:center">-</td>')
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.replace(f'<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">NaN</td>', f'<td style = "color: white; text-align:center">-</td>')
        display(Markdown(sequence_summary_statistics_result_df))


    def display_available_fields_result(self):
        """Displays the available fields result html table.
        """
        available_fields_result_df = self.available_fields_result_df.to_html(notebook=False, 
                                                                             index=False, 
                                                                             sparsify=True)

        for i in self.available_fields_result_df.iloc[:, 0]:
            available_fields_result_df = available_fields_result_df.replace(f'<td>{i}</td>', f'<td style = "background-color: rgb(80, 80, 80); color: white">{i}</td>')

        # available_fields_result_df = available_fields_result_df.replace(f'<th>{ResultTables.field_str}</th>', '<th> </th>')
        # available_fields_result_df = available_fields_result_df.replace('<th></th>', f'<th>{ResultTables.field_str}</th>')
        available_fields_result_df = available_fields_result_df.replace('<th colspan="4" halign="left">', '<th colspan="4" halign="left" style = "background-color: royalblue; color: white; text-align:center">')
        available_fields_result_df = available_fields_result_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        available_fields_result_df = available_fields_result_df.replace('<td>True</td>', '<td style = "background-color: green; color: white; text-align:center">True</td>')
        available_fields_result_df = available_fields_result_df.replace('<td>False</td>', '<td style = "background-color: red; color: white; text-align:center">False</td>')
        available_fields_result_df = available_fields_result_df.replace(f'<th colspan="3" halign="left">{AggregatedResultTables.is_available_str}</th>', f'<th colspan="3" halign="left", style = "background-color: royalblue; color: white; text-align:center">{AggregatedResultTables.is_available_str}</th>')

        display(Markdown(available_fields_result_df))

    def display_score_is_correct_relationship_result(self):
        """Displays the sequence summary statistics result html table.
        """
        score_is_corrcet_relationship_result_df = self.score_is_correct_relationship_result_df
        dataset_names = score_is_corrcet_relationship_result_df[self.dataset_name_str]
        field_names = score_is_corrcet_relationship_result_df[self.field_str]
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.set_index([self.dataset_name_str, self.field_str])
        score_is_corrcet_relationship_result_df.index.names = [None, None]
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.to_html(notebook=False, 
                                                                                                  index=True, 
                                                                                                  sparsify=True)
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace('<th rowspan="3" valign="top">', '<th rowspan="3" valign="top" style = "background-color: royalblue; color: white; text-align:center">')
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace('<th>', '<th style = "background-color: royalblue; color: white; text-align:center">')
        for dataset_name in dataset_names:
            score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace(f'<th rowspan="3" valign="top" style = "background-color: royalblue; color: white; text-align:center">{dataset_name}</th>', f'<th rowspan="3" valign="top" style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{dataset_name}</th>')
        for field_name in field_names: 
            score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace(f'<th style = "background-color: royalblue; color: white; text-align:center">{field_name}</th>', f'<th style = "background-color: rgb(80, 80, 80); color: white; text-align:center">{field_name}</th>')
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace('<td>True</td>', '<td style = "background-color: green; color: white; text-align:center">True</td>')
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace('<td>False</td>', '<td style = "background-color: red; color: white; text-align:center">False</td>')
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace('<td>', '<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">')
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.replace(f'<td style = "background-color: rgb(0, 0, 204); color: white; text-align:center">NaN</td>', f'<td style = "color: white; text-align:center">-</td>')

        display(Markdown(score_is_corrcet_relationship_result_df))

    def print_latex_summary_statistics_result(self):
        """Prints the summary statistics result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            summary_statistics_result_df = self.summary_statistics_result_df
            summary_statistics_result_df = summary_statistics_result_df.set_index(self.dataset_name_str)
            summary_statistics_result_df.index.name = None
            print(summary_statistics_result_df.style.format(precision=2)
                                                    .format_index(axis=1, escape='latex')
                                                    .format_index(axis=0, escape='latex')
                                                    .to_latex()
                                                    .replace('<NA>', '-')
                                                    .replace('nan', '-'))

    def print_latex_sequence_summary_statistics_result(self):
        """Prints the sequence summary statistics result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            sequence_summary_statistics_result_df = self.sequence_summary_statistics_result_df
            sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.set_index(self.dataset_name_str)
            sequence_summary_statistics_result_df.index.name = None
            print(sequence_summary_statistics_result_df.style.format(precision=2)
                                                    .format_index(axis=1, escape='latex')
                                                    .format_index(axis=0, escape='latex')
                                                    .to_latex()
                                                    .replace('<NA>', '-')
                                                    .replace('nan', '-'))

    def print_latex_available_fields_result(self):
        """Prints the available fields result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            print(self.available_fields_result_df.style.format_index(axis=1, escape='latex')
                                                       .hide(axis=0)
                                                       .to_latex()
                                                       .replace('\multicolumn{4}{r}', '\multicolumn{4}{c}'))

    def print_latex_score_is_correct_relationship_result(self):
        """Prints the score is_correct relationship result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            package_str = '\\usepackage{multirow} \n'
            score_is_correct_relationship_df = self.score_is_correct_relationship_result_df
            score_is_correct_relationship_df = score_is_correct_relationship_df.set_index([self.dataset_name_str, self.field_str])
            score_is_correct_relationship_df.index.names = [None, None]
            print(package_str + score_is_correct_relationship_df.style.format(precision=2)
                                                                      .format_index(axis=1, escape='latex')
                                                                      .format_index(axis=0, escape='latex')
                                                                      .to_latex()
                                                                      .replace('<NA>', '-')
                                                                      .replace('nan', '-'))

class ResultPlots():
    """A class which upon initialization holds data about summary statistics, sequence summary statistics, the available fields and the score is_correct relationship of 
    the analysed datasets.
    """    
    # class vars
    path_to_html_table_dir = PATH_TO_PICKLED_OBJECTS_FOLDER + PATH_TO_RESULT_TABLES_PICKLE_FOLDER
    n_rows_str = N_ROWS_STR
    n_unique_users_str = N_UNIQUE_USERS_STR
    n_unique_learning_activities_str = N_UNIQUE_LEARNING_ACTIVITIES_STR
    n_unique_groups_str = N_UNIQUE_GROUPS_STR
    sparsity_user_learning_activity_matrix_str = SPARSITY_USER_LEARNING_ACTIVITY_MATRIX_STR
    sparsity_user_group_matrix_str = SPARSITY_USER_GROUP_MATRIX_STR
    n_sequences_str = N_SEQUENCES_STR
    n_unique_sequences_str = N_UNIQUE_SEQUENCES_STR
    is_available_str = IS_AVAILABLE_STR
    field_str = FIELD_STR
    dataset_name_str = DATASET_NAME_STR

    path_to_sequence_distances_analytics_dir = PATH_TO_PICKLED_OBJECTS_FOLDER + PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_PICKLE_FOLDER

    def __init__(self):

        # calculated upon initialization
        # html table for each dataset
        self.html_tables_dict = self._load_html_tables()
        self.avg_sequence_distance_per_group_agg_df, self.avg_unique_sequence_distance_per_group_agg_df  = self._load_sequence_distance_analytics()

        # result dataframes
        self.available_fields_result_df = self._gen_available_fields_result_df()
        self.summary_statistics_result_df = self._gen_summary_statistics_result_df()
        self.sequence_summary_statistics_result_df = self._gen_sequence_summary_statistics_result_df()
        self.score_is_correct_relationship_result_df = self._gen_score_is_correct_relationship_result_df()


    
    def _load_html_tables(self):
        """Returns a dictionary containing the html table objects of the analysed datasets

        Returns
        -------
        pd.DataFrame
            A dictionary containing the html table objects of the analysed datasets
        """        
        html_table_dir = AggregatedResultTables.path_to_html_table_dir
        html_tables_list = glob.glob(html_table_dir + '*.pickle')

        html_tables_dict = {}
        for i in html_tables_list:
            with open(i, 'rb') as f:
                html_table = pickle.load(f)
                dataset_name = html_table.dataset_name
            html_tables_dict[dataset_name] = html_table
        
        return html_tables_dict

    def _load_sequence_distance_analytics(self):
        sequence_distances_analytics_dir = AggregatedResultTables.path_to_sequence_distances_analytics_dir
        sequence_distances_analytics_list = glob.glob(sequence_distances_analytics_dir + '*.pickle')

        avg_sequence_distance_per_group_agg_df = pd.DataFrame()
        avg_unique_sequence_distance_per_group_agg_df = pd.DataFrame()
        for i in sequence_distances_analytics_list:
            with open(i, 'rb') as f:
                sequence_distance_analytics = pickle.load(f)
            
            avg_sequence_distance_per_group_df = sequence_distance_analytics.avg_sequence_distance_per_group_df
            avg_unique_sequence_distance_per_group_df = sequence_distance_analytics.avg_unique_sequence_distance_per_group_df
        
        avg_sequence_distance_per_group_agg_df = pd.concat([avg_sequence_distance_per_group_agg_df, 
                                                            avg_sequence_distance_per_group_df])
        avg_unique_sequence_distance_per_group_agg_df = pd.concat([avg_unique_sequence_distance_per_group_agg_df, 
                                                                   avg_unique_sequence_distance_per_group_df])
        
        return avg_sequence_distance_per_group_agg_df, avg_unique_sequence_distance_per_group_agg_df


class AggregatedResults():
    """docstring for ClassName."""
    def __init__(self):

        self._path_to_result_tables = self._return_result_tables_paths()

    @avg_sequence_statistics_per_group_per_dataset_decorator
    def plot_avg_sequence_statistics_per_group_per_dataset(self) -> None:

        avg_learning_activity_sequence_stats_per_group_per_dataset = self._generate_avg_sequence_stats_per_group_per_dataset_df()

        for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST:

            print(STAR_STRING)
            print(field.value)
            print(STAR_STRING)

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:
                    x_axis_lim = return_axis_limits(avg_learning_activity_sequence_stats_per_group_per_dataset[field.value],
                                                    False,
                                                    False)
                    x_axis_ticks = None

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE:
                    x_axis_lim = return_axis_limits(avg_learning_activity_sequence_stats_per_group_per_dataset[field.value],
                                                    True,
                                                    True)
                    x_axis_ticks = np.arange(0, 1.1, 0.1)

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                     SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:
                    x_axis_lim = return_axis_limits(avg_learning_activity_sequence_stats_per_group_per_dataset[field.value],
                                                    True,
                                                    False)
                    x_axis_ticks = np.arange(0, 110, 10)

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{field}')

            # boxplot
            g = sns.boxplot(
                            avg_learning_activity_sequence_stats_per_group_per_dataset, 
                            x=field.value, 
                            y=DATASET_NAME_FIELD_NAME_STR, 
                            hue=DATASET_NAME_FIELD_NAME_STR,
                            palette=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE,
                            showfliers=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_OUTLIERS,
                            linewidth=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_LINE_WIDTH,
                            width=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WIDTH,
                            whis=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WHISKERS,
                            showmeans=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_MEANS,
                            meanprops=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_MARKER,
                           )
            # strip or swarmplot
            g = sns.swarmplot(
                              avg_learning_activity_sequence_stats_per_group_per_dataset, 
                              x=field.value, 
                              y=DATASET_NAME_FIELD_NAME_STR, 
                              size=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_SIZE, 
                              color=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_COLOR,
                              alpha=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_ALPHA,
                              edgecolor=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_EDGECOLOR,
                              linewidth=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_LINEWIDTH,
                             )
            g.set(
                xlabel=field.value,
                ylabel='',
                xlim=x_axis_lim,
                )
            plt.xticks(x_axis_ticks)
            plt.tight_layout()
            plt.savefig(
                        f'{PATH_TO_RESULT_PLOTS}/{AVG_SEQUENCE_STATISTICS_PLOT_NAME}{field}.{SAVE_FIGURE_IMAGE_FORMAT}', 
                        dpi=SAVE_FIGURE_DPI,
                        format=SAVE_FIGURE_IMAGE_FORMAT,
                        bbox_inches=SAVE_FIGURE_BBOX_INCHES)
            plt.show(g);

    @summary_sequence_statistics_per_group_per_dataset_decorator
    def plot_summary_sequence_statistics_per_group_per_dataset(self) -> None:

        summary_stats_per_group_per_dataset = self._generate_summary_sequence_stats_per_group_per_dataset_df()

        for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST:

            print(STAR_STRING)
            print(field.value)
            print(STAR_STRING)

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE:

                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT_RATIO
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT_RATIO

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{field}')

            n_cols = set_facet_grid_column_number(summary_stats_per_group_per_dataset[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            # TODO: set new statistcs variable?
            g = sns.relplot(summary_stats_per_group_per_dataset,
                            x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                            y=field.value,
                            hue=GROUP_FIELD_NAME_STR,
                            col=DATASET_NAME_FIELD_NAME_STR,
                            col_wrap=n_cols,
                            height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                            aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                            kind=SUMMARY_SEQUENCE_STATISTICS_KIND,
                            legend=SUMMARY_SEQUENCE_STATISTICS_PLOT_LEGEND,
                            linewidth=SUMMARY_SEQUENCE_STATISTICS_LINE_WIDTH,
                            markers=SUMMARY_SEQUENCE_STATISTICS_PLOT_MARKERS,
                            marker=SUMMARY_SEQUENCE_STATISTICS_MARKER_TYPE,
                            markersize=SUMMARY_SEQUENCE_STATISTICS_MARKER_SIZE,
                            markerfacecolor=SUMMARY_SEQUENCE_STATISTICS_MARKER_FACECOLOR,
                            markeredgecolor=SUMMARY_SEQUENCE_STATISTICS_MARKER_EDGECOLOR,
                            markeredgewidth=SUMMARY_SEQUENCE_STATISTICS_MARKER_EDGEWIDTH,
                            alpha=SUMMARY_SEQUENCE_STATISTICS_MARKER_ALPHA,
                            facet_kws=dict(sharex=share_x,
                                           sharey=share_y)
                        )

            for ax, (facet_val, facet_data) in zip(g.axes.flat, summary_stats_per_group_per_dataset.groupby(DATASET_NAME_FIELD_NAME_STR)):

                y_axis_lim = return_axis_limits(facet_data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio)
                ax.set_ylim(*y_axis_lim)

                n_groups = facet_data[GROUP_FIELD_NAME_STR].nunique()

                color_palette = sns.color_palette(SUMMARY_SEQUENCE_STATISTICS_COLOR_PALETTE, 
                                                  n_colors=n_groups)
                
                for line, color in zip(ax.lines, color_palette):
                    line.set_color(color)
                    line.set_alpha(SUMMARY_SEQUENCE_STATISTICS_LINE_ALPHA)

                ax.spines['top'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_TOP)
                ax.spines['bottom'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_LEFT)
                ax.spines['right'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_RIGHT)

            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            plt.tight_layout()
            plt.savefig(f'{PATH_TO_RESULT_PLOTS}/{SUMMARY_SEQUENCE_STATISTICS_PLOT_NAME}{field}.{SAVE_FIGURE_IMAGE_FORMAT}', 
                        dpi=SAVE_FIGURE_DPI,
                        format=SAVE_FIGURE_IMAGE_FORMAT,
                        bbox_inches=SAVE_FIGURE_BBOX_INCHES)
            plt.show(g);

    @sequence_statistics_distribution_per_group_per_dataset_decorator
    def plot_sequence_statistics_distribution_per_group_per_dataset(self) -> None:

        sequence_statistics_per_group_per_dataset = self._generate_sequence_statistics_distribtion_per_group_per_dataset_df()

        for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST:

            print(STAR_STRING)
            print(field.value)
            print(STAR_STRING)

            if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                print(f'{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_IS_SORTED_STR}{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC.value}')
            else:
                print(f'{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_IS_SORTED_STR}{GROUP_FIELD_NAME_STR}{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_GROUP_NUMBER_STR}')
            print('')

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_RAW

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE:

                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT_RATIO
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT_RATIO

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{field}')

            n_cols = set_facet_grid_column_number(sequence_statistics_per_group_per_dataset[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            def plot_boxplot(data, **kwargs):

                if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                    data = self._sort_groups_by_metric(data,
                                                       field.value)
                
                sns.boxplot(
                            data=data, 
                            x=field.value,
                            y=GROUP_FIELD_NAME_STR,
                            hue=GROUP_FIELD_NAME_STR,
                            orient=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_ORIENTATION,
                            palette=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_PALETTE,
                            showfliers=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_OUTLIERS,
                            linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LINE_WIDTH,
                            width=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WIDTH,
                            whis=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WHISKERS,
                            showmeans=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_MEANS,
                            meanprops=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_MARKER,
                            **kwargs)

            g = sns.FacetGrid(sequence_statistics_per_group_per_dataset,
                              col=DATASET_NAME_FIELD_NAME_STR,
                              col_wrap=n_cols,
                              height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                              aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                              sharex=share_x,
                              sharey=share_y,
            )
            g.map_dataframe(plot_boxplot)

            for ax, (facet_val, facet_data) in zip(g.axes.flat, sequence_statistics_per_group_per_dataset.groupby(DATASET_NAME_FIELD_NAME_STR)):

                x_axis_lim = return_axis_limits(facet_data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio)
                ax.set_xlim(*x_axis_lim)

                n_groups = facet_data[GROUP_FIELD_NAME_STR].nunique()

                color_palette = sns.color_palette(SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_PALETTE, 
                                                  n_colors=n_groups)

                if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                    groups = np.unique(facet_data[GROUP_FIELD_NAME_STR])
                    color_dict = dict(zip(groups, color_palette))

                    labels = [int(tick.get_text()) for tick in ax.get_yticklabels()]

                    for box, group in zip(ax.patches, labels):
                        box.set_facecolor(color_dict[group])
                else:
                    for box, color in zip(ax.patches, color_palette):
                        box.set_facecolor(color)

                ax.spines['top'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_TOP)
                ax.spines['bottom'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_LEFT)
                ax.spines['right'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_RIGHT)

            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            plt.tight_layout()
            plt.savefig(f'{PATH_TO_RESULT_PLOTS}/{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_NAME}{field}.{SAVE_FIGURE_IMAGE_FORMAT}', 
                        dpi=SAVE_FIGURE_DPI,
                        format=SAVE_FIGURE_IMAGE_FORMAT,
                        bbox_inches=SAVE_FIGURE_BBOX_INCHES)
            plt.show(g);


    def _return_result_tables_paths(self) -> list[str]:

        path_to_result_tables_dir = Path(PATH_TO_PICKLED_OBJECTS_FOLDER) / PATH_TO_RESULT_TABLES_PICKLE_FOLDER
        extension = '.pickle'
        path_to_result_tables = [file for file in path_to_result_tables_dir.rglob(f'*{extension}')]

        return path_to_result_tables
    
    def _generate_avg_sequence_stats_per_group_per_dataset_df(self) -> pd.DataFrame:

        fields_to_plot = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]

        avg_learning_activity_sequence_stats_per_group_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                result_table = pickle.load(f)
            
            learning_activity_sequence_stats_per_group = result_table.learning_activity_sequence_stats_per_group
            avg_learning_activity_sequence_stats_per_group = (learning_activity_sequence_stats_per_group
                                                              .groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])
                                                              [fields_to_plot].agg(np.mean))

            avg_learning_activity_sequence_stats_per_group = avg_learning_activity_sequence_stats_per_group.reset_index()

            avg_learning_activity_sequence_stats_per_group_df_list.append(avg_learning_activity_sequence_stats_per_group)

        avg_learning_activity_sequence_stats_per_group_per_dataset = pd.concat(avg_learning_activity_sequence_stats_per_group_df_list)
        avg_learning_activity_sequence_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)
    
        return avg_learning_activity_sequence_stats_per_group_per_dataset 

    def _generate_summary_sequence_stats_per_group_per_dataset_df(self) -> pd.DataFrame:

        # helper functions for quartiles 
        def first_quartile(array):
            return np.quantile(array, 0.25)
        def third_quartile(array):
            return np.quantile(array, 0.75)

        fields_to_plot_list = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]

        summary_stats_per_group_list= []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                result_table = pickle.load(f)

                summary_statistic_per_field_long_list = []
                for field in fields_to_plot_list:

                    summary_statistic_per_field = (result_table.learning_activity_sequence_stats_per_group
                                                   .groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])[field]
                                                   .agg([min, max, np.median, first_quartile, third_quartile])
                                                   .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                    'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                    'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                                    'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                    'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})
                                                   .reset_index())

                    field_list = [DATASET_NAME_FIELD_NAME_STR, 
                                  GROUP_FIELD_NAME_STR, 
                                  LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                  LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                  LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                  LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                  LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]

                    summary_statistic_per_field_long = pd.melt(summary_statistic_per_field[field_list], 
                                                               id_vars=[DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR],
                                                               var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                               value_name=field)
                    sort_list = [DATASET_NAME_FIELD_NAME_STR, 
                                 GROUP_FIELD_NAME_STR]
                    summary_statistic_per_field_long.sort_values(by=sort_list,
                                                                 inplace=True)

                    summary_statistic_per_field_long_list.append(summary_statistic_per_field_long)
            
            merge_field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR]
            summary_stats_per_group = reduce(lambda left, right: pd.merge(left=left, 
                                                                          right=right, 
                                                                          how='inner', 
                                                                          on=merge_field_list), summary_statistic_per_field_long_list)
            summary_stats_per_group_list.append(summary_stats_per_group)

        summary_stats_per_group_per_dataset = pd.concat(summary_stats_per_group_list)
        summary_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)

        return summary_stats_per_group_per_dataset

    def _generate_sequence_statistics_distribtion_per_group_per_dataset_df(self) -> pd.DataFrame:

        fields_to_plot = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]
        field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, *fields_to_plot]

        learning_activity_sequence_stats_per_group_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                result_table = pickle.load(f)
            
            learning_activity_sequence_stats_per_group = result_table.learning_activity_sequence_stats_per_group[field_list]

            learning_activity_sequence_stats_per_group_df_list.append(learning_activity_sequence_stats_per_group)

        summary_stats_per_group_per_dataset = pd.concat(learning_activity_sequence_stats_per_group_df_list)
        summary_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)

        return summary_stats_per_group_per_dataset
    
    def _sort_groups_by_metric(self,
                               data: pd.DataFrame,
                               sequence_statistic: str) -> pd.DataFrame:

        data = data.copy()

        match SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC:
            case SequenceStatisticsDistributionBoxplotSortMetric.MEAN:
                sort_metric = np.mean

            case SequenceStatisticsDistributionBoxplotSortMetric.MEDIAN:
                sort_metric = np.median

            case SequenceStatisticsDistributionBoxplotSortMetric.MAX:
                sort_metric = np.max

            case SequenceStatisticsDistributionBoxplotSortMetric.MIN:
                sort_metric = np.min

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC}')

        sort_metric_values = data.groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])[sequence_statistic]\
                                 .agg(sort_metric)\
                                 .reset_index()

        sort_metric_values.rename({sequence_statistic: SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC.name}, 
                                  axis=1, 
                                  inplace=True)

        sort_metric_values.sort_values(by=[DATASET_NAME_FIELD_NAME_STR, SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC.name], inplace=True, ascending=False)

        data[GROUP_FIELD_NAME_STR] = pd.Categorical(data[GROUP_FIELD_NAME_STR], categories=sort_metric_values[GROUP_FIELD_NAME_STR], ordered=True)
        data.sort_values(by=GROUP_FIELD_NAME_STR, inplace=True)

        return data