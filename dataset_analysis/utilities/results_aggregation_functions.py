from .standard_import import *
from .constants import *

class ResultTables():
    """A class which upon initialization holds data about summary statistics, sequence summary statistics, the available fields and the score is_correct relationship of 
    the analysed datasets.
    """    
    # class vars
    path_to_html_table_dir = PATH_TO_PICKLED_OBJECTS_FOLDER + PATH_TO_HTML_TABLES_PICKLE_FOLDER
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

    def __init__(self):

        # calculated upon initialization
        # html table for each dataset
        self.html_tables_dict = self._load_html_tables()

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
        html_table_dir = self.path_to_html_table_dir
        html_tables_list = glob.glob(html_table_dir + '*.pickle')

        html_tables_dict = {}
        for i in html_tables_list:
            with open(i, 'rb') as f:
                html_table = pickle.load(f)
                dataset_name = html_table.dataset_name
            html_tables_dict[dataset_name] = html_table
        
        return html_tables_dict

    def _load_sequence_distances(self):
        pass

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
        idx = [ResultTables.n_rows_str,
               ResultTables.n_unique_users_str, 
               ResultTables.n_unique_groups_str, 
               ResultTables.n_unique_learning_activities_str]
        typecast_dict = {i: 'Int64' for i in idx}
        summary_statistics_result_df = summary_statistics_result_df.astype(typecast_dict)
        summary_statistics_result_df = summary_statistics_result_df.reset_index(names=self.dataset_name_str)

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
        idx = [ResultTables.n_sequences_str,
               ResultTables.n_unique_sequences_str]
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
        available_fields_result_df.columns = pd.MultiIndex.from_product([[self.is_available_str], available_fields_result_df.columns])
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
        available_fields_result_df = available_fields_result_df.replace(f'<th colspan="3" halign="left">{ResultTables.is_available_str}</th>', f'<th colspan="3" halign="left", style = "background-color: royalblue; color: white; text-align:center">{ResultTables.is_available_str}</th>')

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