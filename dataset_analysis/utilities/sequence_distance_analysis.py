from .standard_import import *
from .constants import *
from .config import *
from .functions import *
from .sequence_distance import *
from .io_functions import *

class SequenceDistanceAnalytics:
    """docstring for ClassName."""

    # class variables
    # paths to pickle file directories
    path_to_pickled_objects_folder = PATH_TO_PICKLED_OBJECTS_FOLDER
    path_to_sequence_distances_pickle_folder = PATH_TO_SEQUENCE_DISTANCES_PICKLE_FOLDER
    path_to_sequence_distance_analytics_data_pickle_folder = PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_DATA_PICKLE_FOLDER
    path_to_sequence_distance_analytics_distance_square_matrix_pickle_folder = PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_DISTANCE_SQUARE_MATRIX_PICKLE_FOLDER
    
    # pickle file names for intermediate results which can be too large for memory for some datasets
    sequence_distance_analytics_sequence_dist_data_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_DATA_PICKLE_NAME
    sequence_distance_analytics_unique_sequence_dist_data_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_DIST_DATA_PICKLE_NAME
    sequence_distance_analytics_sequence_len_data_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_LEN_DATA_PICKLE_NAME
    sequence_distance_analytics_unique_sequence_len_data_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_LEN_DATA_PICKLE_NAME
    sequence_distance_analytics_sequence_dist_square_matrix_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_PICKLE_NAME
    sequence_distance_analytics_unique_sequence_dist_square_matrix_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_DIST_SQUARE_MATRIX_PICKLE_NAME
    sequence_distance_analytics_sequence_dist_square_matrix_normalized_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_NORMALIZED_PICKLE_NAME
    sequence_distance_analytics_unique_sequence_dist_square_matrix_normalized_pickle_name = SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_DIST_SQUARE_MATRIX_NORMALIZED_PICKLE_NAME

    # result dict keys
    learning_activity_sequence_distance_based_on_user_combinations_name_str = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR
    learning_activity_sequence_distance_based_on_sequence_combinations_name_str = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR

    # subdict keys used for creating the final dataframes
    # sequence distance
    learning_activity_sequence_distance_name_str = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_sequence_max_length_name_str = LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR
    learning_activity_normalized_sequence_distance_name_str = LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_sequence_user_combination_user_a_name_str = LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_A_NAME_STR
    learning_activity_sequence_user_combination_user_b_name_str = LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_B_NAME_STR
    learning_activity_sequence_sequence_id_combination_sequence_a_name_str = LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_A_NAME_STR
    learning_activity_sequence_sequence_id_combination_sequence_b_name_str = LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_B_NAME_STR
    learning_activity_sequence_user_name_str = LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR
    learning_activity_sequence_id_name_str = LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR
    # sequence length 
    learning_activity_sequence_sequence_id_name_str = LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR
    learning_activity_sequence_length_name_str = LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR

    # name of aggregated fields
    # sequence distance
    learning_activity_mean_sequence_distance_name_str = LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_median_sequence_distance_name_str = LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_mean_normalized_sequence_distance_name_str = LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_median_normalized_sequence_distance_name_str = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR   
    # sequence length 
    learning_activity_mean_sequence_length_name_str = LEARNING_ACTIVITY_MEAN_SEQUENCE_LENGTH_NAME_STR
    learning_activity_median_sequence_length_name_str = LEARNING_ACTIVITY_MEDIAN_SEQUENCE_LENGTH_NAME_STR

    # sequence distance square matrix fields
    sequence_distance_analytics_sequence_dist_square_matrix_field_name_str = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_FIELD_NAME_STR

    # additional field names
    dataset_name_str = DATASET_NAME_STR
    group_field_name_str = GROUP_FIELD_NAME_STR
    user_field_name_str = USER_FIELD_NAME_STR
    sequence_str = SEQUENCE_STR
    
    # list of fields to be imported into the respective dataframes
    # for the sequence distance df
    seq_dist_per_group_fields_list = [group_field_name_str,
                                      learning_activity_sequence_distance_name_str,
                                      learning_activity_sequence_max_length_name_str,
                                      learning_activity_sequence_user_combination_user_a_name_str,
                                      learning_activity_sequence_user_combination_user_b_name_str,
                                      learning_activity_sequence_sequence_id_combination_sequence_a_name_str,
                                      learning_activity_sequence_sequence_id_combination_sequence_b_name_str]
    # for the unique sequence distance df
    unique_seq_dist_per_group_fields_list = [group_field_name_str,
                                             learning_activity_sequence_distance_name_str,
                                             learning_activity_sequence_max_length_name_str,
                                             learning_activity_sequence_sequence_id_combination_sequence_a_name_str,
                                             learning_activity_sequence_sequence_id_combination_sequence_b_name_str]
    # for both sequence length dfs
    seq_len_per_group_fields_list = [group_field_name_str,
                                     learning_activity_sequence_sequence_id_name_str,
                                     learning_activity_sequence_length_name_str]
                                            
    # list of field names used for aggregation + list of field names used for naming aggregated fields in the respective dataframes
    # fields to be aggregated for the avg seq dist and avg unique seq dist dfs
    seq_distance_fields = [learning_activity_sequence_distance_name_str, 
                           learning_activity_normalized_sequence_distance_name_str]
    # names of the aggregated fields
    avg_seq_dist_fields_list = [learning_activity_mean_sequence_distance_name_str,
                                learning_activity_median_sequence_distance_name_str,
                                learning_activity_mean_normalized_sequence_distance_name_str,
                                learning_activity_median_normalized_sequence_distance_name_str]
    
    # fields to be aggregated for the avg seq len and avg unique seq len dfs
    seq_length_fields = [learning_activity_sequence_length_name_str]
    # names of the aggregated fields
    avg_seq_length_fields_list = [learning_activity_mean_sequence_length_name_str,
                                  learning_activity_median_sequence_length_name_str]

    def __init__(self, 
                 dataset_name: str):
        self.dataset_name = dataset_name
        # specify sequence distance directory where pickle files will be read from
        self.sequence_distance_directory_list = [SequenceDistanceAnalytics.path_to_sequence_distances_pickle_folder, 
                                                 self.dataset_name]

        # specify sequence distance analytics data directory where pickle files will be written to and read from
        self.sequence_distance_analytics_data_directory_list = [SequenceDistanceAnalytics.path_to_sequence_distance_analytics_data_pickle_folder,
                                                                self.dataset_name]

        # specify sequence distance matrix directory where pickle files will be written to
        self.sequence_distance_analytics_distance_matrix_directory_list = [SequenceDistanceAnalytics.path_to_sequence_distance_analytics_distance_square_matrix_pickle_folder,
                                                                           self.dataset_name]

        # delete old pickle files to prevent keeping results for groups which are not part of current calculation anymore
        delete_all_pickle_files_within_directory(self.sequence_distance_analytics_data_directory_list)
        delete_all_pickle_files_within_directory(self.sequence_distance_analytics_distance_matrix_directory_list)

        # calculate the sequence distance and sequence length dataframes and write to disk as pickle in order to save memory
        # sequence distance dataframes
        self.seq_dist_data_param = (self.learning_activity_sequence_distance_based_on_user_combinations_name_str, 
                                    self.seq_dist_per_group_fields_list,
                                    self.sequence_distance_directory_list,
                                    SequenceDistanceAnalytics._add_normalized_seq_dist_field)
        self.seq_dist_data_name = self.dataset_name + SequenceDistanceAnalytics.sequence_distance_analytics_sequence_dist_data_pickle_name

        self.unique_seq_dist_data_param = (self.learning_activity_sequence_distance_based_on_sequence_combinations_name_str, 
                                           self.unique_seq_dist_per_group_fields_list,
                                           self.sequence_distance_directory_list,
                                           SequenceDistanceAnalytics._add_normalized_seq_dist_field)
        self.unique_seq_dist_data_name = self.dataset_name + SequenceDistanceAnalytics.sequence_distance_analytics_unique_sequence_dist_data_pickle_name

        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.seq_dist_data_param, insert_position=2),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.seq_dist_data_name)
        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.unique_seq_dist_data_param, insert_position=2),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.unique_seq_dist_data_name)
        
        # sequence length dataframes
        self.seq_len_data_param = (self.learning_activity_sequence_distance_based_on_user_combinations_name_str, 
                                   self.seq_len_per_group_fields_list,
                                   self.sequence_distance_directory_list)
        self.seq_len_data_name = self.dataset_name + SequenceDistanceAnalytics.sequence_distance_analytics_sequence_len_data_pickle_name

        self.unique_seq_len_data_param = (self.learning_activity_sequence_distance_based_on_sequence_combinations_name_str, 
                                          self.seq_len_per_group_fields_list,
                                          self.sequence_distance_directory_list)
        self.unique_seq_len_data_name = self.dataset_name + SequenceDistanceAnalytics.sequence_distance_analytics_unique_sequence_len_data_pickle_name

        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.seq_len_data_param),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.seq_len_data_name)
        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.unique_seq_len_data_param),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.unique_seq_len_data_name)

        # avg sequence distance dataframes
        self.avg_sequence_distance_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                                 self.seq_dist_data_name),
                                                                                  SequenceDistanceAnalytics.seq_distance_fields,
                                                                                  SequenceDistanceAnalytics.avg_seq_dist_fields_list)
        self.avg_unique_sequence_distance_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                                        self.unique_seq_dist_data_name),
                                                                                         SequenceDistanceAnalytics.seq_distance_fields,
                                                                                         SequenceDistanceAnalytics.avg_seq_dist_fields_list)

        # avg sequence length dataframes
        self.avg_sequence_length_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                               self.seq_len_data_name),
                                                                                SequenceDistanceAnalytics.seq_length_fields,
                                                                                SequenceDistanceAnalytics.avg_seq_length_fields_list)
        self.avg_unique_sequence_length_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                                      self.unique_seq_len_data_name),
                                                                                       SequenceDistanceAnalytics.seq_length_fields,
                                                                                       SequenceDistanceAnalytics.avg_seq_length_fields_list)

        # merged sequence distance - sequence length dataframes
        # base: user combinations
        self.avg_sequence_distance_sequence_length_per_group_df = self._merge_df_on_group(self.avg_sequence_distance_per_group_df,
                                                                                          self.avg_sequence_length_per_group_df)
        # base: sequence combinations
        self.avg_unique_sequence_distance_unique_sequence_length_per_group_df = self._merge_df_on_group(self.avg_unique_sequence_distance_per_group_df,
                                                                                                        self.avg_unique_sequence_length_per_group_df)

        ## generate sequence distance square matrices per group and write per group pickle files to disk
        # normalize vs non normalize sequences distances // sequence distance based on user combinations vs sequence combinations 
        self._generate_seq_dist_square_matrix_per_group_dict(False, False)
        self._generate_seq_dist_square_matrix_per_group_dict(False, True)
        self._generate_seq_dist_square_matrix_per_group_dict(True, False)
        self._generate_seq_dist_square_matrix_per_group_dict(True, True)

    def _read_in_df(self, 
                    file_name: str,
                    subdict_name: str,
                    fields_list) -> pd.DataFrame:
        data_dict = pickle_read(self.sequence_distance_directory_list,
                                file_name)

        data_dict = {k:v for k,v in data_dict[subdict_name].items() if k in fields_list}
        return pd.DataFrame(data_dict).apply(pd.to_numeric, downcast='integer')
    
    def _generate_df_all_group_from_seq_dist_dict(self,
                                                  subdict_name: str,
                                                  fields_list: list,
                                                  path_within_pickle_directory_list: list[str],
                                                  *args,
                                                  **kwargs) -> pd.DataFrame:

        pickle_files_list = return_pickled_files_list(path_within_pickle_directory_list)
        
        # use a generator expression to reduce memory usage
        df_list = (self._read_in_df(file, subdict_name, fields_list) for file in pickle_files_list)
        df_all_group = pd.concat(df_list, ignore_index=True)

        for function in args:
            df_all_group  = function(df_all_group, **kwargs)

        return df_all_group
    
    def _generate_df_avg_per_group(self,
                                   data: pd.DataFrame,
                                   fields_to_aggregate: list[str],
                                   aggregated_fields_names: list[str]) -> pd.DataFrame:

        grouper = SequenceDistanceAnalytics.group_field_name_str
        df_name_group_list = [SequenceDistanceAnalytics.dataset_name_str, grouper]
        avg_per_group_df = (data.groupby(grouper)[fields_to_aggregate]
                                .agg([np.mean, np.median]))
        avg_per_group_df = avg_per_group_df.droplevel(0, axis=1)
        avg_per_group_df = avg_per_group_df.reset_index()
        avg_per_group_df.insert(0, SequenceDistanceAnalytics.dataset_name_str, self.dataset_name)
        avg_per_group_df.columns = df_name_group_list + aggregated_fields_names

        return avg_per_group_df
    
    @classmethod
    def _add_normalized_seq_dist_field(cls,
                                       df: pd.DataFrame,
                                       **kwargs) -> pd.DataFrame:
                                    
        if 'insert_position' in kwargs:
            insert_position = kwargs['insert_position']
        else:
            insert_position = 0
             
        seq_dist = df[SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str]
        max_seq_len = df[SequenceDistanceAnalytics.learning_activity_sequence_max_length_name_str]
        normalized_seq_dist = seq_dist / max_seq_len
        df.insert(insert_position, 
                  SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str, 
                  normalized_seq_dist)
        return df

    @classmethod
    def _merge_df_on_group(cls,
                           df_left: pd.DataFrame,
                           df_right: pd.DataFrame) -> pd.DataFrame:

        merged_df = pd.merge(df_left, 
                             df_right, 
                             how='inner', 
                             on=[SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str])

        return merged_df
    
    def _write_object_as_pickle(self,
                                object_to_pickle,
                                path_within_pickle_directory_list: list[str],
                                file_name: str):

        pickle_write(object_to_pickle, 
                     path_within_pickle_directory_list, 
                     file_name)
    
    def _read_object_frome_pickle(self,
                                  path_within_pickle_directory_list: list[str],
                                  file_name: str) -> pd.DataFrame:

        seq_dist_df = pickle_read(path_within_pickle_directory_list,
                                  file_name)
        
        return seq_dist_df

    def _return_seq_dist_square_matrix_per_group_substring(self,
                                                           normalize_distance: bool,
                                                           use_unique_sequence_distances: bool) -> str:

        if (use_unique_sequence_distances and normalize_distance):
            file_name_substring = SequenceDistanceAnalytics.sequence_distance_analytics_unique_sequence_dist_square_matrix_normalized_pickle_name
        elif (use_unique_sequence_distances and not normalize_distance):
            file_name_substring = SequenceDistanceAnalytics.sequence_distance_analytics_unique_sequence_dist_square_matrix_pickle_name
        elif (not use_unique_sequence_distances and normalize_distance):
            file_name_substring = SequenceDistanceAnalytics.sequence_distance_analytics_sequence_dist_square_matrix_normalized_pickle_name
        else:
            file_name_substring = SequenceDistanceAnalytics.sequence_distance_analytics_sequence_dist_square_matrix_pickle_name

        file_name = self.dataset_name + file_name_substring + SequenceDistanceAnalytics.group_field_name_str + '_'

        return file_name

    def _generate_seq_dist_square_matrix_per_group_dict(self,
                                                        normalize_distance: bool,
                                                        use_unique_sequence_distances: bool) -> None:

        if use_unique_sequence_distances:
            sequence_distance_base = SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_sequence_combinations_name_str
            label_type = SequenceDistanceAnalytics.learning_activity_sequence_id_name_str
        else:
            sequence_distance_base = SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_user_combinations_name_str
            label_type = SequenceDistanceAnalytics.learning_activity_sequence_user_name_str

        seq_dist_fields_list = [SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str,
                                SequenceDistanceAnalytics.learning_activity_sequence_max_length_name_str]
        
        group_label_fields_list = [SequenceDistanceAnalytics.group_field_name_str, label_type]

        pickle_files_list = return_pickled_files_list(self.sequence_distance_directory_list)
        for file in pickle_files_list:
            seq_dist_df = self._read_in_df(file, 
                                           sequence_distance_base, 
                                           seq_dist_fields_list)

            seq_dist_df = self._add_normalized_seq_dist_field(seq_dist_df,
                                                              insert_position=1)

            group_label_df = self._read_in_df(file,
                                              sequence_distance_base,
                                              group_label_fields_list) 

            if normalize_distance:
                distances = seq_dist_df[SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str]
            else:
                distances = seq_dist_df[SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str]
            
            labels = list(group_label_df[label_type])
            group = group_label_df[SequenceDistanceAnalytics.group_field_name_str][0]

            square_matrix = squareform(distances)
        
            square_matrix_df = pd.DataFrame(square_matrix,
                                            index=labels, 
                                            columns=labels) 
            labels = sorted(labels, key=int)
            square_matrix_df = square_matrix_df.loc[labels, labels]

            file_name_substring = self._return_seq_dist_square_matrix_per_group_substring(normalize_distance,
                                                                                          use_unique_sequence_distances)
            file_name = file_name_substring + str(group)

            square_matrix_result_dict = {SequenceDistanceAnalytics.group_field_name_str: str(group),
                                         SequenceDistanceAnalytics.sequence_distance_analytics_sequence_dist_square_matrix_field_name_str: square_matrix_df}

            self._write_object_as_pickle(square_matrix_result_dict,
                                         self.sequence_distance_analytics_distance_matrix_directory_list,
                                         file_name)

    @classmethod
    def _unpivot_seq_dist_square_matrix(cls,
                                        square_matrix_result_dict: dict) -> pd.DataFrame:
            
        square_matrix_df = square_matrix_result_dict[SequenceDistanceAnalytics.sequence_distance_analytics_sequence_dist_square_matrix_field_name_str]
        group = square_matrix_result_dict[SequenceDistanceAnalytics.group_field_name_str]

        square_matrix_df = (pd.melt(square_matrix_df, 
                                    ignore_index=False, 
                                    var_name='row', 
                                    value_name='value')
                                .reset_index()
                                .rename(columns={'index': 'column'}))
        square_matrix_df[SequenceDistanceAnalytics.group_field_name_str] = group

        return square_matrix_df

    def plot_sequence_distance_per_group(self):

        # plot sequence distance per group
        data_name_list = [self.seq_dist_data_name, 
                          self.unique_seq_dist_data_name]
        caption_list = [f'Base: All {SequenceDistanceAnalytics.user_field_name_str}-{SequenceDistanceAnalytics.sequence_str} Combinations', 
                        f'Base: All Unique-{SequenceDistanceAnalytics.sequence_str} Combinations',]
        
        # use generator to save memory when reading in sequence distance results dataframes
        def return_generator(data_name_list, caption_list):
            data_generator = ((self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list, name), caption) for name, caption in zip(data_name_list, caption_list))
            return data_generator

        print('*'*100)
        print('*'*100)
        print(' ')
        for x_var in SequenceDistanceAnalytics.seq_distance_fields:
            for data, caption in return_generator(data_name_list, caption_list):
                print('-'*100)
                print(f'{x_var} per {SequenceDistanceAnalytics.group_field_name_str}:')
                print(caption)
                print('-'*100)
                print('\n')
                print('Plots:')
                g = sns.boxplot(data=data, 
                                x=x_var, 
                                y=SequenceDistanceAnalytics.group_field_name_str,
                                showmeans=True, 
                                meanprops=marker_config,
                                orient='h');

                if x_var == SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str:
                    g.set(xlim=(-0.01, 1.01))

                for patch in g.patches:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0.5))

                # g = sns.stripplot(data=data, 
                #                 x=x_var, 
                #                 y=SequenceDistanceAnalytics.group_field_name_str,
                #                 size=2, 
                #                 color="red",
                #                 alpha=0.1)
                # g.set(xlabel=x_var);
                plt.show()
                print('*'*100)
                print('*'*100)
                print(' ')

        # plot avg sequence distance per group
        avg_per_group_data_list = [self.avg_sequence_distance_sequence_length_per_group_df, 
                                   self.avg_unique_sequence_distance_unique_sequence_length_per_group_df] 
        avg_data_caption_list = list(zip(avg_per_group_data_list, caption_list))
        for var in SequenceDistanceAnalytics.avg_seq_dist_fields_list:
            for data, caption in avg_data_caption_list:
                print('-'*100)
                print(f'{var} per {SequenceDistanceAnalytics.group_field_name_str}:')
                print(caption)
                print('-'*100)
                print('\n')
                print('Plots:')
                # sequence distance distribution
                plot_distribution(data, 
                                  var,
                                  var,
                                  False)

                # avg sequence length vs avg sequence distance
                g = sns.regplot(data=data, 
                                x=var, 
                                y=SequenceDistanceAnalytics.learning_activity_mean_sequence_length_name_str)
                g.set(xlabel=f'{var} per {SequenceDistanceAnalytics.group_field_name_str}', 
                      ylabel=f'{SequenceDistanceAnalytics.learning_activity_mean_sequence_length_name_str} per {SequenceDistanceAnalytics.group_field_name_str}');
                plt.show()
                g = sns.regplot(data=data, 
                                x=var, 
                                y=SequenceDistanceAnalytics.learning_activity_median_sequence_length_name_str)
                g.set(xlabel=f'{var} per {SequenceDistanceAnalytics.group_field_name_str}', 
                      ylabel=f'{SequenceDistanceAnalytics.learning_activity_median_sequence_length_name_str} per {SequenceDistanceAnalytics.group_field_name_str}');
                plt.show()
            print('*'*100)
            print('*'*100)
            print(' ')
    
    def plot_sequence_distance_matrix_per_group(self,
                                                normalize_distance: bool,
                                                use_unique_sequence_distances: bool,
                                                height: int,
                                                group_str=None) -> None:
        """Plot the sequence distance matrix group.

        Parameters
        ----------
        normalize_distance : bool
            A boolean indicating whether the sequence distances are being normalized between 0 and 1
        use_unique_sequence_distances: bool
            A boolean indicating whether only unique sequences are being used as the basis for distance calculations
        height : int
            The height of the subplots
        group_str : list of str, optional
            A list of groups for which the sequence distance matrices will be plotted. If None all groups will de displayed , by default None
        """

        file_name_substring = self._return_seq_dist_square_matrix_per_group_substring(normalize_distance,
                                                                                      use_unique_sequence_distances)

        pickle_files_list = return_pickled_files_list(self.sequence_distance_analytics_distance_matrix_directory_list,
                                                      file_name_substring)

        # filter the filenames by groups in group_str
        if group_str:
            pickle_files_list = [i for i in pickle_files_list if i.split('.')[0][-1] in group_str]

        # use a generator expression to reduce memory usage
        square_matrix_df_generator = (self._unpivot_seq_dist_square_matrix(
                                      self._read_object_frome_pickle(self.sequence_distance_analytics_distance_matrix_directory_list, file_name)) 
                                      for file_name in pickle_files_list)
        
        square_matrices_per_group_df = pd.concat(square_matrix_df_generator)
        
        # helper function
        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            labels = d.columns
            labels_sorted = sorted(labels, key=int)
            d = d.loc[labels_sorted,labels_sorted]
            sns.heatmap(d, **kwargs)

        if normalize_distance:
            distance_str = f'Normalized {SequenceDistanceAnalytics.sequence_str} Distance Matrix per {SequenceDistanceAnalytics.group_field_name_str}:'
        else:
            distance_str = f'{SequenceDistanceAnalytics.sequence_str} Distance Matrix per {SequenceDistanceAnalytics.group_field_name_str}:'

        if use_unique_sequence_distances:
            base_str = f'Base: All Unique-{SequenceDistanceAnalytics.sequence_str} Combinations'
        else:
            base_str = f'Base: All {SequenceDistanceAnalytics.user_field_name_str}-{SequenceDistanceAnalytics.sequence_str} Combinations' 

        print('*'*100)
        print('*'*100)
        print(' ')
        print('-'*100)
        print(distance_str)
        print(' ')
        print(base_str)
        print('-'*100)
        g = sns.FacetGrid(square_matrices_per_group_df, 
                          col=SequenceDistanceAnalytics.group_field_name_str,
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
        g.set(xlabel=SequenceDistanceAnalytics.sequence_str, 
              ylabel=SequenceDistanceAnalytics.sequence_str)
        # get figure background color
        facecolor=plt.gcf().get_facecolor()
        for ax in g.axes.flat:
            # set aspect of all axis
            ax.set_aspect('equal','box')
            # set background color of axis instance
            ax.set_facecolor(facecolor)
        plt.show()