from .standard_import import *
from .constants import *
from .config import *
from .preprocessing_functions import *
from .sequence_distance import *
from .io_functions import *
from .sequence_statistics_functions import SequenceStatistics
from .plotting_functions import plot_distribution

class SequenceDistanceAnalytics:
    """A class for analysing sequence distances per group"""
    
    # list of fields to be imported into the respective dataframes
    # for the sequence distance df
    seq_dist_per_group_fields_list = [GROUP_FIELD_NAME_STR,
                                      LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR,
                                      LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR,
                                      LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_A_NAME_STR,
                                      LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_B_NAME_STR,
                                      LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_A_NAME_STR,
                                      LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_B_NAME_STR]
    # for the unique sequence distance df
    unique_seq_dist_per_group_fields_list = [GROUP_FIELD_NAME_STR,
                                             LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR,
                                             LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR,
                                             LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_A_NAME_STR,
                                             LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_B_NAME_STR]
    # for both sequence length dfs
    seq_len_per_group_fields_list = [GROUP_FIELD_NAME_STR,
                                     LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR,
                                     LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]
                                            
    # list of field names used for aggregation + list of field names used for naming aggregated fields in the respective dataframes
    # fields to be aggregated for the avg seq dist and avg unique seq dist dfs
    seq_distance_fields = [LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR, 
                           LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR]
    # names of the aggregated fields
    avg_seq_dist_fields_list = [LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR]
    
    # fields to be aggregated for the avg seq len and avg unique seq len dfs
    seq_length_fields = [LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]
    # names of the aggregated fields
    avg_seq_length_fields_list = [LEARNING_ACTIVITY_MEAN_SEQUENCE_LENGTH_NAME_STR,
                                  LEARNING_ACTIVITY_MEDIAN_SEQUENCE_LENGTH_NAME_STR]

    def __init__(self, 
                 dataset_name: str,
                 interactions: pd.DataFrame,
                 unique_learning_activity_sequence_stats_per_group: pd.DataFrame,
                 learning_activity_sequence_stats_per_group: pd.DataFrame,
                 result_tables: Type[Any]) -> None:
        self.dataset_name = dataset_name
        self.interactions = interactions.copy()
        self.unique_learning_activity_sequence_stats_per_group = unique_learning_activity_sequence_stats_per_group.copy()
        self.learning_activity_sequence_stats_per_group = learning_activity_sequence_stats_per_group.copy()


        # specify sequence distance directory where pickle files will be read from
        self.sequence_distance_directory_list = [PATH_TO_SEQUENCE_DISTANCES_PICKLE_FOLDER, 
                                                 self.dataset_name]

        # specify sequence distance analytics data directory where pickle files will be written to and read from
        self.sequence_distance_analytics_data_directory_list = [PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_DATA_PICKLE_FOLDER,
                                                                self.dataset_name]

        # specify sequence distance matrix directory where pickle files will be written to
        self.sequence_distance_analytics_distance_matrix_directory_list = [PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_DISTANCE_SQUARE_MATRIX_PICKLE_FOLDER,
                                                                           self.dataset_name]

        # delete old pickle files to prevent keeping results for groups which are not part of current calculation anymore
        delete_all_pickle_files_within_directory(self.sequence_distance_analytics_data_directory_list)
        delete_all_pickle_files_within_directory(self.sequence_distance_analytics_distance_matrix_directory_list)

        ################################################################################################################
        # calculate the sequence distance and sequence length dataframes and write to disk as pickle in order to save memory
        # sequence distance dataframes
        self.seq_dist_data_param = (LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR, 
                                    self.seq_dist_per_group_fields_list,
                                    self.sequence_distance_directory_list,
                                    SequenceDistanceAnalytics._add_normalized_seq_dist_field)
        self.seq_dist_data_name = self.dataset_name + SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_DATA_PICKLE_NAME

        self.unique_seq_dist_data_param = (LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR, 
                                           self.unique_seq_dist_per_group_fields_list,
                                           self.sequence_distance_directory_list,
                                           SequenceDistanceAnalytics._add_normalized_seq_dist_field)
        self.unique_seq_dist_data_name = self.dataset_name + SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_DIST_DATA_PICKLE_NAME

        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.seq_dist_data_param, insert_position=2),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.seq_dist_data_name)
        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.unique_seq_dist_data_param, insert_position=2),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.unique_seq_dist_data_name)
        
        # sequence length dataframes
        self.seq_len_data_param = (LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR, 
                                   self.seq_len_per_group_fields_list,
                                   self.sequence_distance_directory_list)
        self.seq_len_data_name = self.dataset_name + SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_LEN_DATA_PICKLE_NAME

        self.unique_seq_len_data_param = (LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR, 
                                          self.seq_len_per_group_fields_list,
                                          self.sequence_distance_directory_list)
        self.unique_seq_len_data_name = self.dataset_name + SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_LEN_DATA_PICKLE_NAME

        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.seq_len_data_param),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.seq_len_data_name)
        self._write_object_as_pickle(self._generate_df_all_group_from_seq_dist_dict(*self.unique_seq_len_data_param),
                                     self.sequence_distance_analytics_data_directory_list,
                                     self.unique_seq_len_data_name)

        ################################################################################################################
        # avg sequence distance dataframes - entities = all_sequences/unique_sequences per group. 
        # the distance value for a sequence (or unique sequence) is the aggregate(mean/median) seq distance to all other sequences (all other unique sequences)
        # -> all sequences in unique_learning_activity_sequence_stats_per_group and learning_activity_sequence_stats_per_group receive
        # a mean/median distance metric value based on either 
        # 1. the mean/median distance of a respective sequence with all other sequences (includes duplicates) within a group
        # 2. the mean/median distance of a respective unique sequence with all other unique sequences (no duplicates included) within a group
        self.avg_dist_per_seq_per_group_base_all_seq_df = self.return_avg_sequence_distance_per_sequences_per_group_df(SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR)
        self.avg_dist_per_seq_per_group_base_unique_seq_df = self.return_avg_sequence_distance_per_sequences_per_group_df(SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_UNIQUE_SEQ_NAME_STR)

        # add avg seq distances per sequence with all other sequences to the sequence stats dataframe
        # sequence base: all seq
        self.learning_activity_sequence_stats_per_group = self._add_seq_dist_to_seq_stats_per_group(SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR,
                                                                                                    self.learning_activity_sequence_stats_per_group)
        self.unique_learning_activity_sequence_stats_per_group = self._add_seq_dist_to_seq_stats_per_group(SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR,
                                                                                                           self.unique_learning_activity_sequence_stats_per_group)
        # sequence base: unique seq
        self.learning_activity_sequence_stats_per_group = self._add_seq_dist_to_seq_stats_per_group(SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_UNIQUE_SEQ_NAME_STR,
                                                                                                    self.learning_activity_sequence_stats_per_group)
        self.unique_learning_activity_sequence_stats_per_group = self._add_seq_dist_to_seq_stats_per_group(SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_UNIQUE_SEQ_NAME_STR,
                                                                                                           self.unique_learning_activity_sequence_stats_per_group)

        # add data to results_table
        result_tables.unique_learning_activity_sequence_stats_per_group = self.unique_learning_activity_sequence_stats_per_group.copy()
        result_tables.learning_activity_sequence_stats_per_group = self.learning_activity_sequence_stats_per_group.copy()

        ################################################################################################################
        # avg sequence distance dataframes - entities = user/sequence_id combinations distances aggregated over groups (more distance pairs than actual sequences)
        self.avg_dist_per_all_seq_combinations_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                                             self.seq_dist_data_name),
                                                                                              SequenceDistanceAnalytics.seq_distance_fields,
                                                                                              SequenceDistanceAnalytics.avg_seq_dist_fields_list)
        self.avg_dist_per_unique_seq_combinations_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                                                self.unique_seq_dist_data_name),
                                                                                                 SequenceDistanceAnalytics.seq_distance_fields,
                                                                                                 SequenceDistanceAnalytics.avg_seq_dist_fields_list)

        # avg sequence length dataframes
        self.avg_seq_length_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                          self.seq_len_data_name),
                                                                           SequenceDistanceAnalytics.seq_length_fields,
                                                                           SequenceDistanceAnalytics.avg_seq_length_fields_list)
        self.avg_unique_seq_length_per_group_df = self._generate_df_avg_per_group(self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                                                                                 self.unique_seq_len_data_name),
                                                                                  SequenceDistanceAnalytics.seq_length_fields,
                                                                                  SequenceDistanceAnalytics.avg_seq_length_fields_list)

        # merged sequence distance - sequence length dataframes
        # base: user combinations
        self.avg_dist_length_per_all_seq_combinations_per_group_df = self._merge_df_on_group(self.avg_dist_per_all_seq_combinations_per_group_df,
                                                                                             self.avg_seq_length_per_group_df)
        # base: sequence combinations
        self.avg_dist_length_per_unique_seq_combinations_per_group_df = self._merge_df_on_group(self.avg_dist_per_unique_seq_combinations_per_group_df,
                                                                                                self.avg_unique_seq_length_per_group_df)

        ################################################################################################################
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

        grouper = GROUP_FIELD_NAME_STR
        df_name_group_list = [DATASET_NAME_STR, grouper]
        avg_per_group_df = (data.groupby(grouper)[fields_to_aggregate]
                                .agg([np.mean, np.median]))
        avg_per_group_df = avg_per_group_df.droplevel(0, axis=1)
        avg_per_group_df = avg_per_group_df.reset_index()
        avg_per_group_df.insert(0, DATASET_NAME_STR, self.dataset_name)
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
             
        seq_dist = df[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
        max_seq_len = df[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
        normalized_seq_dist = seq_dist / max_seq_len
        df.insert(insert_position, 
                  LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR, 
                  normalized_seq_dist)

        return df

    @classmethod
    def _merge_df_on_group(cls,
                           df_left: pd.DataFrame,
                           df_right: pd.DataFrame) -> pd.DataFrame:

        merged_df = pd.merge(df_left, 
                             df_right, 
                             how='inner', 
                             on=[DATASET_NAME_STR, 
                                 GROUP_FIELD_NAME_STR])

        return merged_df
    
    def _write_object_as_pickle(self,
                                object_to_pickle,
                                path_within_pickle_directory_list: list[str],
                                file_name: str) -> None:

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
            file_name_substring = SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_DIST_SQUARE_MATRIX_NORMALIZED_PICKLE_NAME
        elif (use_unique_sequence_distances and not normalize_distance):
            file_name_substring = SEQUENCE_DISTANCE_ANALYTICS_UNIQUE_SEQUENCE_DIST_SQUARE_MATRIX_PICKLE_NAME
        elif (not use_unique_sequence_distances and normalize_distance):
            file_name_substring = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_NORMALIZED_PICKLE_NAME
        else:
            file_name_substring = SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_PICKLE_NAME

        file_name = self.dataset_name + file_name_substring + GROUP_FIELD_NAME_STR + '_'

        return file_name

    def _generate_seq_dist_square_matrix_per_group_dict(self,
                                                        normalize_distance: bool,
                                                        use_unique_sequence_distances: bool) -> None:

        if use_unique_sequence_distances:
            sequence_distance_base = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR
            label_type = LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR
        else:
            sequence_distance_base = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR
            label_type = LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR

        seq_dist_fields_list = [LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR,
                                LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
        
        group_label_fields_list = [GROUP_FIELD_NAME_STR, 
                                   label_type]

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
                distances = seq_dist_df[LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR]
            else:
                distances = seq_dist_df[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
            
            labels = list(group_label_df[label_type])
            group = group_label_df[GROUP_FIELD_NAME_STR][0]

            square_matrix = squareform(distances)
        
            square_matrix_df = pd.DataFrame(square_matrix,
                                            index=labels, 
                                            columns=labels) 
            labels = sorted(labels, key=int)
            square_matrix_df = square_matrix_df.loc[labels, labels]

            file_name_substring = self._return_seq_dist_square_matrix_per_group_substring(normalize_distance,
                                                                                          use_unique_sequence_distances)
            file_name = file_name_substring + str(group)

            square_matrix_result_dict = {GROUP_FIELD_NAME_STR: int(group),
                                         SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_FIELD_NAME_STR: square_matrix_df}

            self._write_object_as_pickle(square_matrix_result_dict,
                                         self.sequence_distance_analytics_distance_matrix_directory_list,
                                         file_name)

    @classmethod
    def _unpivot_seq_dist_square_matrix(cls,
                                        square_matrix_result_dict: dict) -> pd.DataFrame:
            
        square_matrix_df = square_matrix_result_dict[SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_FIELD_NAME_STR]
        group = square_matrix_result_dict[GROUP_FIELD_NAME_STR]

        square_matrix_df = (pd.melt(square_matrix_df, 
                                    ignore_index=False, 
                                    var_name='row', 
                                    value_name='value')
                                .reset_index()
                                .rename(columns={'index': 'column'}))
        square_matrix_df[GROUP_FIELD_NAME_STR] = group

        return square_matrix_df

    def _add_seq_dist_to_seq_stats_per_group(self,
                                             distance_base: Literal['all_sequences', 'unique_sequences'],
                                             seq_stats_per_group_df: pd.DataFrame) -> pd.DataFrame:

        seq_stats_per_group_df[GROUP_FIELD_NAME_STR] = seq_stats_per_group_df[GROUP_FIELD_NAME_STR]
        seq_stats_per_group_df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR] = seq_stats_per_group_df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]

        # eliminate repeated sequences in order to perform correct merging
        if distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR:
            df_list = []
            for _, df in self.avg_dist_per_seq_per_group_base_all_seq_df.groupby(GROUP_FIELD_NAME_STR):
                df = df.loc[~df.duplicated(LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR), :]
                df_list.append(df)

            seq_distances_df_per_group_uniq_seq = pd.concat(df_list)

            seq_distances_df_per_group_uniq_seq = seq_distances_df_per_group_uniq_seq[[GROUP_FIELD_NAME_STR,
                                                                                       LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR]]
        else:
            seq_distances_df_per_group_uniq_seq = self.avg_dist_per_seq_per_group_base_unique_seq_df 

            seq_distances_df_per_group_uniq_seq = seq_distances_df_per_group_uniq_seq[[GROUP_FIELD_NAME_STR,
                                                                                       LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                                                                       LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR]]

        seq_stats_per_group_df = pd.merge(seq_stats_per_group_df,
                                          seq_distances_df_per_group_uniq_seq,
                                          how='left',
                                          on=[GROUP_FIELD_NAME_STR, 
                                              LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR])

        return seq_stats_per_group_df

    def return_sequence_distance_df(self,
                                    distance_base: Literal['all_sequences', 'unique_sequences']) -> pd.DataFrame:
        """Returns a dataframe containing sequence distances per group. Sequence distances can be based on
        all or unique sequences per group

        Parameters
        ----------
        distance_base : Literal[all_sequences, unique_sequences]
            The base for sequence distance calculations

        Returns
        -------
        pd.DataFrame
            A dataframe containing sequence distances per group
        """                                    

        if distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR:
            data_name = self.seq_dist_data_name

        elif distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_UNIQUE_SEQ_NAME_STR:
            data_name = self.unique_seq_dist_data_name

        seq_dist_data = self._read_object_frome_pickle(self.sequence_distance_analytics_data_directory_list,
                                                       data_name)

        return seq_dist_data

    def return_avg_sequence_distance_per_sequences_per_group_df(self,
                                                                distance_base: Literal['all_sequences', 'unique_sequences']) -> pd.DataFrame:
        """Returns a dataframe containing average sequence distances per sequence per group. The distance between a sequence 
        and all other sequences is used for averaging. Depending on distance_base all or only the unique sequences are used 
        in the calculations.     

        Parameters
        ----------
        distance_base : Literal[all_sequences, unique_sequences]
            The base for sequence distance calculations

        Returns
        -------
        pd.DataFrame
            A dataframe containing average sequence distances per sequence per group.
        """
        user_seq_id_mapping_per_group_dict = {}
        seq_id_user_mapping_per_group_dict = {}
        for group, df in self.interactions.groupby(GROUP_FIELD_NAME_STR):
            user_seq_id_mapping = dict(zip(df[USER_FIELD_NAME_STR], df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]))
            seq_id_user_mapping = dict(zip(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR], df[USER_FIELD_NAME_STR]))

            user_seq_id_mapping_per_group_dict[group] = user_seq_id_mapping
            seq_id_user_mapping_per_group_dict[group] = seq_id_user_mapping

        if distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR:
            seq_entity_field_name =  USER_FIELD_NAME_STR
            seq_entity_a_field_name = LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_A_NAME_STR
            seq_entity_b_field_name = LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_B_NAME_STR
            avg_seq_entities_per_group = user_seq_id_mapping_per_group_dict
            seq_distance_aggregate_field_names = [[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                                   LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR],
                                                  [LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                                   LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR]]

        elif distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_UNIQUE_SEQ_NAME_STR:
            seq_entity_field_name =  LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR
            seq_entity_a_field_name = LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_A_NAME_STR
            seq_entity_b_field_name = LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_B_NAME_STR
            avg_seq_entities_per_group = seq_id_user_mapping_per_group_dict
            seq_distance_aggregate_field_names = [[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                                   LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR],
                                                  [LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                                   LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR]]

        data = self.return_sequence_distance_df(distance_base)

        seq_distance_types = [LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR,
                              LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR]
             
        seq_distances_per_seq_entities_df_list = []
        for group, df in data.groupby(GROUP_FIELD_NAME_STR):

            avg_seq_dist_entities = list(avg_seq_entities_per_group[group].keys())

            seq_dist_types = []
            for seq_dist, agg_label_list in zip(seq_distance_types, 
                                                seq_distance_aggregate_field_names):
                a = df.groupby(seq_entity_a_field_name)[seq_dist].agg(tuple)
                dif_a = np.setdiff1d(avg_seq_dist_entities, a.index)
                dif_a_series = pd.Series([()]*len(dif_a), index=dif_a)
                a = pd.concat([a, dif_a_series], axis=0)

                b = df.groupby(seq_entity_b_field_name)[seq_dist].agg(tuple)
                dif_b = np.setdiff1d(avg_seq_dist_entities, b.index)
                dif_b_series = pd.Series([()]*len(dif_b), index=dif_b)
                b = pd.concat([b, dif_b_series], axis=0)

                seq_distances_per_seq_entity = a.add(b)
                seq_distances_per_seq_entity.index.name = seq_entity_field_name

                seq_distances_per_seq_entity = seq_distances_per_seq_entity.reset_index(name=seq_dist)
                
                seq_distances_per_seq_entity[agg_label_list[0]] = seq_distances_per_seq_entity[seq_dist].apply(np.mean)
                seq_distances_per_seq_entity[agg_label_list[1]] = seq_distances_per_seq_entity[seq_dist].apply(np.median)

                seq_dist_types.append(seq_distances_per_seq_entity)
            
            seq_distances_per_seq_entities_df = pd.merge(seq_dist_types[0],
                                                         seq_dist_types[1],
                                                         how='inner',
                                                         on=seq_entity_field_name)

            if distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR:
                seq_id_series = seq_distances_per_seq_entities_df[seq_entity_field_name].map(avg_seq_entities_per_group[group])
                seq_distances_per_seq_entities_df.insert(1,
                                                         LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR,
                                                         seq_id_series)
            
            seq_distances_per_seq_entities_df.insert(0, 
                                                     GROUP_FIELD_NAME_STR,
                                                     group)

            seq_distances_per_seq_entities_df_list.append(seq_distances_per_seq_entities_df)

        seq_distances_per_seq_entities_df_per_group = pd.concat(seq_distances_per_seq_entities_df_list,
                                                                ignore_index=True)

        seq_distances_per_seq_entities_df_per_group = seq_distances_per_seq_entities_df_per_group.drop(labels=seq_distance_types,
                                                                                                       axis=1)

        seq_distances_per_seq_entities_df_per_group[GROUP_FIELD_NAME_STR] = seq_distances_per_seq_entities_df_per_group[GROUP_FIELD_NAME_STR]
        seq_distances_per_seq_entities_df_per_group[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR] = seq_distances_per_seq_entities_df_per_group[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]

        return seq_distances_per_seq_entities_df_per_group

    def return_sequence_distance_matrix_per_group(self,
                                                  group: int,
                                                  normalize_distance: bool,
                                                  use_unique_sequence_distances: bool) -> dict:
        """Returns a sequence distance matrix for the specified group.

        Parameters
        ----------
        group : list of str, optional
            The group for which the distance matrix will be returned
        normalize_distance : bool
            A boolean indicating whether the sequence distances are being normalized between 0 and 1
        use_unique_sequence_distances: bool
            A boolean indicating whether only unique sequences are being used as the basis for distance calculations

        Returns
        -------
        dict
            A dictionary containing the group string and the corresponding sequence distance matrix
        """
        if isinstance(group, int):
            try:
                file_name_substring = self._return_seq_dist_square_matrix_per_group_substring(normalize_distance,
                                                                                              use_unique_sequence_distances)

                pickle_files_list = return_pickled_files_list(self.sequence_distance_analytics_distance_matrix_directory_list,
                                                              file_name_substring)

                # filter the filenames by groups in group_str
                pickle_files_list = [i for i in pickle_files_list if int(i.split('_')[-1]) in [group]]

                distance_matrix_dict = self._read_object_frome_pickle(self.sequence_distance_analytics_distance_matrix_directory_list,     
                                                                      pickle_files_list[0])
                
                return distance_matrix_dict

            except: ValueError(f'There is no distance matrix associated with group: {group}')

        else:
            raise TypeError('group_str needs to be of type int')

    def plot_sequence_distances_per_sequence_per_group(self,
                                                       distance_avg_metric: Literal['mean', 'median'],
                                                       distance_base: Literal['all_sequences', 'unique_sequences']) -> None:
        """Plot various distribution plots for the average sequence distances per sequence per group.

        Parameters
        ----------
        distance_avg_metric : Literal[mean, median]
            The function used for calculating average sequence distances per sequence
        distance_base : Literal[all_sequences, unique_sequences]
            The base for sequence distance calculations. 
            1. all_sequences: calculates averages based on the distances of combinations between all sequences  
            1. unique_sequences: calculates averages based on the distances of combinations between unique sequences  

        Returns
        ------
        None
        """
        if distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_ALL_SEQ_NAME_STR:

            seq_dist_measures = [[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                  LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR],
                                 [LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                  LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR]]       

            seq_dist_labels = [[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_LABEL_NAME_STR,
                                LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_LABEL_NAME_STR],
                               [LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_LABEL_NAME_STR,
                                LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_LABEL_NAME_STR]]       

            stat_plot_titles = [[[SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR],
                                 [SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR]],
                                [[SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR],
                                 [SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_TITLE_NAME_STR]]]

            dist_vs_freq_scatter_titles = [[[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR]],
                                           [[LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR]]]

            dist_vs_stat_scatter_titles = [[[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR]],
                                           [[LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_ALL_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR]]]

        elif distance_base == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_UNIQUE_SEQ_NAME_STR:

            seq_dist_measures = [[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                  LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR],
                                 [LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR,
                                  LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_NAME_STR]]       

            seq_dist_labels = [[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_LABEL_NAME_STR,
                                LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_LABEL_NAME_STR],
                               [LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_LABEL_NAME_STR,
                                LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_LABEL_NAME_STR]]       

            stat_plot_titles = [[[SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR],
                                 [SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR]],
                                [[SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR],
                                 [SEQUENCE_DISTANCE_ANALYTICS_BOXPLOT_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_MIN_VS_MEDIAN_VS_MAX_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR,
                                  SEQUENCE_DISTANCE_ANALYTICS_ECDF_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_TITLE_NAME_STR]]]

            dist_vs_freq_scatter_titles = [[[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR]],
                                           [[LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR]]]
                                            
            dist_vs_stat_scatter_titles = [[[LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR]],
                                           [[LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR],
                                            [LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_SEQUENCE_LENGTH_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                                             LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQ_VS_REPEATED_LEARNING_ACTIVITIES_PCT_PER_GROUP_TITLE_NAME_STR]]]
        else:
            raise ValueError('distance_base must be either of ["all_sequences", "unique_sequences"]')

        seq_dist_measure_is_pct = [False,
                                   True]

        seq_dist_measure_is_ratio = [False,
                                     True]
        
        if distance_avg_metric == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_AVG_MEAN_NAME_STR:
            idx = 0
        elif distance_avg_metric == SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_AVG_MEDIAN_NAME_STR:
            idx = 1
        else:
            raise ValueError('distance_avg_metric must be either of ["mean", "median"]')
        
        plotting_data = zip(seq_dist_measures[idx],
                            seq_dist_labels[idx],
                            stat_plot_titles[idx],
                            dist_vs_freq_scatter_titles[idx],
                            dist_vs_stat_scatter_titles[idx],
                            seq_dist_measure_is_pct,
                            seq_dist_measure_is_ratio)
            
        # create aggregated dataframes for plotting    
        learning_activity_sequence_stats_merged = pd.concat([self.learning_activity_sequence_stats_per_group,
                                                             self.unique_learning_activity_sequence_stats_per_group],
                                                             axis=0)

        for (distance_measure, 
             label, 
             stat_plot_titles, 
             dist_vs_freq_titles, 
             dist_vs_stat_titles, 
             distance_measure_is_pct, 
             distance_measure_is_ratio) in plotting_data:

            # data in long format for pointplot
            seq_dist_per_sequence_stats_per_group_long = SequenceStatistics.return_aggregated_statistic_per_group(self.learning_activity_sequence_stats_per_group,
                                                                                                                  distance_measure,
                                                                                                                  LEARNING_ACTIVITY_SEQUENCE_TYPE_ALL_SEQ_VALUE_STR)
            unique_seq_dist_per_sequence_stats_per_group_long = SequenceStatistics.return_aggregated_statistic_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                                                                         distance_measure,
                                                                                                                         LEARNING_ACTIVITY_SEQUENCE_TYPE_UNIQUE_SEQ_VALUE_STR)
            seq_dist_per_sequence_stats_per_group_long_merged = pd.concat([seq_dist_per_sequence_stats_per_group_long,
                                                                           unique_seq_dist_per_sequence_stats_per_group_long], 
                                                                           axis=0)
        
            # all sequence vs unique sequences - sequence distance statistics plots
            print(' ')
            print(STAR_STRING)
            print(STAR_STRING)
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQ_DIST_PER_GROUP_TITLE_NAME_STR}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_TYPE_TITLE_NAME_STR}')
            print(f'{distance_measure}')  
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_IS_NORMALIZED_TITLE_NAME_STR}')
            print(distance_measure_is_ratio)
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_TITLE_NAME_STR}')
            print(f'{distance_base}')  
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_AVG_METHOD_TITLE_NAME_STR}')
            print(f'{distance_avg_metric}')
            print(' ')
            print(f'{LEARNING_ACTIVITY_STATS_SEQUENCE_FACET_BASE_STR}')
            print(LEARNING_ACTIVITY_STATS_SEQUENCE_NAME_STR)
            print('VS')
            print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
            print(' ')
            plot_stat_plot(learning_activity_sequence_stats_merged,
                           seq_dist_per_sequence_stats_per_group_long_merged,
                           distance_measure,
                           distance_measure_is_pct,
                           distance_measure_is_ratio,
                           stat_plot_titles[0],
                           stat_plot_titles[1],
                           stat_plot_titles[2])
        
            # all sequence vs unique sequences - sequence distance histogram
            print(' ')
            print(STAR_STRING)
            print(STAR_STRING)
            print(' ')
            print(f'{LEARNING_ACTIVITY_SEQ_DIST_HISTOGRAM_PER_GROUP_TITLE_NAME_STR}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_TYPE_TITLE_NAME_STR}')
            print(f'{distance_measure}')  
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_IS_NORMALIZED_TITLE_NAME_STR}')
            print(distance_measure_is_ratio)
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_TITLE_NAME_STR}')
            print(f'{distance_base}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_AVG_METHOD_TITLE_NAME_STR}')
            print(f'{distance_avg_metric}')
            print(' ')
            print(f'{LEARNING_ACTIVITY_STATS_SEQUENCE_FACET_BASE_STR}')
            print(LEARNING_ACTIVITY_STATS_SEQUENCE_NAME_STR)
            print('VS')
            print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
            print(' ')
            plot_stat_hist_plot_per_group(learning_activity_sequence_stats_merged,
                                          distance_measure,
                                          distance_measure_is_pct,
                                          distance_measure_is_ratio,
                                          label,
                                          LEARNING_ACTIVITY_SEQ_DIST_HISTOGRAM_PER_GROUP_TITLE_NAME_STR,
                                          True,
                                          True)

            # unique sequences per group figures - sequences distance vs frequency
            print(' ')
            print(STAR_STRING)
            print(STAR_STRING)
            print(' ')
            print(f'{LEARNING_ACTIVITY_UNIQUE_SEQUENCES_SEQ_DIST_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_TYPE_TITLE_NAME_STR}')
            print(f'{distance_measure}')  
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_IS_NORMALIZED_TITLE_NAME_STR}')
            print(distance_measure_is_ratio)
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_TITLE_NAME_STR}')
            print(f'{distance_base}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_AVG_METHOD_TITLE_NAME_STR}')
            print(f'{distance_avg_metric}')
            print(' ')
            print(f'{LEARNING_ACTIVITY_STATS_SEQUENCE_FACET_BASE_STR}')
            print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
            print(' ')
            plot_stat_scatter_plot_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                             LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR,
                                             distance_measure,
                                             LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR,
                                             label,
                                             True,
                                             distance_measure_is_pct,
                                             False,
                                             distance_measure_is_ratio,
                                             dist_vs_freq_titles[0],
                                             True,
                                             True)
            plot_stat_scatter_plot_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                             LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR,
                                             distance_measure,
                                             LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR,
                                             label,
                                             False,
                                             distance_measure_is_pct,
                                             False,
                                             distance_measure_is_ratio,
                                             dist_vs_freq_titles[1],
                                             True,
                                             True)
                                             
            # unique sequences per group figures - sequences distance vs stats
            print(' ')
            print(STAR_STRING)
            print(STAR_STRING)
            print(' ')
            print(f'{LEARNING_ACTIVITY_UNIQUE_SEQUENCES_SEQ_DIST_VS_SEQ_STATS_PER_GROUP_TITLE_NAME_STR}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_TYPE_TITLE_NAME_STR}')
            print(f'{distance_measure}')  
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_IS_NORMALIZED_TITLE_NAME_STR}')
            print(distance_measure_is_ratio)
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_BASE_TITLE_NAME_STR}')
            print(f'{distance_base}')
            print(' ')
            print(f'{SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_AVG_METHOD_TITLE_NAME_STR}')
            print(f'{distance_avg_metric}')
            print(' ')
            print(f'{LEARNING_ACTIVITY_STATS_SEQUENCE_FACET_BASE_STR}')
            print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
            print(' ')
            plot_stat_scatter_plot_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                             LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                             distance_measure,
                                             LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                             label,
                                             False,
                                             distance_measure_is_pct,
                                             False,
                                             distance_measure_is_ratio,
                                             dist_vs_stat_titles[0],
                                             True,
                                             True)
            plot_stat_scatter_plot_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                             LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                                             distance_measure,
                                             LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
                                             label,
                                             True,
                                             distance_measure_is_pct,
                                             False,
                                             distance_measure_is_ratio,
                                             dist_vs_stat_titles[1],
                                             True,
                                             True)
            plot_stat_scatter_plot_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                             LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                                             distance_measure,
                                             LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
                                             label,
                                             True,
                                             distance_measure_is_pct,
                                             False,
                                             distance_measure_is_ratio,
                                             dist_vs_stat_titles[2],
                                             True,
                                             True)

    def plot_sequence_distance_matrix_per_group(self,
                                                normalize_distance: bool,
                                                use_unique_sequence_distances: bool,
                                                height: Union[int, None],
                                                group_list=Union[list[int], None]) -> None:
        """Plot the sequence distance matrix group.

        Parameters
        ----------
        normalize_distance : bool
            A boolean indicating whether the sequence distances are being normalized between 0 and 1
        use_unique_sequence_distances: bool
            A boolean indicating whether only unique sequences are being used as the basis for distance calculations
        height : Union[int, None]
            The height of the subplots. If None the standard height for quadratic facet plots will be used 
        group_list : list of str, optional
            A list of groups for which the sequence distance matrices will be plotted. If None all groups will de displayed
        """
        if not group_list:
            group_list = list(self.unique_learning_activity_sequence_stats_per_group[GROUP_FIELD_NAME_STR].unique())

        # use a generator expression to reduce memory usage
        square_matrix_df_generator = (self._unpivot_seq_dist_square_matrix(
                                      self.return_sequence_distance_matrix_per_group(int(group),
                                                                                     normalize_distance,
                                                                                     use_unique_sequence_distances)) 
                                      for group in group_list)
        
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
            distance_str = f'Normalized {SEQUENCE_STR} Distance Matrix per {GROUP_FIELD_NAME_STR}:'
        else:
            distance_str = f'{SEQUENCE_STR} Distance Matrix per {GROUP_FIELD_NAME_STR}:'

        if use_unique_sequence_distances:
            base_str = f'Base: All Unique-{SEQUENCE_STR} Combinations'
            label = SEQUENCE_ID_FIELD_NAME_STR
        else:
            base_str = f'Base: All {USER_FIELD_NAME_STR}-{SEQUENCE_STR} Combinations' 
            label = USER_FIELD_NAME_STR
        
        if not height:  
            height = SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_DISTANCE_MATRIX
        

        n_cols = set_facet_grid_column_number(group_list,
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(DASH_STRING)
        print(distance_str)
        print(' ')
        print(base_str)
        print(DASH_STRING)
        g = sns.FacetGrid(square_matrices_per_group_df, 
                          col=GROUP_FIELD_NAME_STR,
                          col_wrap=n_cols, 
                          sharex=False,
                          sharey=False,
                          height=height, 
                          aspect= SEABORN_FIGURE_LEVEL_ASPECT_SQUARE)
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
        g.set(xlabel=label, 
              ylabel=label)
        # get figure background color
        facecolor=plt.gcf().get_facecolor()
        for ax in g.axes.flat:
            # set aspect of all axis
            ax.set_aspect('equal','box')
            # set background color of axis instance
            ax.set_facecolor(facecolor)
        plt.show()