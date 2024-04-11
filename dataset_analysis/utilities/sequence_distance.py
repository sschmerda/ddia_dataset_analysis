from .standard_import import *
from .preprocessing_functions import *
from .constants import *
from .io_functions import *

class SeqDist:
    """
    A class to calculate distances of learning activity sequences for different groups.

    Parameters
    ----------
    data: DataFrame
        A dataframe containing group, user and learning activity fields.
    user_field: str
        Then name of the user field.
    group_field: str
        Then name of the group field.
    learning_activity_field: str
        Then name of the learning activity field.
    sequence_id_field: str
        Then name of the sequence id field.
    has_groups: str
        A flag indicating whether the learning activity sequences in data are separated into different groups.

    Methods
    -------
    calc_user_sequence_distances_per_group(distance function,\
                                          min_number_of_users_per_group=None,\
                                          min_number_avg_seq_len=None,\
                                          top_n_groups_by_user_number=None,\
                                          is_pct_top_n_groups_by_user_number,\
                                          top_n_groups_by_median_seq_len=None,\
                                          is_pct_top_n_groups_by_median_seq_len,\
                                          sample_pct=None):
    For each group calculates the (learning activity-) sequence distances between each possible user\
    combination(seq_distances) and sequence combination(unique_seq_distances) pair.
    """

    GROUP_COL_INDEX = 0
    USER_COL_INDEX = 1
    LEARNING_ACTIVITY_COL_INDEX = 2
    SEQUENCE_ID_COL_INDEX = 3
    LEARNING_ACTIVITY_CHAR_COL_INDEX = 4

    def __init__(self,
                 dataset_name: str,
                 data: pd.DataFrame,
                 user_field: str,
                 group_field: str,
                 learning_activity_field: str,
                 sequence_id_field: str,
                 has_groups: bool):

        self.dataset_name = dataset_name
        self.data = data.copy()
        self.user_field = user_field
        self.group_field = group_field
        self.learning_activity_field = learning_activity_field
        self.sequence_id_field = sequence_id_field

        if not has_groups:
            self.data[self.group_field] = 0

        # initial data transformation
        self._select_fields_and_transform_to_numpy()
        self._transform_to_char()

        self.group_array = np.unique(self.data[:, SeqDist.GROUP_COL_INDEX])
        self.user_array = np.unique(self.data[:, SeqDist.USER_COL_INDEX])

        # directory where sequence distance pickle files will be written to
        self.result_directory = [PATH_TO_SEQUENCE_DISTANCES_PICKLE_FOLDER,
                                 self.dataset_name]

    def _select_fields_and_transform_to_numpy(self) -> None:
        """Selects the fields from data used for sequence distance calculations
        """
        # data type of fields need to be transformed to string in order to hold characters for seq dist calculation in matrix
        self.data = (self.data[[self.group_field,
                                self.user_field,
                                self.learning_activity_field,
                                self.sequence_id_field,
                                self.learning_activity_field]].astype('str').values)

    def _transform_to_char(self) -> None:
        """Maps unique characters to learning activities and returns the transformed series (tokenization)

        Raises
        ------
        Exception
            Raised if there are more unique learning activities than characters.
        """
        unique_lr = np.unique(self.data[:, SeqDist.LEARNING_ACTIVITY_COL_INDEX])
        lr_mapping = {}
        chars = []

        i = 0
        while i >= 0:
            try:
                char = chr(i)
                chars.append(char)
                i += 1
            except:
                break

        # pandas cannot distinguish between some unicode characters -> reduce chars to the ones
        # readable by pandas
        chars = pd.Series(chars).unique()

        if len(unique_lr) > len(chars):
            raise Exception(f'There are more unique {LEARNING_ACTIVITY_FIELD_NAME_STR}s than possible characters')

        for n, i in enumerate(unique_lr):
            lr_mapping[i] = chars[n]

        char_array = [lr_mapping[i] for i in self.data[:, SeqDist.LEARNING_ACTIVITY_COL_INDEX]]

        self.data[:, SeqDist.LEARNING_ACTIVITY_CHAR_COL_INDEX] = char_array

    def _filter_data_by_group(self,
                              data: np.ndarray,
                              group: str) -> np.ndarray:
        """Filters data by group

        Parameters
        ----------
        data : np.ndarray
            The data matrix
        group : str
            The group which is used for filtering data and char_array

        Returns
        -------
        np.ndarray
            A np.ndarray containing the filtered data
        """
        data_copy = data.copy()
        data_copy = data_copy[data_copy[:, self.GROUP_COL_INDEX] == group, :]

        return data_copy

    def _calculate_sequence_distances(self,
                                      group: str,
                                      group_data: np.ndarray, 
                                      distance_function: Any, 
                                      *args, 
                                      **kwargs) -> None:
        """Calculates sequence distances(base all sequences and base unique sequences) for a specified group and saves the
        result dataframe to disk in the directory specified in self.result_directory.

        Parameters
        ----------
        group : str
            The group for which sequence distances will be calculated
        group_data : np.ndarray
            The data matrix for the specified group
        distance_function : Any
            A function for calculating the sequence distance

        Returns
        -------
        None
        """                                      
        user_array, idx = np.unique(group_data[:, self.USER_COL_INDEX], return_index=True)
        user_sequence_id_array = group_data[:, self.SEQUENCE_ID_COL_INDEX][idx]
        user_sequence_id_mapping_dict = dict(zip(user_array, user_sequence_id_array))

        users = group_data[:, self.USER_COL_INDEX]
        learning_activities_char = group_data[:, self.LEARNING_ACTIVITY_CHAR_COL_INDEX]
        learning_activities = group_data[:, self.LEARNING_ACTIVITY_COL_INDEX]

        # generate a user - sequence mapping + user sequence length/ user sequence array(learning activity ids)
        user_learning_activity_df = pd.DataFrame({USER_FIELD_NAME_STR: users,
                                                  LEARNING_ACTIVITY_SEQUENCE_CHARACTERS_NAME_STR: learning_activities_char,
                                                  LEARNING_ACTIVITY_FIELD_NAME_STR: learning_activities})
        user_char_string_mapping_dict = (user_learning_activity_df.groupby(USER_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_CHARACTERS_NAME_STR]
                                                                  .sum()
                                                                  .to_dict())
        len_seq_df = user_learning_activity_df.groupby(USER_FIELD_NAME_STR)[LEARNING_ACTIVITY_FIELD_NAME_STR].agg([len, tuple])

        user_sequence_length = len_seq_df['len'].to_list()
        user_sequence_array = len_seq_df['tuple'].to_list()
        user_sequence_array = [tuple(map(int, i)) for i in user_sequence_array]

        # generate a sequence_combination - sequence_distance/max_sequence_length mapping
        sequence_id_char_string_mapping_dict = {seq_id: user_char_string_mapping_dict[user] 
                                                for user, seq_id in user_sequence_id_mapping_dict.items()}
        sequence_id_array = list(sequence_id_char_string_mapping_dict.keys())
        sequence_len_array = [len(char_string) for char_string in sequence_id_char_string_mapping_dict.values()]

        sequence_combinations_with_replacement = [tuple(sorted(i, key=int)) for i in list(combinations_with_replacement(sequence_id_array, 2))]
        sequence_combinations = [tuple(sorted(i, key=int)) for i in list(combinations(sequence_id_array, 2))]

        sequence_distance_dict = {}
        for sequence_combination in sequence_combinations_with_replacement:

            char_sequence_1 = sequence_id_char_string_mapping_dict[sequence_combination[0]]
            char_sequence_2 = sequence_id_char_string_mapping_dict[sequence_combination[1]]
            sequence_distance = distance_function(char_sequence_1, char_sequence_2, *args, **kwargs)
            max_sequence_length = max(len(char_sequence_1), len(char_sequence_2))

            sequence_distance_dict[sequence_combination] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: sequence_distance,
                                                            LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_sequence_length}
        # generate the sequence_distances_per_group dictionary
        user_combinations = [tuple(sorted(i, key=int)) for i in combinations(user_array, 2)]

        sequence_distance_list = []
        max_sequence_length_list = []
        user_a_list = []
        user_b_list = []
        sequence_id_a_list = []
        sequence_id_b_list = []
        for user_combination in user_combinations:

            sequence_1 = user_sequence_id_mapping_dict[user_combination[0]]
            sequence_2 = user_sequence_id_mapping_dict[user_combination[1]]

            sequence_tuple = tuple(sorted((sequence_1, sequence_2), key=int))

            sequence_distance = sequence_distance_dict[sequence_tuple][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
            max_sequence_length = sequence_distance_dict[sequence_tuple][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
            sequence_id_combination = tuple(sorted((sequence_1, sequence_2), key=int))

            sequence_distance_list.append(sequence_distance)
            max_sequence_length_list.append(max_sequence_length)
            user_a_list.append(int(user_combination[0]))
            user_b_list.append(int(user_combination[1]))
            sequence_id_a_list.append(int(sequence_id_combination[0]))
            sequence_id_b_list.append(int(sequence_id_combination[1]))

        sequence_distances = {DATASET_NAME_STR: self.dataset_name,
                              GROUP_FIELD_NAME_STR: int(group),
                              LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: sequence_distance_list,
                              LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_sequence_length_list,
                              LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_A_NAME_STR: user_a_list,
                              LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_USER_B_NAME_STR: user_b_list,
                              LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_A_NAME_STR: sequence_id_a_list,
                              LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_B_NAME_STR: sequence_id_b_list,
                              LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR: list(map(int, user_array)),
                              LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: list(map(int, user_sequence_id_array)),
                              LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: user_sequence_length,
                              LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR: user_sequence_array}

        # generate the unique_sequence_distances_per_group dictionary
        unique_sequence_distance_list = []
        max_unique_sequence_length_list = []
        unique_sequence_id_a_list = []
        unique_sequence_id_b_list = []

        for sequence_combination in sequence_combinations:

            unique_sequence_distance = sequence_distance_dict[sequence_combination][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
            max_unique_sequence_length = sequence_distance_dict[sequence_combination][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]

            unique_sequence_distance_list.append(unique_sequence_distance)
            max_unique_sequence_length_list.append(max_unique_sequence_length)
            unique_sequence_id_a_list.append(int(sequence_combination[0]))
            unique_sequence_id_b_list.append(int(sequence_combination[1]))

        unique_sequence_distances = {DATASET_NAME_STR: self.dataset_name,
                                     GROUP_FIELD_NAME_STR: int(group),
                                     LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: unique_sequence_distance_list,
                                     LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_unique_sequence_length_list,
                                     LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_A_NAME_STR: unique_sequence_id_a_list,
                                     LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_SEQUENCE_B_NAME_STR: unique_sequence_id_b_list,
                                     LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: list(map(int, sequence_id_array)),
                                     LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: sequence_len_array}

        sequence_distances_dict = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR: sequence_distances,
                                   LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR: unique_sequence_distances}

        # write dictionary to disk
        filename = f'{self.dataset_name}{SEQUENCE_DISTANCE_DICT_PICKLE_NAME}{GROUP_FIELD_NAME_STR}_{group}'
        pickle_write(sequence_distances_dict, 
                     self.result_directory, filename)

        return None

    def _get_col1_by_min_number_col2(self,
                                     current_col1: np.ndarray,
                                     min_n_col2: int,
                                     col_1_index: int,
                                     col_2_index: int,
                                     col_1_name: str,
                                     col_2_name: str) -> np.ndarray:
        """Filter col1 vals by the number of uniques col2 vals who interacted with it.

        Parameters
        ----------
        current_col1 : ndarray
            An array of col1 val names.
        min_n_col2 : int
            The minimum number of col2 vals a col1 val is allowed to have.

        Returns
        -------
        ndarray
            An array containing the names of col1 vals with more unique col2 vals than the specified threshold.
        """    
        filtered_data = self.data[np.isin(self.data[:, col_1_index], current_col1), :]

        col1_vals = filtered_data[:, col_1_index]
        col2_vals = filtered_data[:, col_2_index]

        col1_array = np.array([col1 for col1 in current_col1 if np.unique(col2_vals[col1_vals==col1]).size >= min_n_col2])

        n_kept_col1s = col1_array.size

        print(DASH_STRING)
        print(f'Filtering out {col_1_name} with less than {min_n_col2} {col_2_name}:')
        print(f'Total number of {col_1_name} left: {n_kept_col1s}')
        print(DASH_STRING)

        return col1_array

    def _get_col1_by_min_number_median_seq_len(self,
                                               current_col1: np.ndarray,
                                               min_median_seq_len: int,
                                               col_1_index: int,
                                               col_2_index: int,
                                               col_1_name: str,
                                               col_2_name: str) -> np.ndarray:
        """Filter col1 vals by a minimum median sequence length of col2 vals.

        Parameters
        ----------
        current_col1 : ndarray
            An array of col1 val names.
        min_median_seq_len : int
            The minimum median sequence length of col2 vals a col1 val is allowed to have.

        Returns
        -------
        ndarray
            An array containing the names of col1 vals with a median sequence length of col2 vals of more than the\
            specified threshold.
        """
        filtered_data = self.data[np.isin(self.data[:, col_1_index], current_col1), :]

        col1_vals = filtered_data[:, col_1_index]
        col2_vals = filtered_data[:, col_2_index]
        
        col1_array = []

        for col1 in current_col1:
            filtered_col2 = col2_vals[col1_vals==col1]
            filtered_col2 = filtered_col2[np.argsort(filtered_col2)]
            vlen = np.vectorize(len)
            grouped_col2 = np.array(np.split(filtered_col2, np.unique(filtered_col2, return_index=True)[1][1:]))
            grouped_col2_len = vlen(grouped_col2)
            median_seq_len = np.median(grouped_col2_len)
            
            if median_seq_len >= min_median_seq_len:
                col1_array.append(col1)

        col1_array = np.array(col1_array)

        n_kept_col1s = col1_array.size

        print(DASH_STRING)
        print(f'Filtering out {col_1_name} with a median sequence length of {col_2_name} of less than {min_median_seq_len}:')
        print(f'Total number of {col_1_name} left: {n_kept_col1s}')
        print(DASH_STRING)
        
        return col1_array

    def _get_top_n_col1_by_col2(self,
                                current_col1: np.ndarray,
                                top_n: int,
                                is_pct: bool,
                                col_1_index: int,
                                col_2_index: int,
                                col_1_name: str,
                                col_2_name: str) -> np.ndarray:
        """Returns the names of the top_n col1 vals by number of unique col2 vals.

        Parameters
        ----------
        current_col1 : ndarray
            An array of col1 val names.
        top_n : int
            The number of top_n col1 vals by unique col2 vals.
        is_pct : bool
            A flag indicating whether top_n represents a percentage number.

        Returns 
        -------
        ndarray
            An array containing the names of the top_n (percent if is_pct == TRUE) col1 vals by number of col2 vals in\
            descending order.
        """    
        if is_pct:
            if (top_n <= 0) or (top_n > 100):
                raise Exception(f'top_n needs to be between 1 and 100. The value of sample_pct was {top_n}.')
        else:
            if top_n > current_col1.size:
                raise Exception(f'top_n needs to be smaller than the number of current_{col_1_name}. The value of top_n\
                                  was {top_n} and the number of current {col_1_name} is {current_col1.size}.')

        filtered_data = self.data[np.isin(self.data[:, col_1_index], current_col1), :]

        col1_vals = filtered_data[:, col_1_index]
        col2_vals = filtered_data[:, col_2_index]

        if is_pct:
            pct_kept = top_n
            top_n = int(round(current_col1.size * top_n / 100))

        sorting_array = np.argsort(np.array([np.unique(col2_vals[col1_vals==col1]).size for col1 in current_col1]))
        col1_array = current_col1[sorting_array][::-1][:top_n]

        n_kept_col1 = col1_array.size

        if is_pct:
            print(DASH_STRING)
            print(f'Keeping top {pct_kept}% of {col_1_name} by {col_2_name} size:')
            print(f'Total number of {col_1_name} left: {n_kept_col1}')
            print(DASH_STRING)
        else:
            print(DASH_STRING)
            print(f'Keeping top {top_n} {col_1_name} by {col_2_name} size:')
            print(f'Total number of groups left: {n_kept_col1}')
            print(DASH_STRING)

        return col1_array

    def _get_top_n_col1_by_median_seq_len(self,
                                          current_col1: np.ndarray,
                                          top_n: int,
                                          is_pct: bool,
                                          col_1_index: int,
                                          col_2_index: int,
                                          col_1_name: str,
                                          col_2_name: str) -> np.ndarray:
        """Return the names of the top_n col1 vals by median sequence length of col2 vals.

        Parameters
        ----------
        current_col1 : ndarray
            An array of col1 val names.
        top_n : int
            The number of top_n col1 vals by median sequence length of col2 vals.
        is_pct : bool
            A flag indicating whether top_n represents a percentage number.

        Returns
        -------
        ndarray
            An array containing the names of the top_n (percent if is_pct == TRUE) col1 vals by median sequence length\
            of col2 vals in descending order.
        """
        if is_pct:
            if (top_n <= 0) or (top_n > 100):
                raise Exception(f'top_n needs to be between 1 and 100. The value of sample_pct was {top_n}.')
        else:
            if top_n > current_col1.size:
                raise Exception(f'top_n needs to be smaller than the number of current_{col_1_name}. The value of top_n\
                     was {top_n} and the number of current {col_1_name} is {current_col1.size}.')

        filtered_data = self.data[np.isin(self.data[:, col_1_index], current_col1), :]

        col1_vals = filtered_data[:, col_1_index]
        col2_vals = filtered_data[:, col_2_index]

        if is_pct:
            pct_kept = top_n
            top_n = int(round(current_col1.size * top_n / 100))
        
        median_seq_len_array = []

        for col1 in current_col1:
            filtered_col2 = col2_vals[col1_vals==col1]
            filtered_col2 = filtered_col2[np.argsort(filtered_col2)]
            vlen = np.vectorize(len)
            grouped_col2 = np.array(np.split(filtered_col2 , np.unique(filtered_col2, return_index=True)[1][1:]))
            grouped_col2_len = vlen(grouped_col2)
            median_seq_len = np.median(grouped_col2_len)
            
            median_seq_len_array.append(median_seq_len)

        median_seq_len_array = np.array(median_seq_len_array)
        sorting_array = np.argsort(median_seq_len_array)
        col1_array = current_col1[sorting_array][::-1][:top_n]

        n_kept_col1 = col1_array.size
        
        if is_pct:
            print(DASH_STRING)
            print(f'Keeping top {pct_kept}% of {col_1_name} by median sequence length of {col_2_name}:')
            print(f'Total number of {col_1_name} left: {n_kept_col1}')
            print(DASH_STRING)
        else:
            print(DASH_STRING)
            print(f'Keeping top {top_n} {col_1_name} by median sequence length of {col_2_name}:')
            print(f'Total number of groups left: {n_kept_col1}')
            print(DASH_STRING)

        return col1_array
        

    @staticmethod
    def _sample_col1(current_col1: np.ndarray,
                        sample_pct: int,
                        col_1_name: str) -> np.ndarray:
        """Returns the names of sampled col1 vals

        Parameters
        ----------
        current_col1 : ndarray
            An array of col1 val names.
        sample_pct : int
            The percentage of col1 vals to be sampled. Must be between 1 and 100.

        Returns
        -------
        ndarray
            An array containing the names of the sampled col1 vals.

        Raises
        ------
        Exception
            Raised if the sample percentage is not beween 1 and 100.
        """

        if (sample_pct <= 0) or (sample_pct > 100):
            raise Exception(f'sample_pct needs to be between 1 and 100. The value of sample_pct was {sample_pct}.')

        n_col1_to_sample = int(round(current_col1.size * sample_pct / 100))
        col1_array = np.random.choice(current_col1, size=n_col1_to_sample, replace=False)

        n_kept_col1 = col1_array.size

        print(DASH_STRING)
        print(f'Sampling {sample_pct} percent of {col_1_name}:')
        print(f'Total number of {col_1_name} left: {n_kept_col1}')
        print(DASH_STRING)

        return col1_array

    def calc_user_sequence_distances_per_group(self,
                                               parallelize_computation: bool,
                                               distance_function,
                                               *args,
                                               min_number_of_users_per_group=None,
                                               min_number_avg_seq_len=None,
                                               top_n_groups_by_user_number=None,
                                               is_pct_top_n_groups_by_user_number=False,
                                               top_n_groups_by_median_seq_len=None,
                                               is_pct_top_n_groups_by_median_seq_len=False,
                                               sample_pct=None,
                                               sample_pct_user=None,
                                               **kwargs) -> None:
        """For each group calculates the (learning activity-) sequence distances between each possible user\
           combination(seq_distances) and sequence combination(unique_seq_distances) pair. The result,\
           a dictionary containing sequence distances between user combinations and sequence combinations,\
           will be saved as pickle file for each group. 

        Parameters
        ----------
         parallelize_computation : 
            A flag indicating whether the sequence distance calculations should be done in parallel
        distance_function : 
            A sequence distance function taking 2 strings as the first two positional arguments as input.
        *args :
            Positional arguments for the respective distance function.
        min_number_of_users_per_group : int, optional
            The minimum number of users a group is allowed to have., by default None
        min_number_avg_seq_len : int, optional
            The minimum median sequence length a group is allowed to have., by default None
        top_n_groups_by_user_number : int, optional
            The number of top_n groups by users to be selected., by default None
        is_pct_top_n_groups_by_user_number : bool
            A flag indicating whether top_n_groups_by_user_number represents a percentage number.
        top_n_groups_by_median_seq_len : int, optional
            The number of top_n groups by median sequence length to be selected., by default None
        is_pct_top_n_groups_by_median_seq_len : bool
            A flag indicating whether top_n_groups_by_median_seq_len represents a percentage number.
        sample_pct : int, optional
            The percentage of groups to be sampled. Must be between 1 and 100., by default None
        sample_pct_user : int, optional
            The percentage of users to be sampled. This takes place after all group filter. Must be between 1 and 100.,\
            by default None
        **kwargs :
            Keyword arguments for the respective distance function.

        Returns
        -------
        None
        """
        # aLgorithm start
        start_time = time.time()

        group_array = self.group_array.copy()
        print(DASH_STRING)
        print(f'Total number of {GROUP_FIELD_NAME_STR}: {group_array.size}')
        print(DASH_STRING)

        # methods to reduce the number of groups used in the sequence distance calculations
        if min_number_of_users_per_group:
            group_array = self._get_col1_by_min_number_col2(group_array,
                                                            min_number_of_users_per_group,
                                                            self.GROUP_COL_INDEX,
                                                            self.USER_COL_INDEX,
                                                            f'{GROUP_FIELD_NAME_STR}s',
                                                            f'{USER_FIELD_NAME_STR}s')

        if min_number_avg_seq_len:
            group_array = self._get_col1_by_min_number_median_seq_len(group_array,
                                                                      min_number_avg_seq_len,
                                                                      self.GROUP_COL_INDEX,
                                                                      self.USER_COL_INDEX,
                                                                      f'{GROUP_FIELD_NAME_STR}s',
                                                                      f'{USER_FIELD_NAME_STR}s')

        if top_n_groups_by_user_number:
            group_array = self._get_top_n_col1_by_col2(group_array,
                                                       top_n_groups_by_user_number,
                                                       is_pct_top_n_groups_by_user_number,
                                                       self.GROUP_COL_INDEX,
                                                       self.USER_COL_INDEX,
                                                       f'{GROUP_FIELD_NAME_STR}s',
                                                       f'{USER_FIELD_NAME_STR}s')

        if top_n_groups_by_median_seq_len:
            group_array = self._get_top_n_col1_by_median_seq_len(group_array,
                                                                 top_n_groups_by_median_seq_len,
                                                                 is_pct_top_n_groups_by_median_seq_len,
                                                                 self.GROUP_COL_INDEX,
                                                                 self.USER_COL_INDEX,
                                                                 f'{GROUP_FIELD_NAME_STR}s',
                                                                 f'{USER_FIELD_NAME_STR}s')

        if sample_pct:
            group_array = self._sample_col1(group_array, sample_pct, f'{GROUP_FIELD_NAME_STR}s')

        # filter original dataframe by sampled groups
        data = self.data[np.isin(self.data[:, self.GROUP_COL_INDEX], group_array), :].copy()
        user_after_group_filter = np.unique(data[:, self.USER_COL_INDEX])

        # filter original dataframe by sampled users
        if sample_pct_user:
            user_array = self._sample_col1(user_after_group_filter, sample_pct_user, f'{USER_FIELD_NAME_STR}s')
            data = data[np.isin(data[:, self.USER_COL_INDEX], user_array), :]

        groups_left = len(np.unique(data[:, self.GROUP_COL_INDEX]))
        users_left = len(np.unique(data[:, self.USER_COL_INDEX]))
        interactions_left = data.shape[0]

        print(DASH_STRING)
        print(f'Final number of {GROUP_FIELD_NAME_STR}s: {groups_left}')
        print(f'Final number of {USER_FIELD_NAME_STR}s: {users_left}')
        print(f'Final number of {ROWS_NAME_STR}: {interactions_left}')
        print(DASH_STRING)

        # delete old pickle files to prevent keeping results of groups which are not part of current calculation anymore
        delete_all_pickle_files_within_directory(self.result_directory)

        # calculate sequence distances for each group and write result to disk
        if parallelize_computation:
            results = (Parallel(n_jobs=NUMBER_OF_CORES)
                               (delayed(self._calculate_sequence_distances)
                               (group,
                                self._filter_data_by_group(data, group),
                                distance_function,
                                *args,
                                **kwargs) for group in tqdm(group_array)))
        else:
            results = [self._calculate_sequence_distances(group,
                                                          self._filter_data_by_group(data, group),
                                                          distance_function,
                                                          *args,
                                                          **kwargs) for group in tqdm(group_array)]

        # algorithm end
        end_time = time.time()
        duration = end_time - start_time
        print(DASH_STRING)
        print(f'Duration in seconds: {duration}')
        print(DASH_STRING)

        return None

    def calc_user_sequence_distances(self,
                                     parallelize_computation: bool,
                                     distance_function,
                                     *args,
                                     min_number_of_groups_per_user=None,
                                     min_number_avg_seq_len=None,
                                     top_n_users_by_group_number=None,
                                     is_pct_top_n_users_by_group_number=False,
                                     top_n_users_by_median_seq_len=None,
                                     is_pct_top_n_users_by_median_seq_len=False,
                                     sample_pct=None,
                                     **kwargs) -> None:
        """For group '0' calculates the (learning activity-) sequence distances between each possible user\
           combination(seq_distances) and sequence combination(unique_seq_distances) pair. The result,\
           a dictionary containing sequence distances between user combinations and sequence combinations,\
           will be saved as pickle file. The sequence distance results will be treated as if they belong to\
           a single group(group '0') ranging over the entire length of the interactions dataframe.

        Parameters
        ----------
         parallelize_computation : 
            A flag indicating whether the sequence distance calculations should be done in parallel
        distance_function : 
            A sequence distance function taking 2 strings as the first two positional arguments as input.
        *args :
            Positional arguments for the respective distance function.
        min_number_of_groups_per_user : int, optional
            The minimum number of groups a user is allowed to have., by default None
        min_number_avg_seq_len : int, optional
            The minimum median sequence length a user is allowed to have., by default None
        top_n_users_by_group_number : int, optional
            The number of top_n users by groups to be selected., by default None
        is_pct_top_n_users_by_group_number : bool
            A flag indicating whether top_n_users_by_group_number represents a percentage number.
        top_n_users_by_median_seq_len : int, optional
            The number of top_n users by median sequence length to be selected., by default None
        is_pct_top_n_users_by_median_seq_len : bool
            A flag indicating whether top_n_users_by_median_seq_len represents a percentage number.
        sample_pct : int, optional
            The percentage of users to be sampled. Must be between 1 and 100., by default None
        **kwargs :
            Keyword arguments for the respective distance function.

        Returns
        -------
        None
        """
        # aLgorithm start
        start_time = time.time()

        user_array = self.user_array.copy()
        print(DASH_STRING)
        print(f'Total number of {USER_FIELD_NAME_STR}s: {user_array.size}')
        print(DASH_STRING)

        # methods to reduce the number of users used in the sequence distance calculations
        if min_number_of_groups_per_user:
            user_array = self._get_col1_by_min_number_col2(user_array,
                                                           min_number_of_groups_per_user,
                                                           self.USER_COL_INDEX,
                                                           self.GROUP_COL_INDEX,
                                                           f'{USER_FIELD_NAME_STR}s',
                                                           f'{GROUP_FIELD_NAME_STR}s')

        if min_number_avg_seq_len:
            user_array = self._get_col1_by_min_number_median_seq_len(user_array,
                                                                     min_number_avg_seq_len,
                                                                     self.USER_COL_INDEX,
                                                                     self.GROUP_COL_INDEX,
                                                                     f'{USER_FIELD_NAME_STR}s',
                                                                     f'{GROUP_FIELD_NAME_STR}s')

        if top_n_users_by_group_number:
            user_array = self._get_top_n_col1_by_col2(user_array,
                                                      top_n_users_by_group_number,
                                                      is_pct_top_n_users_by_group_number,
                                                      self.USER_COL_INDEX,
                                                      self.GROUP_COL_INDEX,
                                                      f'{USER_FIELD_NAME_STR}s',
                                                      f'{GROUP_FIELD_NAME_STR}s')

        if top_n_users_by_median_seq_len:
            user_array = self._get_top_n_col1_by_median_seq_len(user_array,
                                                                top_n_users_by_median_seq_len,
                                                                is_pct_top_n_users_by_median_seq_len,
                                                                self.USER_COL_INDEX,
                                                                self.GROUP_COL_INDEX,
                                                                f'{USER_FIELD_NAME_STR}s',
                                                                f'{GROUP_FIELD_NAME_STR}s')

        # filter original dataframe by sampled users
        if sample_pct:
            user_array = self._sample_col1(user_array, sample_pct, 'users')

        data = self.data[np.isin(self.data[:, self.USER_COL_INDEX], user_array), :].copy()

        groups_left = len(np.unique(data[:, self.GROUP_COL_INDEX]))
        users_left = len(np.unique(data[:, self.USER_COL_INDEX]))
        interactions_left = data.shape[0]

        print(DASH_STRING)
        print(f'Final number of {GROUP_FIELD_NAME_STR}s: {groups_left}')
        print(f'Final number of {USER_FIELD_NAME_STR}s: {users_left}')
        print(f'Final number of interactions: {interactions_left}')
        print(DASH_STRING)

        # create dummy group
        data[:, self.GROUP_COL_INDEX] = '0'
        group_array = np.array(['0'])

        # delete old pickle files to prevent keeping results of groups which are not part of current calculation anymore
        delete_all_pickle_files_within_directory(self.result_directory)

        # calculate sequence distances for each group and write result to disk
        if parallelize_computation:
            results = (Parallel(n_jobs=NUMBER_OF_CORES)
                               (delayed(self._calculate_sequence_distances)
                               (group,
                                data,
                                distance_function, 
                                *args, 
                                **kwargs) for group in tqdm(group_array)))
        else:
            results = [self._calculate_sequence_distances(group,
                                                          data, 
                                                          distance_function, 
                                                          *args, 
                                                          **kwargs) for group in tqdm(group_array)]

        # algorithm end
        end_time = time.time()
        duration = end_time - start_time
        print(DASH_STRING)
        print(f'Duration in seconds: {duration}')
        print(DASH_STRING)

        return None


def calculate_sequence_distances(dataset_name: str, 
                                 interactions: pd.DataFrame, 
                                 group_field: str, 
                                 ignore_groups: bool) -> None:
    """For each group calculates the (learning activity-) sequence distances between each possible user\
    combination(seq_distances) and sequence combination(unique_seq_distances) pair.\
    If an interactions dataframe does not contain a grouping field, the sequence distance results will be treated as if\
    they belong to a single group(group '0') ranging over the entire length of a users learning activities in the\
    interactions dataframe.


    Parameters
    ----------
    dataset_name: str,
        The name of the dataset
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field
    ignore_groups: bool
        A boolean indicating whether the group field should be ignored (even when the datasets has groups)\
        and sequence distances be calculated over the entire length of a user's learning activities in the\
        interactions dataframe.

    Returns
    -------
    None
    """
    if group_field:
        if ignore_groups:
            print(DASH_STRING)
            print(f'{GROUP_FIELD_NAME_STR}-Field Available But Will Be Ignored:')
            print(f'Calculate {SEQUENCE_STR} Distances')
            print(DASH_STRING)
        else:
            print(DASH_STRING)
            print(f'{GROUP_FIELD_NAME_STR}-Field Available:')
            print(f'Calculate {SEQUENCE_STR} Distances for each {GROUP_FIELD_NAME_STR}')
            print(DASH_STRING)
    else:
        print(DASH_STRING)
        print(f'{GROUP_FIELD_NAME_STR}-Field NOT Available:')
        print(f'Calculate {SEQUENCE_STR} Distances')
        print(DASH_STRING)
    seq_sim = SeqDist(dataset_name,
                      interactions,
                      USER_FIELD_NAME_STR,
                      GROUP_FIELD_NAME_STR,
                      LEARNING_ACTIVITY_FIELD_NAME_STR,
                      SEQUENCE_ID_FIELD_NAME_STR,
                      group_field)
    if ignore_groups:
        _ = seq_sim.calc_user_sequence_distances(PARALLELIZE_COMPUTATIONS,
                                                 distance,
                                                 *SEQUENCE_DISTANCE_FUNCTION_ARGS,
                                                 **SEQUENCE_DISTANCE_FUNCTION_KWARGS)
    else:
        _ = seq_sim.calc_user_sequence_distances_per_group(PARALLELIZE_COMPUTATIONS,
                                                           distance,
                                                           *SEQUENCE_DISTANCE_FUNCTION_ARGS,
                                                           **SEQUENCE_DISTANCE_FUNCTION_KWARGS)
