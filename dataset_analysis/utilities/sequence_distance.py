from .standard_import import *
from .constants import *

class SeqDist:
    """
    A class to calculate distances of learning activity sequences for differend groups.

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

    Methods
    -------
    get_user_sequence_distances_per_group(distance function,\
                                          min_number_of_users_per_group=None,\
                                          min_number_avg_seq_len=None,\
                                          top_n_groups_by_user_number=None,\
                                          is_pct_top_n_groups_by_user_number,\
                                          top_n_groups_by_median_seq_len=None,\
                                          is_pct_top_n_groups_by_median_seq_len,\
                                          sample_pct=None):
    For each group calculates the (learning activity-) sequence distances between each possible user combination pair.
    """
    GROUP_COL_INDEX = 0
    USER_COL_INDEX = 1
    LEARNING_ACTIVITY_COL_INDEX = 2
    SEQUENCE_ID_COL_INDEX = 3
    LEARNING_ACTIVITY_CHAR_COL_INDEX = 4

    def __init__(self,
                 data: pd.DataFrame,
                 user_field: str,
                 group_field: str,
                 learning_activity_field: str,
                 sequence_id_field: str):

        self.data = data
        self.user_field = user_field
        self.group_field = group_field
        self.learnin_activity_field = learning_activity_field
        self.sequence_id_field = sequence_id_field
        
        #Initial data transformation
        self._select_fields_and_transform_to_numpy()
        self._transform_to_char()

        self.group_array = np.unique(self.data[:, SeqDist.GROUP_COL_INDEX])
        self.user_array = np.unique(self.data[:, SeqDist.USER_COL_INDEX])

    def _select_fields_and_transform_to_numpy(self):

        self.data = self.data[[self.group_field, self.user_field, self.learnin_activity_field, self.sequence_id_field, self.learnin_activity_field]].astype('string').values

    def _transform_to_char(self):
        """Maps unique characters to learning activities and returns the transformed series (tokenization)

        Returns
        ------
        None

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
        
        # pandas cannot distinguish between some unicode charactes -> reduce chars to the ones
        # readable by pandas
        chars = pd.Series(chars).unique()

        if len(unique_lr) > len(chars):
            raise Exception(f'There are more unique {LEARNING_ACTIVITY_FIELD_NAME_STR}s than possible characters')

        for n,i in enumerate(unique_lr):
            lr_mapping[i] = chars[n] 

        char_array = [lr_mapping[i] for i in self.data[:, SeqDist.LEARNING_ACTIVITY_COL_INDEX]]

        self.data[:, SeqDist.LEARNING_ACTIVITY_CHAR_COL_INDEX] = char_array
        
    @staticmethod
    def _generate_char_string(learning_activities: np.ndarray):
        """Concatenates the elements of an array into a single string

        Parameters
        ----------
        learning_activity : ndarray
            An numpy array 

        Returns
        -------
        str
            the concatenated string
        """
        char_string = "".join(learning_activities)
        
        return char_string


    def _get_col1_by_min_number_col2(self,
                                     current_col1: np.ndarray,
                                     min_n_col2: int,
                                     col_1_index: int,
                                     col_2_index: int,
                                     col_1_name: str,
                                     col_2_name: str):
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

        print(50*'-')
        print(f'Filtering out {col_1_name} with less than {min_n_col2} {col_2_name}:')
        print(f'Total number of {col_1_name} left: {n_kept_col1s}')
        print(50*'-')

        return col1_array

    def _get_col1_by_min_number_median_seq_len(self,
                                               current_col1: np.ndarray,
                                               min_median_seq_len: int,
                                               col_1_index: int,
                                               col_2_index: int,
                                               col_1_name: str,
                                               col_2_name: str):
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

        print(50*'-')
        print(f'Filtering out {col_1_name} with a median sequence length of {col_2_name} of less than {min_median_seq_len}:')
        print(f'Total number of {col_1_name} left: {n_kept_col1s}')
        print(50*'-')
        
        return col1_array

    def _get_top_n_col1_by_col2(self,
                                current_col1: np.ndarray,
                                top_n: int,
                                is_pct: bool,
                                col_1_index: int,
                                col_2_index: int,
                                col_1_name: str,
                                col_2_name: str):
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
            print(50*'-')
            print(f'Keeping top {pct_kept}% of {col_1_name} by {col_2_name} size:')
            print(f'Total number of {col_1_name} left: {n_kept_col1}')
            print(50*'-')
        else:
            print(50*'-')
            print(f'Keeping top {top_n} {col_1_name} by {col_2_name} size:')
            print(f'Total number of groups left: {n_kept_col1}')
            print(50*'-')

        return col1_array

    def _get_top_n_col1_by_median_seq_len(self,
                                          current_col1: np.ndarray,
                                          top_n: int,
                                          is_pct: bool,
                                          col_1_index: int,
                                          col_2_index: int,
                                          col_1_name: str,
                                          col_2_name: str):
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
            print(50*'-')
            print(f'Keeping top {pct_kept}% of {col_1_name} by median sequence length of {col_2_name}:')
            print(f'Total number of {col_1_name} left: {n_kept_col1}')
            print(50*'-')
        else:
            print(50*'-')
            print(f'Keeping top {top_n} {col_1_name} by median sequence length of {col_2_name}:')
            print(f'Total number of groups left: {n_kept_col1}')
            print(50*'-')

        return col1_array
        

    @staticmethod
    def _sample_col1(current_col1: np.ndarray,
                        sample_pct: int,
                        col_1_name: str):
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

        print(50*'-')
        print(f'Sampling {sample_pct} percent of {col_1_name}:')
        print(f'Total number of {col_1_name} left: {n_kept_col1}')
        print(50*'-')

        return col1_array

    def get_user_sequence_distances_per_group(self,
                                              distance_function,
                                              min_number_of_users_per_group=None,
                                              min_number_avg_seq_len=None,
                                              top_n_groups_by_user_number=None,
                                              is_pct_top_n_groups_by_user_number=False,
                                              top_n_groups_by_median_seq_len=None,
                                              is_pct_top_n_groups_by_median_seq_len=False,
                                              sample_pct=None,
                                              sample_pct_user=None):
        """For each group calculates the (learning activity-) sequence distances between each possible user combination pair.

        Parameters
        ----------
        distance_function : 
            A sequence distance function from the textdistance library.
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

        Returns
        -------
        dict 
            A dictionary containing for every group a ndarray of sequence distances, a ndarray of lengths of the longer\
            of two compared sequences, a ndarray of users id combinations used for sequence distance calculation,\
            a ndarray of user ids per group, a ndarray of sequence lengths for every user per group, a ndarray of sequence\
            ids and a ndarray of tuples containing the sequence of learning activities the sequence ids map to.
        """        
        # aLgorithm start
        start_time = time.time()

        group_array = self.group_array
        print(50*'-')
        print(f'Total number of {GROUP_FIELD_NAME_STR}: {group_array.size}')
        print(50*'-')

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
            group_array = self._sample_col1(group_array, 
                                            sample_pct, 
                                            f'{GROUP_FIELD_NAME_STR}s')
        
        # filter original dataframe by sampled groups
        data = self.data[np.isin(self.data[:, self.GROUP_COL_INDEX], group_array), :]
        user_after_group_filter = np.unique(data[:, self.USER_COL_INDEX])

        # filter original dataframe by sampled users
        if sample_pct_user:
            user_array = self._sample_col1(user_after_group_filter, 
                                           sample_pct_user, 
                                           f'{USER_FIELD_NAME_STR}s')
            data = data[np.isin(data[:, self.USER_COL_INDEX], user_array), :]

        groups_left = len(np.unique(data[:, self.GROUP_COL_INDEX]))
        users_left = len(np.unique(data[:, self.USER_COL_INDEX]))
        interactions_left = data.shape[0]

        print(50*'-')
        print(f'Final number of {GROUP_FIELD_NAME_STR}s: {groups_left}')
        print(f'Final number of {USER_FIELD_NAME_STR}s: {users_left}')
        print(f'Final number of interactions: {interactions_left}')
        print(50*'-')

        sequence_distances_per_group = {}

        for group in tqdm(group_array):
            
            group_data = data[data[:, self.GROUP_COL_INDEX]==group, :]
            user_array = np.unique(group_data[:, self.USER_COL_INDEX])
            user_sequence_id_array = [group_data[group_data[:, self.USER_COL_INDEX]==i, self.SEQUENCE_ID_COL_INDEX][0] for i in user_array]
            
            users = group_data[:, self.USER_COL_INDEX]
            learning_activities_char = group_data[:, self.LEARNING_ACTIVITY_CHAR_COL_INDEX]
            learning_activities = group_data[:, self.LEARNING_ACTIVITY_COL_INDEX]


            # generate a user - sequence mapping
            user_sequence = {}
            user_sequence_length = []
            user_sequence_array = []
            for user in user_array:
                user_learning_activities_char = learning_activities_char[users==user]
                user_learning_activities = tuple(learning_activities[users==user])
                char_string = self._generate_char_string(user_learning_activities_char)
                user_sequence[user] = char_string 
                user_sequence_length.append(len(char_string))
                user_sequence_array.append(user_learning_activities)

            # generate a sequence_combination - sequence_distance/max_sequence_length mapping 
            sequence_array = list(set(user_sequence.values()))
            sequence_combinations = combinations_with_replacement(sequence_array, 2)

            sequence_distance_dict={}
            for sequence_combination in sequence_combinations:
                sequence_distance = distance_function(sequence_combination[0], sequence_combination[1])
                max_sequence_length = max(len(sequence_combination[0]), len(sequence_combination[1]))

                sequence_distance_dict[frozenset(sequence_combination)] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: sequence_distance,
                                                                           LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_sequence_length}
            # generate the sequnce_distances_per_group dictionary   
            user_combinations = combinations(user_array, 2) 
            sequence_distance_list = []
            max_sequence_length_list = []
            user_combination_list = []
            for user_combination in user_combinations:

                sequence_1 = user_sequence[user_combination[0]]
                sequence_2 = user_sequence[user_combination[1]]

                sequence_set = frozenset([sequence_1, sequence_2])

                sequence_distance = sequence_distance_dict[sequence_set][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
                max_sequence_length = sequence_distance_dict[sequence_set][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
                
                sequence_distance_list.append(sequence_distance)
                max_sequence_length_list.append(max_sequence_length)
                user_combination_list.append(user_combination)

            sequence_distances_per_group[group] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: np.array(sequence_distance_list),
                                                   LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: np.array(max_sequence_length_list),
                                                   LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_NAME_STR: np.array(user_combination_list),
                                                   LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR: user_array,
                                                   LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: np.array(user_sequence_length),
                                                   LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: np.array(user_sequence_id_array),
                                                   LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR: np.array(user_sequence_array)}

        # algorithm end
        end_time = time.time()
        duration = end_time - start_time
        print(50*'-')
        print(f'Duration in seconds: {duration}')
        print(50*'-')

        return sequence_distances_per_group

    def get_user_sequence_distances(self,
                                    distance_function,
                                    min_number_of_groups_per_user=None,
                                    min_number_avg_seq_len=None,
                                    top_n_users_by_group_number=None,
                                    is_pct_top_n_users_by_group_number=False,
                                    top_n_users_by_median_seq_len=None,
                                    is_pct_top_n_users_by_median_seq_len=False,
                                    sample_pct=None):
        """For each user calculates the (learning activity-) sequence distances between each possible user combination pair.
        The sequence distance results will be treated as if they belong to a single group(group '0') ranging over the entire length of the interactions dataframe.

        Parameters
        ----------
        distance_function : 
            A sequence distance function from the textdistance library.
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

        Returns
        -------
        dict 
            A dictionary containing for group '0' a ndarray of sequence distances between user combinations, a ndarray of lengths\
            of the longer of two compared combination sequences, a ndarray of user id combinations used for sequence\
            distance calculation, a ndarray of user ids, an ndarray of sequence lengths for every user, a ndarray of sequence\
            ids and a ndarray of tuples containing the sequence of learning activities the sequence ids map to.
        """        
        # aLgorithm start
        start_time = time.time()

        user_array = self.user_array
        print(50*'-')
        print(f'Total number of {USER_FIELD_NAME_STR}s: {user_array.size}')
        print(50*'-')

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

        data = self.data[np.isin(self.data[:, self.USER_COL_INDEX], user_array), :]
        user_sequence_id_array = [data[data[:, self.USER_COL_INDEX]==i, self.SEQUENCE_ID_COL_INDEX][0] for i in user_array]

        groups_left = len(np.unique(data[:, self.GROUP_COL_INDEX]))
        users_left = len(np.unique(data[:, self.USER_COL_INDEX]))
        interactions_left = data.shape[0]

        print(50*'-')
        print(f'Final number of {GROUP_FIELD_NAME_STR}s: {groups_left}')
        print(f'Final number of {USER_FIELD_NAME_STR}s: {users_left}')
        print(f'Final number of interactions: {interactions_left}')
        print(50*'-')

        users = data[:, self.USER_COL_INDEX]
        learning_activities_char = data[:, self.LEARNING_ACTIVITY_CHAR_COL_INDEX]
        learning_activities = data[:, self.LEARNING_ACTIVITY_COL_INDEX]

        # generate a user - sequence mapping
        user_sequence = {}
        user_sequence_length = []
        user_sequence_array = []
        for user in user_array:
            user_learning_activities_char = learning_activities_char[users==user]
            user_learning_activities = tuple(learning_activities[users==user])
            char_string = self._generate_char_string(user_learning_activities_char)
            user_sequence[user] = char_string 
            user_sequence_length.append(len(char_string))
            user_sequence_array.append(user_learning_activities)

        # generate a sequence_combination - sequence_distance/max_sequence_length mapping 
        sequence_array = list(set(user_sequence.values()))
        sequence_combinations = combinations_with_replacement(sequence_array, 2)

        sequence_distance_dict={}
        for sequence_combination in tqdm(sequence_combinations):
            sequence_distance = distance_function(sequence_combination[0], sequence_combination[1])
            max_sequence_length = max(len(sequence_combination[0]), len(sequence_combination[1]))

            sequence_distance_dict[frozenset(sequence_combination)] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: sequence_distance,
                                                                       LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_sequence_length}
        # generate the sequence_distances dictionary   
        user_combinations = combinations(user_array, 2) 
        sequence_distance_list = []
        max_sequence_length_list = []
        user_combination_list = []
        for user_combination in user_combinations:

            sequence_1 = user_sequence[user_combination[0]]
            sequence_2 = user_sequence[user_combination[1]]

            sequence_set = frozenset([sequence_1, sequence_2])

            sequence_distance = sequence_distance_dict[sequence_set][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
            max_sequence_length = sequence_distance_dict[sequence_set][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
                
            sequence_distance_list.append(sequence_distance)
            max_sequence_length_list.append(max_sequence_length)
            user_combination_list.append(user_combination)

        # use a dummy group name for non-group df calculation. 
        # this allows consistent usage of the cluster_evaluation module,
        # which takes a sequence_distances dictionary with keys representig groups as input
        sequence_distances = {}
        sequence_distances[DUMMY_GROUP_NAME_FOR_NO_GROUP_DF] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: np.array(sequence_distance_list),
                                                                LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: np.array(max_sequence_length_list),
                                                                LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_NAME_STR: np.array(user_combination_list),
                                                                LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR: user_array,
                                                                LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: np.array(user_sequence_length),
                                                                LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: np.array(user_sequence_id_array),
                                                                LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR: np.array(user_sequence_array)}

        # aLgorithm end
        end_time = time.time()
        duration = end_time - start_time
        print(50*'-')
        print(f'Duration in seconds: {duration}')
        print(50*'-')

        return sequence_distances