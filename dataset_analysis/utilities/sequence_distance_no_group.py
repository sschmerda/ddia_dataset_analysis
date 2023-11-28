from .standard_import import *
from .constants import *

class SeqDistNoGroup:
    """
    A class to calculate distances of learning activity sequences.

    Parameters
    ----------
    data : DataFrame
        A dataframe containing group, user and learning activity fields.
    user_field: str
        Then name of the user field.
    learning_activity_field: str
        Then name of the learning activity field.
    sequence_id_field: str
        Then name of the sequence id field.

    Methods
    -------
    get_user_sequence_distances(distance function,\
                                sample_pct=None):
        For group '0' calculates the (learning activity-) sequence distances between each possible user\
        combination(seq_distances) and sequence combination(unique_seq_distances) pair.
    """
    USER_COL_INDEX = 0 
    LEARNING_ACTIVITY_COL_INDEX = 1
    SEQUENCE_ID_COL_INDEX = 2
    LEARNING_ACTIVITY_CHAR_COL_INDEX = 3

    def __init__(self,
                 data: pd.DataFrame,
                 user_field: str,
                 learning_activity_field: str,
                 sequence_id_field: str):

        self.data = data
        self.user_field = user_field
        self.learnin_activity_field = learning_activity_field
        self.sequence_id_field = sequence_id_field
        
        #Initial data transformation
        self._select_fields_and_transform_to_numpy()
        self._transform_to_char()

        self.user_array = np.unique(self.data[:, SeqDistNoGroup.USER_COL_INDEX])


    def _select_fields_and_transform_to_numpy(self):

        self.data = self.data[[self.user_field, self.learnin_activity_field, self.sequence_id_field, self.learnin_activity_field]].astype('string').values

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
        unique_lr = np.unique(self.data[:, SeqDistNoGroup.LEARNING_ACTIVITY_COL_INDEX])
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

        char_array = [lr_mapping[i] for i in self.data[:, SeqDistNoGroup.LEARNING_ACTIVITY_COL_INDEX]]

        self.data[:, SeqDistNoGroup.LEARNING_ACTIVITY_CHAR_COL_INDEX] = char_array
        
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

    def get_user_sequence_distances(self,
                                    distance_function,
                                    sample_pct=None):
        """For group '0' calculates the (learning activity-) sequence distances between each possible user\
           combination(seq_distances) and sequence combination(unique_seq_distances) pair.
           The sequence distance results will be treated as if they belong to a single group(group '0') ranging over the entire length of the interactions dataframe.

        Parameters
        ----------
        distance_function : 
            A sequence distance function from the textdistance library.
        sample_pct : int, optional
            The percentage of users to be sampled. Must be between 1 and 100., by default None

        Returns
        -------
        tuple
            A tuple consisting of:\

            A dictionary containing for group '0' a ndarray of sequence distances between user combinations, 
            a ndarray of lengths of the longer of two compared sequences, a ndarray of users id combinations used\
            for sequence distance calculation, a ndarray of sequence id combinations, a ndarray of user ids per group,\
            a ndarray of sequence lengths for every user per group, a ndarray of sequence ids and a ndarray of tuples\
            containing the sequence of learning activities the sequence ids map to.\

            A dictionary containing for group '0' a ndarray of sequence distances between sequence combinations,\
            a ndarray of lengths of the longer of two compared sequence and a ndarray of sequence id combinations.
        """        
        # aLgorithm start
        start_time = time.time()

        user_array = self.user_array
        print(50*'-')
        print(f'Total number of users: {user_array.size}')
        print(50*'-')

        # filter original dataframe by sampled users
        if sample_pct:
            user_array = self._sample_col1(user_array, 
                                           sample_pct, 
                                           {USER_FIELD_NAME_STR})

        data = self.data[np.isin(self.data[:, self.USER_COL_INDEX], user_array), :]
        user_sequence_id_array = [data[data[:, self.USER_COL_INDEX]==i, self.SEQUENCE_ID_COL_INDEX][0] for i in user_array]
        user_sequence_id_mapping_dict = dict(zip(user_array, user_sequence_id_array))

        users_left = len(np.unique(data[:, self.USER_COL_INDEX]))
        interactions_left = data.shape[0]

        print(50*'-')
        print(f'Final number of {USER_FIELD_NAME_STR}s: {users_left}')
        print(f'Final number of interactions: {interactions_left}')
        print(50*'-')

        users = data[:, self.USER_COL_INDEX]
        learning_activities_char = data[:, self.LEARNING_ACTIVITY_CHAR_COL_INDEX]
        learning_activities = data[:, self.LEARNING_ACTIVITY_COL_INDEX]

        # generate a user - sequence mapping
        user_char_string_mapping_dict = {}
        user_sequence_length = []
        user_sequence_array = []
        for user in user_array:
            user_learning_activities_char = learning_activities_char[users==user]
            user_learning_activities = tuple(learning_activities[users==user])
            char_string = self._generate_char_string(user_learning_activities_char)
            user_char_string_mapping_dict[user] = char_string 
            user_sequence_length.append(len(char_string))
            user_sequence_array.append(user_learning_activities)

        # generate a sequence_combination - sequence_distance/max_sequence_length mapping 
        sequence_id_char_string_mapping_dict = {seq_id: user_char_string_mapping_dict[user] for user, seq_id in user_sequence_id_mapping_dict.items()}
        sequence_id_array = list(sequence_id_char_string_mapping_dict.keys())

        sequence_combinations_with_replacement = [tuple(sorted(i, key=int)) for i in list(combinations_with_replacement(sequence_id_array, 2))]
        sequence_combinations = [tuple(sorted(i, key=int)) for i in list(combinations(sequence_id_array, 2))]

        sequence_distance_dict={}
        for sequence_combination in sequence_combinations_with_replacement:

            char_sequence_1 = sequence_id_char_string_mapping_dict[sequence_combination[0]]
            char_sequence_2 = sequence_id_char_string_mapping_dict[sequence_combination[1]]
            sequence_distance = distance_function(char_sequence_1, char_sequence_2)
            max_sequence_length = max(len(char_sequence_1), len(char_sequence_2))

            sequence_distance_dict[sequence_combination] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: sequence_distance,
                                                            LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: max_sequence_length}
        # generate the sequence_distances dictionary   
        user_combinations = [tuple(sorted(i, key=int)) for i in combinations(user_array, 2)] 
        sequence_distance_list = []
        max_sequence_length_list = []
        user_combination_list = []
        sequence_id_combination_list = []
        for user_combination in user_combinations:

            sequence_1 = user_sequence_id_mapping_dict[user_combination[0]]
            sequence_2 = user_sequence_id_mapping_dict[user_combination[1]]

            sequence_tuple = tuple(sorted((sequence_1, sequence_2), key=int))

            sequence_distance = sequence_distance_dict[sequence_tuple][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
            max_sequence_length = sequence_distance_dict[sequence_tuple][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
            sequence_id_combination = sorted((sequence_1, sequence_2), key=int)
                
            sequence_distance_list.append(sequence_distance)
            max_sequence_length_list.append(max_sequence_length)
            user_combination_list.append(user_combination)
            sequence_id_combination_list.append(sequence_id_combination)

        # use a dummy group name for non-group df calculation. 
        # this allows consistent usage of the cluster_evaluation module,
        # which takes a sequence_distances dictionary with keys representing groups as input
        sequence_distances = {}
        sequence_distances[DUMMY_GROUP_NAME_FOR_NO_GROUP_DF] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: np.array(sequence_distance_list),
                                                                LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: np.array(max_sequence_length_list),
                                                                LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_NAME_STR: user_combination_list,
                                                                LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_NAME_STR: sequence_id_combination_list,
                                                                LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR: user_array,
                                                                LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: np.array(user_sequence_length),
                                                                LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: np.array(user_sequence_id_array),
                                                                LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR: np.array(user_sequence_array)}

        # generate the unique_sequence_distances dictionary   
        unique_sequence_distances = {}

        unique_sequence_distance_list = []
        max_unique_sequence_length_list = []
        unique_sequence_id_combination_list = []

        for sequence_combination in sequence_combinations:

            unique_sequence_distance = sequence_distance_dict[sequence_combination][LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR]
            max_unique_sequence_length = sequence_distance_dict[sequence_combination][LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR]
            unique_sequence_combination = sequence_combination

            unique_sequence_distance_list.append(unique_sequence_distance)
            max_unique_sequence_length_list.append(max_unique_sequence_length)
            unique_sequence_id_combination_list.append(unique_sequence_combination)
            
        unique_sequence_distances[DUMMY_GROUP_NAME_FOR_NO_GROUP_DF] = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: np.array(unique_sequence_distance_list),
                                                                       LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR: np.array(max_unique_sequence_length_list),
                                                                       LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_NAME_STR: unique_sequence_id_combination_list,
                                                                       LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: np.array(sequence_id_array)}

        sequence_distances_dict = {LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR: sequence_distances,
                                   LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR: unique_sequence_distances}

        # aLgorithm end
        end_time = time.time()
        duration = end_time - start_time
        print(50*'-')
        print(f'Duration in seconds: {duration}')
        print(50*'-')

        return sequence_distances_dict