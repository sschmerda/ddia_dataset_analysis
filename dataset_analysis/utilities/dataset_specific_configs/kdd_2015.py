from ..constants.constants import *
from ..standard_import import *
from ..preprocessing_functions import *
from ..io_functions import *
from ..result_tables import *

########################################################################################################################
### dataset name ###
########################################################################################################################

DATASET_NAME = 'kdd_2015'

########################################################################################################################
### paths ###
########################################################################################################################

PATH_TO_INTERACTION_DATA = '../../../../../../data/ddia/kdd_2015/complete/log_train.csv'
PATH_TO_ENROLLMENT_DATA = '../../../../../../data/ddia/kdd_2015/complete/enrollment_train.csv'
PATH_TO_DROPOUT_DATA = '../../../../../../data/ddia/kdd_2015/complete/truth_train.csv'
PATH_TO_OBJECT_DATA = '../../../../../../data/ddia/kdd_2015/ObjectData/object.csv'

########################################################################################################################
### required fields ###
########################################################################################################################

USER_FIELD = 'username'
GROUP_FIELD = 'course_id'
LEARNING_ACTIVITY_FIELD = 'object'
COURSE_FIELD = 'Course'
TIMESTAMP_FIELD = 'time'
ORDER_FIELD = None

########################################################################################################################
### dataset specific constants ###
########################################################################################################################

# fields used in dataset preparation
ENROLLMENT_FIELD = 'enrollment_id'
DROPOUT_FIELD = 'is_dropout'
MODULE_FIELD = 'module_id'
EVENT_FIELD = 'event'
CATEGORY_FIELD = 'category'

# strings and values used in dataset preparation
VIDEO_VALUE_STR = 'video'
PROBLEM_VALUE_STR = 'problem'

########################################################################################################################
### evaluation fields ###
########################################################################################################################

# learning activity level
EVALUATION_LEARNING_ACTIVITY_SCORE_FIELD = None
EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_FIELD = None
# group level
EVALUATION_GROUP_SCORE_FIELD = None
EVALUATION_GROUP_IS_CORRECT_FIELD = 'no_dropout' 
# course level
EVALUATION_COURSE_SCORE_FIELD = None
EVALUATION_COURSE_IS_CORRECT_FIELD = None 

########################################################################################################################
### data import and preprocessing config ###
########################################################################################################################

INTERACTION_TYPE_VALUES_TO_KEEP_LIST = [PROBLEM_VALUE_STR,
                                        VIDEO_VALUE_STR]

DROP_LEARNING_ACTIVITY_SEQUENCE_IF_NA_FIELD_LIST = [TIMESTAMP_FIELD,
                                                    LEARNING_ACTIVITY_FIELD,
                                                    DROPOUT_FIELD]
FIELD_VALUE_TUPLE_NA_FILTER_EXCEPTION_LIST = []

DROP_ROW_IF_NA_FIELD_LIST = [GROUP_FIELD,
                             USER_FIELD]

########################################################################################################################
### evaluation fields correct field calculation ###
########################################################################################################################

# score threshold is correct
EVALUATION_LEARNING_ACTIVITY_SCORE_CORRECT_THRESHOLD = None
EVALUATION_GROUP_SCORE_CORRECT_THRESHOLD = None
EVALUATION_COURSE_SCORE_CORRECT_THRESHOLD = None

# flag indicating whether to generate correct fields for groups and the course based on the given threshold
ADD_EVALUATION_GROUP_IS_CORRECT_FIELD = False
ADD_EVALUATION_COURSE_IS_CORRECT_FIELD = False

# evaluation min and max in dataset docu
EVALUATION_LEARNING_ACTIVITY_SCORE_MINIMUM_IN_DATASET_DOCU = None
EVALUATION_LEARNING_ACTIVITY_SCORE_MAXIMUM_IN_DATASET_DOCU = None
EVALUATION_GROUP_SCORE_MINIMUM_IN_DATASET_DOCU = None
EVALUATION_GROUP_SCORE_MAXIMUM_IN_DATASET_DOCU = None
EVALUATION_COURSE_SCORE_MINIMUM_IN_DATASET_DOCU = None
EVALUATION_COURSE_SCORE_MAXIMUM_IN_DATASET_DOCU = None

########################################################################################################################
### evaluation field for omnibus test per cluster per group ###
########################################################################################################################

OMNIBUS_TESTS_EVAlUATION_METRIC_NAME_STR = EVALUATION_GROUP_IS_CORRECT_FIELD_NAME_STR
OMNIBUS_TESTS_EVAlUATION_METRIC_IS_CATEGORICAL_NAME_STR = True
OMNIBUS_TESTS_EVAlUATION_METRIC_IS_PCT_NAME_STR = False
OMNIBUS_TESTS_EVAlUATION_METRIC_IS_RATIO_NAME_STR = False

########################################################################################################################
### pickle paths and file names ###
########################################################################################################################

SEQUENCE_STATISTICS_PICKLE_PATH_LIST, SEQUENCE_STATISTICS_PICKLE_FULL_NAME = return_pickle_path_list_and_name(DATASET_NAME, 
                                                                                                              PATH_TO_SEQUENCE_STATISTICS_PICKLE_FOLDER,
                                                                                                              SEQUENCE_STATISTICS_PICKLE_NAME)

SEQUENCE_DISTANCE_ANALYTICS_PICKLE_PATH_LIST, SEQUENCE_DISTANCE_ANALYTICS_PICKLE_FULL_NAME = return_pickle_path_list_and_name(DATASET_NAME, 
                                                                                                                              PATH_TO_SEQUENCE_DISTANCE_ANALYTICS_PICKLE_FOLDER,
                                                                                                                              SEQUENCE_DISTANCE_ANALYTICS_PICKLE_NAME)

SEQUENCE_DISTANCE_CLUSTER_ANALYSIS_PICKLE_PATH_LIST, SEQUENCE_DISTANCE_CLUSTER_ANALYSIS_PICKLE_FULL_NAME = return_pickle_path_list_and_name(DATASET_NAME, 
                                                                                                                                            PATH_TO_SEQUENCE_DIST_CLUSTER_ANALYSIS_PICKLE_FOLDER,
                                                                                                                                            SEQUENCE_DISTANCE_CLUSTER_ANALYSIS_OBJECT_PICKLE_NAME)

CLUSTER_EVAL_METRIC_OMNIBUS_TESTS_PICKLE_PATH_LIST, CLUSTER_EVAL_METRIC_OMNIBUS_TESTS_PICKLE_FULL_NAME = return_pickle_path_list_and_name(DATASET_NAME, 
                                                                                                                                          PATH_TO_CLUSTER_EVAL_METRIC_OMNIBUS_TESTS_PICKLE_FOLDER,
                                                                                                                                          CLUSTER_EVAL_METRIC_OMNIBUS_TESTS_OBJECT_PICKLE_NAME)

RESULT_TABLES_PICKLE_PATH_LIST, RESULT_TABLES_PICKLE_FULL_NAME = return_pickle_path_list_and_name(DATASET_NAME, 
                                                                                                  PATH_TO_RESULT_TABLES_PICKLE_FOLDER,
                                                                                                  RESULT_TABLES_PICKLE_NAME)

########################################################################################################################
### initialize ResultTables which transform and holds preprocessing and result data in table form ###
########################################################################################################################

# used for generating result tables
EVALUATION_SCORE_RANGES_DATA_LIST = [[LEARNING_ACTIVITY_FIELD, EVALUATION_LEARNING_ACTIVITY_SCORE_FIELD, EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_FIELD, EVALUATION_LEARNING_ACTIVITY_SCORE_CORRECT_THRESHOLD, EVALUATION_LEARNING_ACTIVITY_SCORE_MINIMUM_IN_DATASET_DOCU, EVALUATION_LEARNING_ACTIVITY_SCORE_MAXIMUM_IN_DATASET_DOCU], 
                                     [GROUP_FIELD, EVALUATION_GROUP_SCORE_FIELD, EVALUATION_GROUP_IS_CORRECT_FIELD, EVALUATION_GROUP_SCORE_CORRECT_THRESHOLD, EVALUATION_GROUP_SCORE_MINIMUM_IN_DATASET_DOCU, EVALUATION_GROUP_SCORE_MAXIMUM_IN_DATASET_DOCU], 
                                     [COURSE_FIELD, EVALUATION_COURSE_SCORE_FIELD, EVALUATION_COURSE_IS_CORRECT_FIELD, EVALUATION_COURSE_SCORE_CORRECT_THRESHOLD, EVALUATION_COURSE_SCORE_MINIMUM_IN_DATASET_DOCU, EVALUATION_COURSE_SCORE_MAXIMUM_IN_DATASET_DOCU]]

result_tables = ResultTables(DATASET_NAME,
                             GROUP_FIELD,
                             EVALUATION_SCORE_RANGES_DATA_LIST,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None)

########################################################################################################################
### dataset specific functions ###
########################################################################################################################

# calculate_eval_metrics #
def calculate_eval_metrics(df: pd.DataFrame):
    """Calculate evaluation metrics per single learning_activity. This function should be used as input of the
    add_evaluation_fields function.

    Parameters
    ----------
    df : pd.DataFrame
        A subset of the interactions dataframe containing data of a single learning_activity

    Returns
    -------
    tuple
        A tuple of evaluation metric scalars per learning activity
    """
    df = df.copy()

    # number interactions
    number_interactions_total = df.shape[0]

    # number attempts (needs to be the same as interactions)
    number_attempts_total = df.shape[0]

    # number hints
    number_hints_total = None

    # total time
    time_total = None
        
    # scores
    # single score
    single_score = None
    # single score hint lowest 
    single_score_hint_lowest = None 
    # single score not first attempt lowest
    single_score_not_first_attempt_lowest = None
    # highest score
    score_highest = None
    # highest score without hint
    score_highest_without_hint = None
    # score first attempt
    score_first_attempt = None
    # score last attempt
    score_last_attempt = None
    # number interactions until highest score
    number_interactions_until_score_highest = None
    # number attempts until highest score
    number_attempts_until_score_highest = None
    # number hints until highest score
    number_hints_until_score_highest = None
    # time until highest score
    time_until_score_highest = None

    # corrects
    # correct
    is_correct = None
    # correct without hint
    is_correct_without_hint = None
    # correct first attempt
    is_correct_first_attempt = None
    # correct first attempt without hint
    is_correct_first_attempt_without_hint = None
    # correct last attempt
    is_correct_last_attempt = None
    # correct last attempt without hint
    is_correct_last_attempt_without_hint = None
    # number interactions until correct
    number_interactions_until_correct = None
    # number attempts until correct
    number_attempts_until_correct = None
    # number hints until correct
    number_hints_until_correct = None
    # time until correct
    time_until_correct = None

    return (number_interactions_total,
            number_attempts_total,
            number_hints_total,
            time_total,
            single_score,
            single_score_hint_lowest,
            single_score_not_first_attempt_lowest,
            score_highest,
            score_highest_without_hint,
            score_first_attempt,
            score_last_attempt,
            number_interactions_until_score_highest,
            number_attempts_until_score_highest,
            number_hints_until_score_highest,
            time_until_score_highest,
            is_correct,
            is_correct_without_hint,
            is_correct_first_attempt,
            is_correct_first_attempt_without_hint,
            is_correct_last_attempt,
            is_correct_last_attempt_without_hint,
            number_interactions_until_correct,
            number_attempts_until_correct,
            number_hints_until_correct,
            time_until_correct)

#kdd_2015 specific preprocessing functions
class ImportInteractionsKDD2015(ImportInteractions):
    """Imports the interaction dataframe and filters it by interactions of a specified type. 

    Attributes
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    interaction_types_df : pd.DataFrame
        A dataframe counting the occurrences of all interactions types
    """    

    def _create_interactions_df(self) -> pd.DataFrame:

        interactions = pd.read_csv(PATH_TO_INTERACTION_DATA)
        enrollments = pd.read_csv(PATH_TO_ENROLLMENT_DATA)
        dropouts = pd.read_csv(PATH_TO_DROPOUT_DATA, names=[ENROLLMENT_FIELD, DROPOUT_FIELD])
        objects = pd.read_csv(PATH_TO_OBJECT_DATA) 

        interactions = typecast_fields(interactions,
                                       TIMESTAMP_FIELD,
                                       None,
                                       None,
                                       LEARNING_ACTIVITY_FIELD)

        enrollments = typecast_fields(enrollments,
                                      None,
                                      GROUP_FIELD,
                                      USER_FIELD,
                                      None)

        objects = objects[[MODULE_FIELD, CATEGORY_FIELD]].rename(columns={MODULE_FIELD: LEARNING_ACTIVITY_FIELD})
        objects = objects.loc[~objects.duplicated(), :]

        interactions = pd.merge(interactions, enrollments, how='inner', on=ENROLLMENT_FIELD)
        interactions = pd.merge(interactions, dropouts, how='inner', on=ENROLLMENT_FIELD)
        interactions = interactions.merge(objects, how='left', on=LEARNING_ACTIVITY_FIELD)

        return interactions

class TransformInteractionsKDD2015(TransformInteractions):
     
    def _transform_interactions_df(self,
                                   interactions: pd.DataFrame) -> pd.DataFrame:

        interactions[EVALUATION_GROUP_IS_CORRECT_FIELD] = interactions[DROPOUT_FIELD].map({0: True, 1: False})

        return interactions
    