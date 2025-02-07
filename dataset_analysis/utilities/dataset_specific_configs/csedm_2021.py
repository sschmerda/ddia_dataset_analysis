from ..constants.constants import *
from ..standard_import import *
from ..preprocessing_functions import *
from ..io_functions import *
from ..result_tables import *

########################################################################################################################
### dataset name ###
########################################################################################################################

DATASET_NAME = 'csedm_2021'

########################################################################################################################
### paths ###
########################################################################################################################

PATH_TO_INTERACTION_DATA = '../../../../../../data/ddia/CSEDM_2021_F19_Release_All_05_23_22/All/Data/MainTable.csv'
PATH_TO_COURSE_EVALUATION_DATA = '../../../../../../data/ddia/CSEDM_2021_F19_Release_All_05_23_22/All/Data/LinkTables/Subject.csv'

########################################################################################################################
### required fields ###
########################################################################################################################

USER_FIELD = 'SubjectID'
GROUP_FIELD = 'AssignmentID'
LEARNING_ACTIVITY_FIELD = 'ProblemID'
COURSE_FIELD = 'Course'
TIMESTAMP_FIELD = 'ServerTimestamp'
ORDER_FIELD = 'Order'

########################################################################################################################
### dataset specific constants ###
########################################################################################################################

# fields used in dataset preparation
EVENT_TYPE_FIELD = 'EventType'

# strings and values used in dataset preparation
EVENT_TYPE_RUN_PROGRAM_VALUE_STR = 'Run.Program'
EVENT_TYPE_COMPILE_VALUE_STR = 'Compile'
EVENT_TYPE_COMPILE_ERROR_VALUE_STR = 'Compile.Error'

########################################################################################################################
### evaluation fields ###
########################################################################################################################

# learning activity level
EVALUATION_LEARNING_ACTIVITY_SCORE_FIELD = 'Score'
EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_FIELD = None
# group level
EVALUATION_GROUP_SCORE_FIELD = None
EVALUATION_GROUP_IS_CORRECT_FIELD = None 
# course level
EVALUATION_COURSE_SCORE_FIELD = 'X-Grade'
EVALUATION_COURSE_IS_CORRECT_FIELD = None 

########################################################################################################################
### data import and preprocessing config ###
########################################################################################################################

INTERACTION_TYPE_VALUES_TO_KEEP_LIST = [EVENT_TYPE_RUN_PROGRAM_VALUE_STR,
                                        EVENT_TYPE_COMPILE_VALUE_STR,
                                        EVENT_TYPE_COMPILE_ERROR_VALUE_STR]

DROP_LEARNING_ACTIVITY_SEQUENCE_IF_NA_FIELD_LIST = [TIMESTAMP_FIELD,
                                                    LEARNING_ACTIVITY_FIELD,
                                                    ORDER_FIELD]
FIELD_VALUE_TUPLE_NA_FILTER_EXCEPTION_LIST = []

DROP_ROW_IF_NA_FIELD_LIST = [GROUP_FIELD,
                             USER_FIELD]

########################################################################################################################
### evaluation fields correct field calculation  ###
########################################################################################################################

# score threshold is correct
EVALUATION_LEARNING_ACTIVITY_SCORE_CORRECT_THRESHOLD = 1
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

OMNIBUS_TESTS_EVAlUATION_METRIC_NAME_STR = EVALUATION_GROUP_SCORE_HIGHEST_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_NAME_STR
OMNIBUS_TESTS_EVAlUATION_METRIC_IS_CATEGORICAL_NAME_STR = False
OMNIBUS_TESTS_EVAlUATION_METRIC_IS_PCT_NAME_STR = True
OMNIBUS_TESTS_EVAlUATION_METRIC_IS_RATIO_NAME_STR = True

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
def calculate_eval_metrics(df: pd.DataFrame,
                           evaluation_score_field=None,
                           evaluation_score_correct_threshold=None,
                           event_type_field=None,
                           event_type_run_program_value_str=None,
                           server_timestamp_field=None):
    """Calculate evaluation metrics per single learning_activity. This function should be used as input of the
    add_evaluation_fields function.

    Parameters
    ----------
    df : pd.DataFrame
        A subset of the interactions dataframe containing data of a single learning_activity
    evaluation_score_field : _type_
        The learning_activity evaluation score field column, by default None
    evaluation_score_correct_threshold : _type_
        The threshold for the learning_activity score values to be evaluated as correct, by default None
    event_type_field : _type_, optional
        The event_type field column, by default None
    event_type_run_program_value_str : _type_
        The run_program value of the event_type field, by default None
    server_timestamp_field : _type_
        The server_timestamp_field field, by default None

    Returns
    -------
    tuple
        A tuple of evaluation metric scalars per learning activity
    """
    df = df.copy()

    # generate corrects from score
    evaluation_correct_field = 'correct'
    df[evaluation_correct_field] = df[evaluation_score_field].apply(lambda x: float(x) if pd.notna(x) else x) >= evaluation_score_correct_threshold

    # number interactions (Run.Program)
    number_interactions_total = sum(df[event_type_field]==event_type_run_program_value_str)

    # number attempts
    number_attempts_total = sum(df[event_type_field]==event_type_run_program_value_str)

    # number hints
    number_hints_total = None

    # total time
    # (timedelta between first and last run_program interaction, can be 0 for only 1 attempt)
    run_program_df = df.loc[df[event_type_field]==event_type_run_program_value_str, :]
    time_total = round((run_program_df[server_timestamp_field].iloc[-1] - run_program_df[server_timestamp_field].iloc[0]).total_seconds())
        
    # scores
    # single score
    single_score = None
    # single score hint lowest 
    single_score_hint_lowest = None 
    # single score not first attempt lowest
    single_score_not_first_attempt_lowest = None
    # highest score
    score_highest = df[evaluation_score_field].max()
    # highest score without hint
    score_highest_without_hint = None
    # score first attempt
    score_first_attempt = df[evaluation_score_field][df[event_type_field]==event_type_run_program_value_str].to_list()[0]
    # score last attempt
    score_last_attempt = df[evaluation_score_field][df[event_type_field]==event_type_run_program_value_str].to_list()[-1]
    # number interactions until highest score
    number_interactions_until_score_highest = np.argmax(df[evaluation_score_field][df[event_type_field]==event_type_run_program_value_str]) + 1
    # number attempts until highest score
    number_attempts_until_score_highest = np.argmax(df[evaluation_score_field][df[event_type_field]==event_type_run_program_value_str]) + 1
    # number hints until highest score
    number_hints_until_score_highest = None
    # time until highest score
    # (timedelta between first run_program interaction and highest score run_program interaction, can be 0 for only 1 attempt)
    index_highest_score = np.argmax(run_program_df[evaluation_score_field])
    time_until_score_highest = round((run_program_df[server_timestamp_field].iloc[index_highest_score] - run_program_df[server_timestamp_field].iloc[0]).total_seconds())

    # corrects
    # correct
    is_correct = df[evaluation_correct_field].any()
    # correct without hint
    is_correct_without_hint = None
    # correct first attempt
    try:
        is_correct_first_attempt = df[evaluation_correct_field][df[event_type_field]==event_type_run_program_value_str].to_list()[0]
    except:
        is_correct_first_attempt = False
    # correct first attempt without hint
    is_correct_first_attempt_without_hint = None
    # correct last attempt
    try:
        is_correct_last_attempt = df[evaluation_correct_field][df[event_type_field]==event_type_run_program_value_str].to_list()[-1]
    except:
        is_correct_last_attempt = False
    # correct last attempt without hint
    is_correct_last_attempt_without_hint = None
    # number interactions until correct
    if is_correct:
        number_interactions_until_correct = np.argmax(df[evaluation_correct_field][df[event_type_field]==event_type_run_program_value_str]) + 1
    else:
        number_interactions_until_correct = -1 # -1 means that there was no correct -> therefore also no interactions until correct
    # number attempts until correct
    if is_correct:
        number_attempts_until_correct = np.argmax(df[evaluation_correct_field][df[event_type_field]==event_type_run_program_value_str]) + 1
    else:
        number_attempts_until_correct = -1 # -1 means that there was no correct -> therefore also no attempts until correct
    # number hints until correct
    number_hints_until_correct = None
    # time until correct
    # (timedelta between first run_program interaction and first is_correct run_program interaction, can be 0 for only 1 attempt)
    if is_correct:
        index_is_correct = np.argmax(run_program_df[evaluation_correct_field])
        time_until_correct = round((run_program_df[server_timestamp_field].iloc[index_is_correct] - run_program_df[server_timestamp_field].iloc[0]).total_seconds())
    else:
        time_until_correct = -1 # -1 means that there was no correct -> therefore also no attempts until correct

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

#csedm_2021 specific preprocessing functions
class ImportInteractionsCSEDM2021(ImportInteractions):
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
        course_grade = pd.read_csv(PATH_TO_COURSE_EVALUATION_DATA)

        interactions = typecast_fields(interactions,
                                       TIMESTAMP_FIELD,
                                       GROUP_FIELD,
                                       USER_FIELD,
                                       LEARNING_ACTIVITY_FIELD)

        course_grade = typecast_fields(course_grade,
                                       None,
                                       None,
                                       USER_FIELD,
                                       None)

        interactions = interactions.merge(course_grade, how="left", on=USER_FIELD)

        return interactions

class TransformInteractionsCSEDM2021(TransformInteractions):

    def _transform_interactions_df(self,
                                   interactions: pd.DataFrame) -> pd.DataFrame:

        return interactions
    