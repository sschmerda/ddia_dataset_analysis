from ..standard_import import *
from ..constants.constants import *

########################################################################################################################
### option enums ###
########################################################################################################################

class SequencePreprocessingBase(Enum):
    PCT_UNIQUE_LEARNING_RESOURCES = LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_NAME_STR
    PCT_REPEATED_LEARNING_RESOURCES = LEARNING_ACTIVITY_SEQUENCE_PCT_REPEATED_LEARNING_ACTIVITIES_NAME_STR
    SEQUENCE_LENGTH = LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR

########################################################################################################################
### general ###
########################################################################################################################

MIN_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_THRESHOLD = 50
MAX_PCT_REPEATED_LEARNING_ACTIVITIES_IN_SEQ_THRESHOLD = 80
MIN_SEQUENCE_NUMBER_PER_GROUP_THRESHOLD = 100
MIN_UNIQUE_SEQUENCE_NUMBER_PER_GROUP_THRESHOLD = 25

########################################################################################################################
### plot_avg_sequence_statistics_per_group_per_dataset options ###
########################################################################################################################

PLOT_SEQUENCE_FILTER_THRESHOLDS_DATA_IS_PCT_SHARE_X_AXIS = True
PLOT_SEQUENCE_FILTER_THRESHOLDS_DATA_IS_NOT_PCT_SHARE_X_AXIS = False