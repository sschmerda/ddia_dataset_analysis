from ..constants.constants import *
from ..constants.enums import *
from ..standard_import import *

########################################################################################################################
### conf int type config ###
########################################################################################################################

SEQUENCE_STATISTIC_LIST = [SequenceStatistic.SEQUENCE_LENGTH,
                           SequenceStatistic.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ,
                           SequenceStatistic.PCT_REPEATED_LEARNING_ACTIVITIES,
                           SequenceStatistic.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES,
                           SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES,
                           SequenceStatistic.MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES,
                           SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES]

ESTIMATOR_LIST = [ConfIntEstimator.MEAN, 
                  ConfIntEstimator.MEDIAN]

SEQUENCE_TYPE_LIST = [SequenceType.ALL_SEQUENCES,
                      SequenceType.UNIQUE_SEQUENCES]


########################################################################################################################
### bootstrap parameters ###
########################################################################################################################

CONFIDENCE_INTERVAL_N_BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_INTERVAL_BOOTSTRAP_METHOD = 'bca'
CONFIDENCE_INTERVAL_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

########################################################################################################################
### avg sequence distance bootstrap parameters ###
########################################################################################################################

CONFIDENCE_INTERVAL_BOOTSTRAP_SEQ_DIST_SELF_DISTANCE_FILTER = SelfDistanceFilter.ALL_SELF_DISTANCES