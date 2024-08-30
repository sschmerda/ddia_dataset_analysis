from .constants import *
from .standard_import import *

class ContingencyEffectSizeEnum(Enum):
    cramer = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_CRAMERS_V_VALUE_NAME_STR
    tschuprow = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_TSCHUPROWS_T_VALUE_NAME_STR
    pearson = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_PEARSONS_CONTINGENCY_COEFFICIENT_VALUE_NAME_STR
    

########################################################################################################################
### continuous variable test options ###
########################################################################################################################

#TODO

########################################################################################################################
### contingency table test options ###
########################################################################################################################

# min permitted expected frequency for an element in a contingency table
OMNIBUS_TESTS_CONTINGENCY_MIN_EXPECTED_FREQ_VALUE = 5
OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE = True
OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_METHOD_VALUE = ContingencyEffectSizeEnum.cramer 

########################################################################################################################
### bootstrapping options ###
########################################################################################################################

OMNIBUS_TESTS_BOOTSTRAPPING_N_RESAMPLES = 100_000
OMNIBUS_TESTS_BOOTSTRAPPING_VECTORIZED = False
OMNIBUS_TESTS_BOOTSTRAPPING_PAIRED = True
OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL = 0.95
OMNIBUS_TESTS_BOOTSTRAPPING_ALTERNATIVE = 'two-sided'
OMNIBUS_TESTS_BOOTSTRAPPING_METHOD = 'bca' # one in [‘percentile’, ‘basic’, ‘bca’]

