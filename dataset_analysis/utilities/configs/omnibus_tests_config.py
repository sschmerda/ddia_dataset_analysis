from ..constants import *
from ..standard_import import *

########################################################################################################################
### option enums ###
########################################################################################################################

class ContingencyMeasureAssociationEnum(Enum):
    CRAMER = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_CRAMERS_V_VALUE_NAME_STR
    CRAMER_BIAS_CORRECTED = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_CRAMERS_V_BIAS_CORRECTED_VALUE_NAME_STR
    TSCHUPROW = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_TSCHUPROWS_T_VALUE_NAME_STR
    PEARSON = OMNIBUS_TESTS_CONTINGENCY_TEST_EFFECT_SIZE_PEARSONS_CONTINGENCY_COEFFICIENT_VALUE_NAME_STR

class ContingencyMeasureAssociationStrengthGuidelineEnum(Enum):
    COHEN_1988 = OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_COHEN_1988_GUIDELINE_VALUE_NAME_STR
    GIGNAC_SZODORAI_2016 = OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_GIGNAC_SZODORAI_2016_GUIDELINE_VALUE_NAME_STR
    FUNDER_OZER_2019 = OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_FUNDER_OZER_2019_GUIDELINE_VALUE_NAME_STR
    LOVAKOV_AGADULLINA_2021 = OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_LOVAKOV_AGADULLINA_GUIDELINE_VALUE_NAME_STR
    
class AOVMeasueAssociationEnum(Enum):
    ETA_SQUARED = OMNIBUS_TESTS_CONTINUOUS_TEST_EFFECT_SIZE_ETA_SQUARED_VALUE_NAME_STR
    COHENS_F = OMNIBUS_TESTS_CONTINUOUS_TEST_EFFECT_SIZE_COHENS_F_VALUE_NAME_STR
    OMEGA_SQUARED = OMNIBUS_TESTS_CONTINUOUS_TEST_EFFECT_SIZE_OMEGA_SQUARED_VALUE_NAME_STR

class AOVMeasureAssociationStrengthGuidelineEnum(Enum):
    COHEN_1988 =  OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_COHEN_1988_GUIDELINE_VALUE_NAME_STR
    COHEN_1988_F =  OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_COHEN_1988_F_GUIDELINE_VALUE_NAME_STR

class PValueCorrectionEnum(Enum):
    BONFERRONI = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_BONFERRONI_VALUE_NAME_STR
    SIDAK = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_SIDAK_VALUE_NAME_STR 
    HOLM_SIDAK = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_HOLM_SIDAK_VALUE_NAME_STR
    HOLM = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_HOLM_VALUE_NAME_STR
    SIMES_HOCHBERG = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_SIMES_HOCHBERG_VALUE_NAME_STR
    HOMMEL = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_HOMMEL_VALUE_NAME_STR
    FDR_BH = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_FDR_BH_VALUE_NAME_STR
    FDR_BY = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_FDR_BY_VALUE_NAME_STR
    FDR_TSBH = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_FDR_TSBH_VALUE_NAME_STR
    FDR_TSBKY = OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_FDR_TSBKY_VALUE_NAME_STR

########################################################################################################################
### general ###
########################################################################################################################

OMNIBUS_TESTS_ALPHA_LEVEL = 0.05
OMNIBUS_TESTS_ONE_STAR_UPPER_BOUND = 0.05
OMNIBUS_TESTS_TWO_STAR_UPPER_BOUND = 0.01
OMNIBUS_TESTS_THREE_STAR_UPPER_BOUND = 0.001
RESULTS_ROUND_N_DIGITS = 6

########################################################################################################################
### chi squared test options ###
########################################################################################################################

OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES = 100_000
OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_VALUE = 5
OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE = False
# needs to be a dict with entries of the form: key=measure_of_association; value=[measure_of_association_strength_guideline]
OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_DICT = {ContingencyMeasureAssociationEnum.CRAMER: [ContingencyMeasureAssociationStrengthGuidelineEnum.COHEN_1988,
                                                                                                    ContingencyMeasureAssociationStrengthGuidelineEnum.GIGNAC_SZODORAI_2016],
                                                         ContingencyMeasureAssociationEnum.CRAMER_BIAS_CORRECTED: [ContingencyMeasureAssociationStrengthGuidelineEnum.COHEN_1988,
                                                                                                                   ContingencyMeasureAssociationStrengthGuidelineEnum.GIGNAC_SZODORAI_2016]}
# the measure_of_association used for plotting
OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE = ContingencyMeasureAssociationEnum.CRAMER

########################################################################################################################
### analysis of variance test options ###
########################################################################################################################

OMNIBUS_TESTS_AOV_PERMUTATION_N_RESAMPLES = 100_000
OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_BATCH_SIZE = 2000
# needs to be a dict with entries of the form: key=measure_of_association; value=[measure_of_association_strength_guideline]
OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_DICT = {AOVMeasueAssociationEnum.OMEGA_SQUARED: [AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988]}
                                                 
# the measure_of_association used for plotting
OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE = AOVMeasueAssociationEnum.OMEGA_SQUARED

########################################################################################################################
### p-value correction options ###
########################################################################################################################

# needs to be a list with elements of the form: p_value_correction_method
OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_LIST = [PValueCorrectionEnum.FDR_BY]
# the p_value_correction_method used for plotting
OMNIBUS_TESTS_P_VALUE_CORRECTION_PLOT_INCLUDE = PValueCorrectionEnum.FDR_BY

########################################################################################################################
### measure of association ci bootstrapping options ###
########################################################################################################################

OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES = 100_000
OMNIBUS_TESTS_BOOTSTRAPPING_VECTORIZED = False
OMNIBUS_TESTS_BOOTSTRAPPING_PAIRED = True #TODO: is paired=True correct?
OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL = 0.95
OMNIBUS_TESTS_BOOTSTRAPPING_ALTERNATIVE = 'two-sided'
OMNIBUS_TESTS_BOOTSTRAPPING_METHOD = 'bca' # one in [‘percentile’, ‘basic’, ‘bca’]

