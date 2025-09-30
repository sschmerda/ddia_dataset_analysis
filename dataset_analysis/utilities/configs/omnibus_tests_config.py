from ..constants.constants import *
from ..constants.enums import *
from ..standard_import import *

########################################################################################################################
### measure of association strength guideline classes ###
########################################################################################################################

class MeasureAssociationStrength(ABC):
    
    association_strength_values = {}

    @classmethod
    def return_moa_strength(cls,
                            measure_of_association_value: float) -> MeasureAssociationStrengthValuesEnum:
        
        for moa_strength, (lower_bound_incl, upper_bound_excl) in cls.association_strength_values.items():

            if lower_bound_incl <= measure_of_association_value < upper_bound_excl:
                return moa_strength

class Cohen1988MeasureAssociationStrengthContingency(MeasureAssociationStrength):
    """Cohen's (1988, pp. 224 ff.) measure of association strength for Cohen's w, Cramer's phi, Tschuprow’s T and Phi.
    ---
    [1] J. Cohen, Statistical power analysis for the behavioral sciences, 2nd ed. Hillsdale, N.J: L. Erlbaum Associates, 1988.
    [2] “Automated Interpretation of Indices of Effect Size.” Accessed: Apr. 23, 2025. [Online]. Available: https://easystats.github.io/effectsize/articles/interpret.html#correlation-r
    """

    # upper bounds not inclusive
    association_strength_values = {MeasureAssociationStrengthValuesEnum.VERY_SMALL: (-np.inf, 0.1),
                                   MeasureAssociationStrengthValuesEnum.SMALL: (0.1, 0.3),
                                   MeasureAssociationStrengthValuesEnum.MEDIUM: (0.3, 0.5),
                                   MeasureAssociationStrengthValuesEnum.LARGE: (0.5, np.inf)}

class GignacSzodorai2016MeasureAssociationStrengthContingency(MeasureAssociationStrength):
    """Gignac's and Szodorai's (2016) measure of association strength for Cohen's w, Cramer's phi, Tschuprow’s T and Phi.
    ---
    [1] G. E. Gignac and E. T. Szodorai, “Effect size guidelines for individual differences researchers,” Personality and Individual Differences, vol. 102, pp. 74–78, Nov. 2016, doi: 10.1016/j.paid.2016.06.069.
    [2] “Automated Interpretation of Indices of Effect Size.” Accessed: Apr. 23, 2025. [Online]. Available: https://easystats.github.io/effectsize/articles/interpret.html#correlation-r
    """

    # upper bounds not inclusive
    association_strength_values = {MeasureAssociationStrengthValuesEnum.VERY_SMALL: (-np.inf, 0.1),
                                   MeasureAssociationStrengthValuesEnum.SMALL: (0.1, 0.2),
                                   MeasureAssociationStrengthValuesEnum.MEDIUM: (0.2, 0.3),
                                   MeasureAssociationStrengthValuesEnum.LARGE: (0.3, np.inf)}

class FunderOzer2019MeasureAssociationStrengthContingency(MeasureAssociationStrength):
    """Funder's and Ozer's (2019) measure of association strength for Cohen's w, Cramer's phi, Tschuprow’s T and Phi.
    ---
    [1] D. C. Funder and D. J. Ozer, “Evaluating Effect Size in Psychological Research: Sense and Nonsense,” Advances in Methods and Practices in Psychological Science, vol. 2, no. 2, pp. 156–168, Jun. 2019, doi: 10.1177/2515245919847202.
    [2] “Automated Interpretation of Indices of Effect Size.” Accessed: Apr. 23, 2025. [Online]. Available: https://easystats.github.io/effectsize/articles/interpret.html#correlation-r
    """

    # upper bounds not inclusive
    association_strength_values = {MeasureAssociationStrengthValuesEnum.TINY: (-np.inf, 0.05),
                                   MeasureAssociationStrengthValuesEnum.VERY_SMALL: (0.05, 0.1),
                                   MeasureAssociationStrengthValuesEnum.SMALL: (0.1, 0.2),
                                   MeasureAssociationStrengthValuesEnum.MEDIUM: (0.2, 0.3),
                                   MeasureAssociationStrengthValuesEnum.LARGE: (0.3, 0.4),
                                   MeasureAssociationStrengthValuesEnum.VERY_LARGE: (0.4, np.inf)}

class LovakovAgadullina2021MeasureAssociationStrengthContingency(MeasureAssociationStrength):
    """Lovakov's and Agadullina's (2016) measure of association strength for Cohen's w, Cramer's phi, Tschuprow’s T and Phi.
    ---
    [1] A. Lovakov and E. R. Agadullina, “Empirically derived guidelines for effect size interpretation in social psychology,” Euro J Social Psych, vol. 51, no. 3, pp. 485–504, Apr. 2021, doi: 10.1002/ejsp.2752.
    [2] “Automated Interpretation of Indices of Effect Size.” Accessed: Apr. 23, 2025. [Online]. Available: https://easystats.github.io/effectsize/articles/interpret.html#correlation-r
    """

    # upper bounds not inclusive
    association_strength_values = {MeasureAssociationStrengthValuesEnum.VERY_SMALL: (-np.inf, 0.12),
                                   MeasureAssociationStrengthValuesEnum.SMALL: (0.12, 0.24),
                                   MeasureAssociationStrengthValuesEnum.MEDIUM: (0.24, 0.41),
                                   MeasureAssociationStrengthValuesEnum.LARGE: (0.41, np.inf)}

class Cohen1988MeasureAssociationStrengthAOV(MeasureAssociationStrength):
    """A conversion of Cohen's (1988, p.283) measure of association strength for Cohen's f to Omega Squared.
    ---
    [1] J. Cohen, Statistical power analysis for the behavioral sciences, 2nd ed. Hillsdale, N.J: L. Erlbaum Associates, 1988.
    [2] “Automated Interpretation of Indices of Effect Size.” Accessed: Apr. 23, 2025. [Online]. Available: https://easystats.github.io/effectsize/articles/interpret.html#correlation-r
    """

    # upper bounds not inclusive
    association_strength_values = {MeasureAssociationStrengthValuesEnum.VERY_SMALL: (-np.inf, 0.0099),
                                   MeasureAssociationStrengthValuesEnum.SMALL: (0.0099, 0.0588),
                                   MeasureAssociationStrengthValuesEnum.MEDIUM: (0.0588, 0.1379),
                                   MeasureAssociationStrengthValuesEnum.LARGE: (0.1379, np.inf)}

class Cohen1988FMeasureAssociationStrengthAOV(MeasureAssociationStrength):
    """Cohen's (1988, pp. 284 ff.) measure of association strength for Cohen's f.
    ---
    [1] J. Cohen, Statistical power analysis for the behavioral sciences, 2nd ed. Hillsdale, N.J: L. Erlbaum Associates, 1988.
    """

    # upper bounds not inclusive
    association_strength_values = {MeasureAssociationStrengthValuesEnum.VERY_SMALL: (-np.inf, 0.1),
                                   MeasureAssociationStrengthValuesEnum.SMALL: (0.1, 0.25),
                                   MeasureAssociationStrengthValuesEnum.MEDIUM: (0.25, 0.4),
                                   MeasureAssociationStrengthValuesEnum.LARGE: (0.4, np.inf)}

########################################################################################################################
### general ###
########################################################################################################################

OMNIBUS_TESTS_ALPHA_LEVEL = 0.05
OMNIBUS_TESTS_ONE_STAR_UPPER_BOUND = 0.05
OMNIBUS_TESTS_TWO_STAR_UPPER_BOUND = 0.01
OMNIBUS_TESTS_THREE_STAR_UPPER_BOUND = 0.001
OMNIBUS_TEST_RESULTS_ROUND_N_DIGITS = 6
OMNIBUS_TESTS_EXCLUDE_NON_CLUSTERED_SEQUENCES = True
OMNIBUS_TESTS_INCLUDE_R_TEST_RESULTS_SEQUENCES = True

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
# the measure_of_association strength guideline used for plotting
OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_PLOT_INCLUDE = ContingencyMeasureAssociationStrengthGuidelineEnum.GIGNAC_SZODORAI_2016

########################################################################################################################
### analysis of variance test options ###
########################################################################################################################

OMNIBUS_TESTS_AOV_PERMUTATION_N_RESAMPLES = 100_000
# needs to be a dict with entries of the form: key=measure_of_association; value=[measure_of_association_strength_guideline]
OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_DICT = {AOVMeasueAssociationEnum.OMEGA_SQUARED: [AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988]}
                                                 
# the measure_of_association used for plotting
OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE = AOVMeasueAssociationEnum.OMEGA_SQUARED
# the measure_of_association strength guideline used for plotting
OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_PLOT_INCLUDE = AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988

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

########################################################################################################################
### plot aesthetics options ###
########################################################################################################################

OMNIBUS_TESTS_ADJUST_Y_LABEL = True
OMNIBUS_TESTS_Y_LABEL_SPLIT_STRING = ' '
OMNIBUS_TESTS_Y_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT = 3
OMNIBUS_TESTS_Y_LABEL_RIGHT_PADDING = 25
OMNIBUS_TESTS_Y_LABEL_ROTATION = 0

OMNIBUS_TESTS_ADJUST_X_LABEL = False
OMNIBUS_TESTS_X_LABEL_SPLIT_STRING = ' '
OMNIBUS_TESTS_X_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT = 5
OMNIBUS_TESTS_X_LABEL_VERTICAL_PADDING = 30
OMNIBUS_TESTS_X_LABEL_ROTATION = 0