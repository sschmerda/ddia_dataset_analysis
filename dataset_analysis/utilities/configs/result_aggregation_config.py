from .general_config import *
from .omnibus_tests_config import *
from ..constants.constants import *
from ..standard_import import *
from ..plotting_functions import *

########################################################################################################################
### figure save options ###
########################################################################################################################

SAVE_FIGURE_DPI = 400
SAVE_FIGURE_IMAGE_FORMAT = 'png'
SAVE_FIGURE_BBOX_INCHES = 'tight'

########################################################################################################################
### figure plot options ###
########################################################################################################################

RESULT_AGGREGATION_FIG_SIZE_WIDTH_INCH = 16
RESULT_AGGREGATION_FIG_SIZE_HEIGHT_INCH = 8
RESULT_AGGREGATION_FIG_SIZE_DPI = 150
RESULT_AGGREGATION_COLOR_PALETTE = SEABORN_COLOR_PALETTE
RESULT_AGGREGATION_COLOR_SATURATION = 0.75

########################################################################################################################
### facet grid options ###
########################################################################################################################

RESULT_AGGREGATION_FACET_GRID_N_COLUMNS = 3
# facet grid config
RESULT_AGGREGATION_FACET_GRID_ASPECT = 1
RESULT_AGGREGATION_FACET_GRID_HEIGHT = RESULT_AGGREGATION_FIG_SIZE_WIDTH_INCH/(RESULT_AGGREGATION_FACET_GRID_N_COLUMNS * RESULT_AGGREGATION_FACET_GRID_ASPECT)

########################################################################################################################
### option enums ###
########################################################################################################################

class SequenceStatisticsAveragingMethod(Enum):
    MEAN = AVG_SEQUENCE_STATISTICS_AVERAGING_METHOD_MEAN
    MEDIAN = AVG_SEQUENCE_STATISTICS_AVERAGING_METHOD_MEDIAN
    
class SequenceStatisticsPlotFields(Enum):
    SEQUENCE_LENGTH = LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR
    PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ = LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_NAME_STR
    PCT_REPEATED_LEARNING_ACTIVITIES = LEARNING_ACTIVITY_SEQUENCE_PCT_REPEATED_LEARNING_ACTIVITIES_NAME_STR
    MEAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR

class UniqueSequenceFrequencyStatisticsPlotFields(Enum):
    SEQUENCE_FREQUENCY = LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR
    RELATIVE_SEQUENCE_FREQUENCY = LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR

class BoxplotSortMetric(Enum):
    MEAN = BOXPLOT_SORT_METRIC_MEAN
    MEDIAN = BOXPLOT_SORT_METRIC_MEDIAN
    MAX = BOXPLOT_SORT_METRIC_MAX
    MIN = BOXPLOT_SORT_METRIC_MIN

class OmnibusTestResultPValueKind(Enum):
    PVAL = OMNIBUS_TESTS_PVAL_FIELD_NAME_STR
    PVAL_PERM = OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR
    PVAL_R = OMNIBUS_TESTS_R_PVAL_FIELD_NAME_STR
    PVAL_PERM_R = OMNIBUS_TESTS_R_PERM_PVAL_FIELD_NAME_STR

class OmnibusTestResultMeasureAssociationStrengthCalculationBase(Enum):
    MOA_VALUE = 0
    MOA_CONF_INT_LOWER_BOUND = 1
    MOA_CONF_INT_UPPER_BOUND = 2

########################################################################################################################
### sequence statistics plot fields ###
########################################################################################################################

# fields to plot
SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST = [SequenceStatisticsPlotFields.SEQUENCE_LENGTH,
                                           SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ,
                                           SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES,
                                           SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE,
                                           SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE]

UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST = [UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY,
                                                            UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY]

########################################################################################################################
### plot_avg_sequence_statistics_per_group_per_dataset options ###
########################################################################################################################

# averaging method
AVG_SEQUENCE_STATISTICS_AVERAGING_METHOD = SequenceStatisticsAveragingMethod.MEAN

# plot name
AVG_SEQUENCE_STATISTICS_PLOT_NAME = 'average_sequence_statistics_'
# axis labels
AVG_SEQUENCE_STATISTICS_X_LABEL_SUFFIX = f' per {GROUP_FIELD_NAME_STR}' 

# plot decorator
def avg_sequence_statistics_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.spines.left': False,
                              'axes.spines.bottom': True,
                              'axes.spines.right': False,
                              'axes.spines.top': False,
                              'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'figure.figsize': (RESULT_AGGREGATION_FIG_SIZE_WIDTH_INCH, RESULT_AGGREGATION_FIG_SIZE_HEIGHT_INCH),
                           'patch.edgecolor': 'black'}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1.5, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# boxplot color palette
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE = SEABORN_COLOR_PALETTE
# AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE ='vlag'
# color saturation
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_SATURATION = 0.75

# boxplot config
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_OUTLIERS = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_MEANS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WIDTH = 0.8
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_LINE_WIDTH = 2
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WHISKERS = 1.5 
# marker
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_MARKER = {'marker':'o',
                                                                'markerfacecolor':'red', 
                                                                'markeredgecolor':'black',
                                                                'markersize': 10,
                                                                'zorder': 30}

# swarmplot config
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_SIZE = 10
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_COLOR = 'white'
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_ALPHA = 0.8
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_EDGECOLOR = 'black'
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_LINEWIDTH = 1.8

########################################################################################################################
### plot_summary_sequence_statistics_per_group_per_dataset options ###
########################################################################################################################

# plot name
SUMMARY_SEQUENCE_STATISTICS_PLOT_NAME = 'summary_sequence_statistics_'

# plot decorator
def summary_sequence_statistics_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# lineplot config
SUMMARY_SEQUENCE_STATISTICS_KIND = 'line'
SUMMARY_SEQUENCE_STATISTICS_PLOT_LEGEND = False
# marker
SUMMARY_SEQUENCE_STATISTICS_PLOT_MARKERS = True
SUMMARY_SEQUENCE_STATISTICS_MARKER_TYPE = 'o'
SUMMARY_SEQUENCE_STATISTICS_MARKER_SIZE = 5
SUMMARY_SEQUENCE_STATISTICS_MARKER_FACECOLOR = 'red'
SUMMARY_SEQUENCE_STATISTICS_MARKER_EDGECOLOR = 'black'
SUMMARY_SEQUENCE_STATISTICS_MARKER_EDGEWIDTH = 1
SUMMARY_SEQUENCE_STATISTICS_MARKER_ALPHA = 1
# line
SUMMARY_SEQUENCE_STATISTICS_LINE_ALPHA = 0.8
SUMMARY_SEQUENCE_STATISTICS_LINE_WIDTH = 2
# spines
SUMMARY_SEQUENCE_STATISTICS_SHOW_TOP = False
SUMMARY_SEQUENCE_STATISTICS_SHOW_BOTTOM = True
SUMMARY_SEQUENCE_STATISTICS_SHOW_LEFT = True
SUMMARY_SEQUENCE_STATISTICS_SHOW_RIGHT = False
# axes
SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT = False
SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT = True
SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT_RATIO = False
SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT_RATIO = True
SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW = False
SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW = True

########################################################################################################################
### plot_sequence_statistics_distribution_per_group_per_dataset options ###
########################################################################################################################

# plot name
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_NAME = 'sequence_statistics_distribution_'
SEQUENCE_STATISTICS_DISTRIBUTION_NON_UNIQUE_UNIQUE_SPLIT_BOXPLOT_PLOT_NAME = 'sequence_statistics_distribution_non_unique_unique_split_'

# plot decorator
def sequence_statistics_distribution_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# boxplot config
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_OUTLIERS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_ORIENTATION = 'h'
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_MEANS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WIDTH = 0.8
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LINE_WIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WHISKERS = 1.5 
# marker
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_MARKER = {'marker':'o',
                                                   'markerfacecolor':'red', 
                                                   'markeredgecolor':'black',
                                                   'markeredgewidth': 1,
                                                   'alpha': 1,
                                                   'markersize': 5,
                                                   'zorder': 30}
# boxplot sort order
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC = BoxplotSortMetric.MEDIAN
# spines
SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_TOP = False
SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_BOTTOM = True
SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_LEFT = True
SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_RIGHT = False
# axes
SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT = True
SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT = False
SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT_RATIO = True
SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT_RATIO = False
SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_RAW = True
SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_RAW = False

########################################################################################################################
### plot_sequence_sequence_count_per_group_per_dataset options ###
########################################################################################################################

# plot name
SEQUENCE_COUNT_PLOT_NAME = 'sequence_count'

# plot decorator
def sequence_count_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# scatterplot config
SEQUENCE_COUNT_KIND = 'scatter'
SEQUENCE_COUNT_PLOT_LEGEND = False
# marker
SEQUENCE_COUNT_MARKER_SIZE = 150
SEQUENCE_COUNT_MARKER_FACECOLOR = 'red'
SEQUENCE_COUNT_MARKER_EDGECOLOR = 'black'
SEQUENCE_COUNT_MARKER_EDGEWIDTH = 1
SEQUENCE_COUNT_MARKER_ALPHA = 0.7
# 45 degree line
SEQUENCE_COUNT_45_DEGREE_LINE_ALPHA = 1
SEQUENCE_COUNT_45_DEGREE_LINE_WIDTH = 2
SEQUENCE_COUNT_45_DEGREE_LINE_COLOR = 'orange'
# spines
SEQUENCE_COUNT_SHOW_TOP = False
SEQUENCE_COUNT_SHOW_BOTTOM = True
SEQUENCE_COUNT_SHOW_LEFT = True
SEQUENCE_COUNT_SHOW_RIGHT = False
# axes
# no sharey because y will alwyas be shared when x is shared to keep square aspect ratio
SEQUENCE_COUNT_SHAREX = False

########################################################################################################################
### html tables options ###
########################################################################################################################

RESULT_AGGREGATION_STYLE_HTML_TABLE = True

########################################################################################################################
### omnibus test result table test options ###
########################################################################################################################

# p-value kind
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND = OmnibusTestResultPValueKind.PVAL_PERM

# p-value correction
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_CORRECT_P_VALUES = True

# p-value correction method
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_CORRECTION_METHOD = PValueCorrectionEnum.FDR_BY

# measure of association type
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY = ContingencyMeasureAssociationEnum.CRAMER_BIAS_CORRECTED
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV = AOVMeasueAssociationEnum.OMEGA_SQUARED

# measure of association strength guideline
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_CONTINGENCY = ContingencyMeasureAssociationStrengthGuidelineEnum.GIGNAC_SZODORAI_2016
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_AOV = AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988

# measure of association strength calculation base (conf int lower - moa value - conf int upper)
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_CALCULATION_BASE = OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_VALUE

RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES = [MeasureAssociationStrengthValuesEnum.VERY_SMALL.value,
                                                              MeasureAssociationStrengthValuesEnum.SMALL.value,
                                                              MeasureAssociationStrengthValuesEnum.MEDIUM.value,
                                                              MeasureAssociationStrengthValuesEnum.LARGE.value]

# result aggregation omnibus test result table field names
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EVAlUATION_FIELD_IS_CATEGORICAL_DISPLAY_FIELD = f'{CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR} Is Categorical'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EVAlUATION_FIELD_TYPE_DISPLAY_FIELD = f'{CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR}'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_DISPLAY_FIELD = f'# {GROUP_FIELD_NAME_STR}s'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD = f'# (%) of {GROUP_FIELD_NAME_STR}s with Significant Differences in {CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR} between {CLUSTER_FIELD_NAME_STR}'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_VERY_SMALL_EFFECT_SIZE_DISPLAY_FIELD = f"# (%) of Significant {GROUP_FIELD_NAME_STR}s with {' '.join(i.capitalize() for i in MeasureAssociationStrengthValuesEnum.VERY_SMALL.value.split('_'))} Effect Size"
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_SMALL_EFFECT_SIZE_DISPLAY_FIELD = f'# (%) of Significant {GROUP_FIELD_NAME_STR}s with {MeasureAssociationStrengthValuesEnum.SMALL.value.capitalize()} Effect Size'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_MEDIUM_EFFECT_SIZE_DISPLAY_FIELD = f'# (%) of Significant {GROUP_FIELD_NAME_STR}s with {MeasureAssociationStrengthValuesEnum.MEDIUM.value.capitalize()} Effect Size'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_LARGE_EFFECT_SIZE_DISPLAY_FIELD = f'# (%) of Significant {GROUP_FIELD_NAME_STR}s with {MeasureAssociationStrengthValuesEnum.LARGE.value.capitalize()} Effect Size'

# table decimal places
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_ROUND_DECIMAL_POINTS = 1