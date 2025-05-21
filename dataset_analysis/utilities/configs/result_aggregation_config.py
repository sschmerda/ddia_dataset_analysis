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
#TODO: maybe make these options plot dependent
RESULT_AGGREGATION_COLOR_ALPHA = 0.75
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

class ClusterSizePlotFields(Enum):
    NUMBER_OF_CLUSTER = CLUSTER_FIELD_NAME_STR
    SEQUENCE_COUNT = RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR

class MeasureAssociationConfIntPlotFields(Enum):
    EFFECT_SIZE = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_VALUE_NAME_STR
    EFFECT_SIZE_CONF_INT_LOWER = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_VALUE_CONF_INT_LOWER_NAME_STR
    EFFECT_SIZE_CONF_INT_UPPER = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_VALUE_CONF_INT_UPPER_NAME_STR

class Axes(Enum):
    X_AXIS = RESULT_AGGREGATION_X_AXIS_NAME_STR
    Y_AXIS = RESULT_AGGREGATION_Y_AXIS_NAME_STR
    BOTH = RESULT_AGGREGATION_BOTH_AXES_NAME_STR

class SortMetric(Enum):
    MEAN = SEQUENCE_STATISTIC_SORT_METRIC_MEAN
    MEDIAN = SEQUENCE_STATISTIC_SORT_METRIC_MEDIAN
    MAX = SEQUENCE_STATISTIC_SORT_METRIC_MAX
    MIN = SEQUENCE_STATISTIC_SORT_METRIC_MIN

class SortingEntity(Enum):
    GROUP = GROUP_FIELD_NAME_STR
    DATASET = DATASET_NAME_FIELD_NAME_STR

class ConfidenceIntervalType(Enum):
    MEAN = 0
    MEDIAN = 1

class RangeIqrConfIntKind(Enum):
    NONE = 0
    BOX = 1
    LINE = 2

class LineCapSizeAdjustment(Enum):
    NONE = 0
    ALL = 1
    THRESHOLD = 2

class ColorPaletteAggregationLevel(Enum):
    GROUP = GROUP_FIELD_NAME_STR
    DATASET = DATASET_NAME_FIELD_NAME_STR

class ClusterSizeScatterPlotKind(Enum):
    STRIPPLOT = 0
    SWARMPLOT = 1

class OmnibusTestResultPValueKind(Enum):
    PVAL = OMNIBUS_TESTS_PVAL_FIELD_NAME_STR
    PVAL_PERM = OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR
    PVAL_R = OMNIBUS_TESTS_R_PVAL_FIELD_NAME_STR
    PVAL_PERM_R = OMNIBUS_TESTS_R_PERM_PVAL_FIELD_NAME_STR

class OmnibusTestResultMeasureAssociationStrengthCalculationBase(Enum):
    MOA_CONF_INT_LOWER_BOUND = 0
    MOA_VALUE = 1
    MOA_CONF_INT_UPPER_BOUND = 2

class OmnibusTestResultMeasureAssociationConfIntXLabelKind(Enum):
    DYNAMIC = 0
    STATIC = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_FIELD_NAME_STR

class OmnibusTestResultMeasureAssociationConfIntGroupInclusion(Enum):
    ALL_GROUPS = 0
    SIGNIFICANT_GROUPS = 1
    NON_SIGNIFICANT_GROUPS = 2

########################################################################################################################
### html tables options ###
########################################################################################################################

RESULT_AGGREGATION_STYLE_HTML_TABLE = True

#***********************************************************************************************************************
#** sequence statistics results ***
#***********************************************************************************************************************

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
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'figure.figsize': (RESULT_AGGREGATION_FIG_SIZE_WIDTH_INCH, RESULT_AGGREGATION_FIG_SIZE_HEIGHT_INCH),
                           'patch.edgecolor': 'black',
                           'axes.labelpad': 30}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1.5, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_ALPHA = RESULT_AGGREGATION_COLOR_ALPHA
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_SATURATION = RESULT_AGGREGATION_COLOR_SATURATION

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
                                                                'markeredgewidth': 1.8,
                                                                'markersize': 10,
                                                                'zorder': 30}
# boxplot sort order
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_BOXES = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_ASCENDING = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_METRIC = SortMetric.MEAN

# swarmplot config
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_SIZE = 10
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_COLOR = 'white'
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_ALPHA = 0.8
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_EDGECOLOR = 'black'
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_LINEWIDTH = 1.8
# axes
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_PCT = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_PCT_RATIO = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_RAW = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_GRID_LINE_AXIS = Axes.X_AXIS
# axes labels
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_X_AXIS_LABEL = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_Y_AXIS_LABEL = False
#not needed, will be generated with suffix and prefix: AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_AXIS_LABEL = None
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_Y_AXIS_LABEL = None
# ticks
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_X_AXIS_TICKS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_Y_AXIS_TICKS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_X_AXIS_TICK_LABELS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_Y_AXIS_TICK_LABELS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_PCT = np.round(np.arange(0, 110, 10), 0)
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_PCT_RATIO = np.round(np.arange(0, 1.1, 0.1), 1)
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_RAW = None

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
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
SUMMARY_SEQUENCE_STATISTICS_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
SUMMARY_SEQUENCE_STATISTICS_COLOR_ALPHA = RESULT_AGGREGATION_COLOR_ALPHA
SUMMARY_SEQUENCE_STATISTICS_COLOR_SATURATION = RESULT_AGGREGATION_COLOR_SATURATION

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
SUMMARY_SEQUENCE_STATISTICS_LINE_WIDTH = 2
# spines
SUMMARY_SEQUENCE_STATISTICS_SHOW_TOP = False
SUMMARY_SEQUENCE_STATISTICS_SHOW_BOTTOM = True
SUMMARY_SEQUENCE_STATISTICS_SHOW_LEFT = False
SUMMARY_SEQUENCE_STATISTICS_SHOW_RIGHT = False
# axes
SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT = False
SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT = True
SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT_RATIO = False
SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT_RATIO = True
SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW = False
SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW = True
SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_PCT = False
SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_PCT_RATIO = False
SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_RAW = True
SUMMARY_SEQUENCE_STATISTICS_GRID_LINE_AXIS = Axes.BOTH
# axes labels
SUMMARY_SEQUENCE_STATISTICS_PLOT_X_AXIS_LABEL = True
SUMMARY_SEQUENCE_STATISTICS_PLOT_Y_AXIS_LABEL = True
SUMMARY_SEQUENCE_STATISTICS_X_AXIS_LABEL = None
SUMMARY_SEQUENCE_STATISTICS_Y_AXIS_LABEL = None
# ticks
SUMMARY_SEQUENCE_STATISTICS_PLOT_X_AXIS_TICKS = True
SUMMARY_SEQUENCE_STATISTICS_PLOT_Y_AXIS_TICKS = True
SUMMARY_SEQUENCE_STATISTICS_PLOT_X_AXIS_TICK_LABELS = True
SUMMARY_SEQUENCE_STATISTICS_PLOT_Y_AXIS_TICK_LABELS = True
SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_PCT = np.round(np.arange(0, 110, 10), 0)
SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_PCT_RATIO = np.round(np.arange(0, 1.1, 0.1), 1)
SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_RAW = None
# facet grid remove inner plot elements
SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_X_AXIS_LABELS = True
SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_Y_AXIS_LABELS = True
SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_X_AXIS_TICKS = False
SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_Y_AXIS_TICKS = False
SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_X_AXIS_TICK_LABELS = False
SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

########################################################################################################################
### plot_sequence_statistics_distribution_boxplot_per_group_per_dataset options ###
########################################################################################################################

# plot name
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_NAME = 'sequence_statistics_distribution_boxplot_'
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_NON_UNIQUE_UNIQUE_SPLIT_PLOT_NAME = 'sequence_statistics_distribution_boxplot_non_unique_unique_split_'

# plot decorator
def sequence_statistics_distribution_boxplot_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_ALPHA = RESULT_AGGREGATION_COLOR_ALPHA
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_SATURATION = RESULT_AGGREGATION_COLOR_SATURATION

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
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_OUTLIER_MARKER = {'marker':'o',
                                                           'markerfacecolor':'white', 
                                                           'markeredgecolor':'black',
                                                           'markeredgewidth': 1,
                                                           'alpha': 0.75,
                                                           'markersize': 5,
                                                           'zorder': 30}

# boxplot sort order
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_ASCENDING = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC = SortMetric.MEAN
# spines
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_TOP = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_BOTTOM = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_LEFT = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_RIGHT = False
# axes
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_PCT = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_PCT = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_PCT_RATIO = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_PCT_RATIO = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_RAW = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_RAW = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_PCT = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_PCT_RATIO = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_RAW = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_GRID_LINE_AXIS = Axes.X_AXIS
# axes labels
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_X_AXIS_LABEL = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_Y_AXIS_LABEL = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_AXIS_LABEL = None
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_Y_AXIS_LABEL = None
# ticks
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_X_AXIS_TICKS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_Y_AXIS_TICKS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_X_AXIS_TICK_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_Y_AXIS_TICK_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_PCT = np.round(np.arange(0, 110, 10), 0)
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_PCT_RATIO = np.round(np.arange(0, 1.1, 0.1), 1)
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_RAW = None
# facet grid remove inner plot elements
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_X_AXIS_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_Y_AXIS_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_X_AXIS_TICKS = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_Y_AXIS_TICKS = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS = False
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

########################################################################################################################
### plot_sequence_statistics_distribution_ridgeplot_per_group_per_dataset options ###
########################################################################################################################

# plot name
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_NAME = 'sequence_statistics_distribution_ridgeplot_'

# plot decorator
def sequence_statistics_distribution_ridgeplot_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_COLOR_ALPHA = 0.4
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_COLOR_SATURATION = 1

# ridgeplot config
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT = 0.75
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BANDWIDTH_METHOD = 'scott' # scott, silverman, 0.3
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BANDWIDTH_CUT = 3
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_APPLY_BOUNDARY_REFLECTION = False # does not work properly

# ridgeplot sort order
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_BOXES = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_ASCENDING = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_METRIC = SortMetric.MEAN

# ridgeplot lines
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_COLOR = 'white'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_LINEWIDTH = 5
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_ALPHA = 0.6
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_COLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_LINEWIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_ALPHA = 0.8
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INCLUDE_KDE_BOTTOM_LINE = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_COLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_LINEWIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_ALPHA = 0.8

# ridgeplot range, interquartile range and conf int kind
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_RANGE_CONF_INT_KIND = RangeIqrConfIntKind.BOX
# confidence interval
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONFIDENCE_INTERVAL_LEVEL = 0.95
# ridgeplot range, interquartile range and conf int line adjust cap size of line
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_RANGE_CONF_INT_LINE_CAP_ADJUST = LineCapSizeAdjustment.NONE

# ridgeplot range and interquartile range line
# iqr
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_OUTER_LINEPLOT_COLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_OUTER_LINEPLOT_LINEWIDTH = 16
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_INNER_LINEPLOT_COLOR = '0.9'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_INNER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_OUTER_LINEPLOT_LINEWIDTH - 2
# range
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_OUTER_LINEPLOT_COLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_OUTER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_OUTER_LINEPLOT_LINEWIDTH / 4
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_INNER_LINEPLOT_COLOR = '0.9'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_INNER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_OUTER_LINEPLOT_LINEWIDTH - 2

# ridgeplot conf int line
# mean
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_OUTER_LINEPLOT_COLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_OUTER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_OUTER_LINEPLOT_LINEWIDTH / 2
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_INNER_LINEPLOT_COLOR = 'red'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_INNER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_OUTER_LINEPLOT_LINEWIDTH - 2
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_SCATTER_COLOR = 'white'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_SCATTER_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_SCATTER_LINEWIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_SCATTER_SIZE = 10
# median
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_OUTER_LINEPLOT_COLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_OUTER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_OUTER_LINEPLOT_LINEWIDTH / 2
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_INNER_LINEPLOT_COLOR = 'cornflowerblue'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_INNER_LINEPLOT_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_OUTER_LINEPLOT_LINEWIDTH - 2
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_SCATTER_COLOR = 'white'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_SCATTER_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_SCATTER_LINEWIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_SCATTER_SIZE = 10

# ridgeplot range and interquartile range box
# iqr
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGE_LINEWIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_HEIGHT_IN_LINEWIDTH = 16
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_FACECOLOR = '0.9'
# range
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_EDGE_LINEWIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_HEIGHT_IN_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_HEIGHT_IN_LINEWIDTH / 6
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_FACECOLOR = '0.9'

# ridgeplot conf int box
# mean
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_EDGE_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGE_LINEWIDTH
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_FACECOLOR = 'red'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_SCATTER_COLOR = 'white'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_SCATTER_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_SCATTER_LINEWIDTH = 0.6
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEAN_BOX_SCATTER_SIZE = 8
# median
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_EDGE_LINEWIDTH = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGE_LINEWIDTH
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_FACECOLOR = 'cornflowerblue'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_SCATTER_COLOR = 'white'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_SCATTER_EDGECOLOR = 'black'
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_SCATTER_LINEWIDTH = 0.6
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_MEDIAN_BOX_SCATTER_SIZE = 8

# spines
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_TOP = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_BOTTOM = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_LEFT = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_RIGHT = False
# axes
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_PCT = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_PCT = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_PCT_RATIO = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_PCT_RATIO = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_RAW = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_RAW = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_PCT = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_PCT_RATIO = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_RAW = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_GRID_LINE_AXIS = Axes.X_AXIS
# axes data limits
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_PCT = (0, 100)
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_PCT_RATIO = (0, 1)
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_RAW = (0, np.inf)

# axes labels
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_X_AXIS_LABEL = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_Y_AXIS_LABEL = True
# SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_AXIS_LABEL = None
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_LABEL = GROUP_FIELD_NAME_STR
# ticks
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_X_AXIS_TICKS = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_Y_AXIS_TICKS = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_X_AXIS_TICK_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_Y_AXIS_TICK_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_PCT = np.round(np.arange(0, 110, 10), 0)
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_PCT_RATIO = np.round(np.arange(0, 1.1, 0.1), 1)
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_RAW = None
# facet grid remove inner plot elements
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_X_AXIS_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_Y_AXIS_LABELS = True
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_X_AXIS_TICKS = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_Y_AXIS_TICKS = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS = False
SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

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
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
SEQUENCE_COUNT_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
SEQUENCE_COUNT_COLOR_ALPHA = RESULT_AGGREGATION_COLOR_ALPHA
SEQUENCE_COUNT_COLOR_SATURATION = RESULT_AGGREGATION_COLOR_SATURATION

# scatterplot config
SEQUENCE_COUNT_KIND = 'scatter'
SEQUENCE_COUNT_PLOT_LEGEND = False
# marker
SEQUENCE_COUNT_MARKER_SIZE = 150
SEQUENCE_COUNT_MARKER_FACECOLOR = 'red'
SEQUENCE_COUNT_MARKER_EDGECOLOR = 'black'
SEQUENCE_COUNT_MARKER_EDGEWIDTH = 1
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
# no sharey because y will always be shared when x is shared to keep square aspect ratio
SEQUENCE_COUNT_SHAREX = True
SEQUENCE_COUNT_LOG_SCALE_X_RAW = True
SEQUENCE_COUNT_GRID_LINE_AXIS = Axes.BOTH
# axes labels
SEQUENCE_COUNT_PLOT_X_AXIS_LABEL = True
SEQUENCE_COUNT_PLOT_Y_AXIS_LABEL = True
SEQUENCE_COUNT_X_AXIS_LABEL = None
SEQUENCE_COUNT_Y_AXIS_LABEL = None
# ticks
SEQUENCE_COUNT_PLOT_X_AXIS_TICKS = True
SEQUENCE_COUNT_PLOT_Y_AXIS_TICKS = True
SEQUENCE_COUNT_PLOT_X_AXIS_TICK_LABELS = True
SEQUENCE_COUNT_PLOT_Y_AXIS_TICK_LABELS = True
SEQUENCE_COUNT_X_TICKS_RAW = None
SEQUENCE_COUNT_Y_TICKS_RAW = None
# facet grid remove inner plot elements
SEQUENCE_COUNT_REMOVE_INNER_X_AXIS_LABELS = True
SEQUENCE_COUNT_REMOVE_INNER_Y_AXIS_LABELS = True
SEQUENCE_COUNT_REMOVE_INNER_X_AXIS_TICKS = False
SEQUENCE_COUNT_REMOVE_INNER_Y_AXIS_TICKS = False
SEQUENCE_COUNT_REMOVE_INNER_X_AXIS_TICK_LABELS = False
SEQUENCE_COUNT_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

#***********************************************************************************************************************
#** omnibus test results ***
#***********************************************************************************************************************

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

# moa strength values
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES = [MeasureAssociationStrengthValuesEnum.VERY_SMALL.value,
                                                              MeasureAssociationStrengthValuesEnum.SMALL.value,
                                                              MeasureAssociationStrengthValuesEnum.MEDIUM.value,
                                                              MeasureAssociationStrengthValuesEnum.LARGE.value]

# table 
# specifies which of [moa_conf_int_lower_bound, moa_value, moa_conf_int_upper_bound] will be shown in result table
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_TABLE_VALUES = [
                                                                    # OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_CONF_INT_LOWER_BOUND,
                                                                    OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_VALUE,
                                                                    # OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_CONF_INT_UPPER_BOUND
                                                                    ]
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_ROUND_DECIMAL_POINTS = 1
# result aggregation omnibus test result table field names
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EVAlUATION_FIELD_IS_CATEGORICAL_DISPLAY_FIELD = f'{CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR} is Categorical'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EVAlUATION_FIELD_TYPE_DISPLAY_FIELD = f'{CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR}'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_DISPLAY_FIELD = f'# {GROUP_FIELD_NAME_STR}s'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD = f'# (%) Significant {GROUP_FIELD_NAME_STR}s'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_VERY_SMALL_EFFECT_SIZE_DISPLAY_FIELD = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD + f" w/ {' '.join(i.capitalize() for i in MeasureAssociationStrengthValuesEnum.VERY_SMALL.value.split('_'))} Effect Size"
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_SMALL_EFFECT_SIZE_DISPLAY_FIELD = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD + f' w/ {MeasureAssociationStrengthValuesEnum.SMALL.value.capitalize()} Effect Size'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_MEDIUM_EFFECT_SIZE_DISPLAY_FIELD = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD + f' w/ {MeasureAssociationStrengthValuesEnum.MEDIUM.value.capitalize()} Effect Size'
RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_LARGE_EFFECT_SIZE_DISPLAY_FIELD = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD + f' w/ {MeasureAssociationStrengthValuesEnum.LARGE.value.capitalize()} Effect Size'
# RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_VERY_SMALL_EFFECT_SIZE_DISPLAY_FIELD = f"# (%) of Significant {GROUP_FIELD_NAME_STR}s with {' '.join(i.capitalize() for i in MeasureAssociationStrengthValuesEnum.VERY_SMALL.value.split('_'))} Effect Size"
# RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_SMALL_EFFECT_SIZE_DISPLAY_FIELD = f'# (%) of Significant {GROUP_FIELD_NAME_STR}s with {MeasureAssociationStrengthValuesEnum.SMALL.value.capitalize()} Effect Size'
# RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_MEDIUM_EFFECT_SIZE_DISPLAY_FIELD = f'# (%) of Significant {GROUP_FIELD_NAME_STR}s with {MeasureAssociationStrengthValuesEnum.MEDIUM.value.capitalize()} Effect Size'
# RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_LARGE_EFFECT_SIZE_DISPLAY_FIELD = f'# (%) of Significant {GROUP_FIELD_NAME_STR}s with {MeasureAssociationStrengthValuesEnum.LARGE.value.capitalize()} Effect Size'

########################################################################################################################
### plot_cluster_size_per_group_per_dataset_categorical_scatter options ###
########################################################################################################################

# plot name
CLUSTER_SIZE_SCATTERPLOT_NAME = 'cluster_size_scatterplot'

# plot decorator
def cluster_size_per_group_per_dataset_categorical_scatter_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
CLUSTER_SIZE_SCATTERPLOT_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
CLUSTER_SIZE_SCATTERPLOT_COLOR_ALPHA = 0.5
CLUSTER_SIZE_SCATTERPLOT_COLOR_SATURATION = 1

# scatter config
CLUSTER_SIZE_SCATTERPLOT_KIND = ClusterSizeScatterPlotKind.SWARMPLOT
CLUSTER_SIZE_SCATTERPLOT_MARKER_KIND = 'o'
CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE = 8
CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR = 'black'
CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGEWIDTH = 0.8
CLUSTER_SIZE_SCATTERPLOT_MARKER_JITTER = 0.2
CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_KIND = 'X'
CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_COLOR = (1, 1, 1, 0.5)

# scatter sort order
CLUSTER_SIZE_SCATTERPLOT_SORT_BOXES = True
CLUSTER_SIZE_SCATTERPLOT_SORT_ASCENDING = True
CLUSTER_SIZE_SCATTERPLOT_SORT_METRIC = SortMetric.MAX
CLUSTER_SIZE_SCATTERPLOT_SORT_FIELD = ClusterSizePlotFields.NUMBER_OF_CLUSTER

# legend config
CLUSTER_SIZE_SCATTERPLOT_PLOT_LEGEND = True

# spines
CLUSTER_SIZE_SCATTERPLOT_SHOW_TOP = False
CLUSTER_SIZE_SCATTERPLOT_SHOW_BOTTOM = True
CLUSTER_SIZE_SCATTERPLOT_SHOW_LEFT = False
CLUSTER_SIZE_SCATTERPLOT_SHOW_RIGHT = False
# axes
CLUSTER_SIZE_SCATTERPLOT_SHAREX_RAW = False
CLUSTER_SIZE_SCATTERPLOT_SHAREY_RAW = False
CLUSTER_SIZE_SCATTERPLOT_LOG_SCALE_X_RAW = False
CLUSTER_SIZE_SCATTERPLOT_GRID_LINE_AXIS = Axes.BOTH

# axes labels
CLUSTER_SIZE_SCATTERPLOT_PLOT_X_AXIS_LABEL = True
CLUSTER_SIZE_SCATTERPLOT_PLOT_Y_AXIS_LABEL = True
CLUSTER_SIZE_SCATTERPLOT_X_AXIS_LABEL = None
CLUSTER_SIZE_SCATTERPLOT_Y_AXIS_LABEL = GROUP_FIELD_NAME_STR
# ticks
CLUSTER_SIZE_SCATTERPLOT_PLOT_X_AXIS_TICKS = True
CLUSTER_SIZE_SCATTERPLOT_PLOT_Y_AXIS_TICKS = True
CLUSTER_SIZE_SCATTERPLOT_PLOT_X_AXIS_TICK_LABELS = True
CLUSTER_SIZE_SCATTERPLOT_PLOT_Y_AXIS_TICK_LABELS = True
# facet grid remove inner plot elements
CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_X_AXIS_LABELS = True
CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_Y_AXIS_LABELS = True
CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_X_AXIS_TICKS = False
CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_Y_AXIS_TICKS = False
CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS = False
CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

########################################################################################################################
### plot_cluster_size_per_group_per_dataset_lineplot options ###
########################################################################################################################

# plot name
CLUSTER_SIZE_LINEPLOT_NAME = 'cluster_size_lineplot'

# plot decorator
def cluster_size_per_group_per_dataset_lineplot_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# colors
CLUSTER_SIZE_LINEPLOT_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
CLUSTER_SIZE_LINEPLOT_COLOR_ALPHA = RESULT_AGGREGATION_COLOR_ALPHA
CLUSTER_SIZE_LINEPLOT_COLOR_SATURATION = RESULT_AGGREGATION_COLOR_SATURATION

# line line config
CLUSTER_SIZE_LINEPLOT_LINE_WIDTH = 1.6

# line marker config
CLUSTER_SIZE_LINEPLOT_MARKER_KIND = 'o'
CLUSTER_SIZE_LINEPLOT_MARKER_SIZE = 18
CLUSTER_SIZE_LINEPLOT_MARKER_EDGECOLOR = 'black'
CLUSTER_SIZE_LINEPLOT_MARKER_EDGEWIDTH = 1
CLUSTER_SIZE_LINEPLOT_MARKER_ALPHA = 1

# legend config
CLUSTER_SIZE_LINEPLOT_PLOT_LEGEND = False

# spines
CLUSTER_SIZE_LINEPLOT_SHOW_TOP = False
CLUSTER_SIZE_LINEPLOT_SHOW_BOTTOM = True
CLUSTER_SIZE_LINEPLOT_SHOW_LEFT = False
CLUSTER_SIZE_LINEPLOT_SHOW_RIGHT = False
# axes
CLUSTER_SIZE_LINEPLOT_SHAREX_RAW = False
CLUSTER_SIZE_LINEPLOT_SHAREY_RAW = False
CLUSTER_SIZE_LINEPLOT_LOG_SCALE_X_RAW = False
CLUSTER_SIZE_LINEPLOT_GRID_LINE_AXIS = Axes.BOTH

# axes labels
CLUSTER_SIZE_LINEPLOT_PLOT_X_AXIS_LABEL = True
CLUSTER_SIZE_LINEPLOT_PLOT_Y_AXIS_LABEL = True
CLUSTER_SIZE_LINEPLOT_X_AXIS_LABEL = None
CLUSTER_SIZE_LINEPLOT_Y_AXIS_LABEL = None
# ticks
CLUSTER_SIZE_LINEPLOT_PLOT_X_AXIS_TICKS = True
CLUSTER_SIZE_LINEPLOT_PLOT_Y_AXIS_TICKS = True
CLUSTER_SIZE_LINEPLOT_PLOT_X_AXIS_TICK_LABELS = True
CLUSTER_SIZE_LINEPLOT_PLOT_Y_AXIS_TICK_LABELS = True
# facet grid remove inner plot elements
CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_X_AXIS_LABELS = True
CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_Y_AXIS_LABELS = True
CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_X_AXIS_TICKS = False
CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_Y_AXIS_TICKS = False
CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS = False
CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

########################################################################################################################
### plot_aggregated_omnibus_test_result_per_dataset_stacked_barplot options ###
########################################################################################################################

# plot name
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_NAME = 'omnibus_test_result_stacked_barplot'

# plot decorator
def aggregated_omnibus_test_result_per_dataset_stacked_barplot_decorator(func):
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
                           'patch.edgecolor': 'black',
                           'axes.labelpad': 30}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1.5, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# stacked barplot moa strength color palette
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE = 'magma'
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_DESAT = 0.8
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_ALPHA = 0.6
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_NON_SIG_COLOR = 'lightgrey'
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_NUMBER_COLORS = 1000
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_COLOR_INDEX_MIN = 300
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_COLOR_INDEX_MAX = 999

# legend config
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_LEGEND = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_COLOR = 'black'
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_LINEWIDTH = 1.5
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_LENGTH = 4
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_WIDTH = 2

# axes labels
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_X_AXIS_LABEL = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_Y_AXIS_LABEL = False
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_AXIS_LABEL = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_PCT_OF_GROUPS_NAME_STR
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_Y_AXIS_LABEL = ''
# ticks
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_TICKS_ROTATION = 0 
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_X_AXIS_TICKS = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_Y_AXIS_TICKS = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_X_AXIS_TICK_LABELS = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_Y_AXIS_TICK_LABELS = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_TICKS = np.arange(0, 110, 10)

# barplot config
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_EDGECOLOR = 'black'
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_LINEWIDTH = 2
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_WIDTH = 0.8

# grid
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_GRID_LINES_VERTICAL = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_GRID_LINES_HORIZONTAL = False

# annotation text
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_THRESHOLD = 10
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_FONTSIZE = 15
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_COLOR = 'black'
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_V_POS = 'center'
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_H_POS = 'center'

# strings
# group categories str
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_MOA_LABELS_ADD_EFFECT_SIZE_STR = True
OMNIBUS_TEST_RESULT_STACKED_BARPLOT_MOA_LABELS_CAPITALIZE = False

########################################################################################################################
### plot_aggregated_omnibus_test_result_per_dataset_grouped_barplot options ###
########################################################################################################################

# plot name
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_NAME = 'omnibus_test_result_grouped_barplot'

# plot decorator
def aggregated_omnibus_test_result_per_dataset_grouped_barplot_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('white', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# grouped barplot config
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ORIENTATION = 'v'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_WIDTH = 0.8
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_EDGECOLOR = 'black'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_EDGEWIDTH = 2

# grouped barplot moa strength color palette
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE = 'magma'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_DESAT = 0.8
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_ALPHA = 0.6
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_NON_SIG_COLOR = 'lightgrey'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_NUMBER_COLORS = 1000
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_COLOR_INDEX_MIN = 300
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_COLOR_INDEX_MAX = 999

# legend config
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_LEGEND = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_COLOR = 'black'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_LINEWIDTH = 1.5
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_LENGTH = 4
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_WIDTH = 2

# spines
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_TOP = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_BOTTOM = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_LEFT = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_RIGHT = False

# axes
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHAREX = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHAREY = True
# axes labels
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_X_AXIS_LABEL = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_Y_AXIS_LABEL = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_X_AXIS_LABEL = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EFFECT_SIZE_STRENGTH_NAME_STR
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_Y_AXIS_LABEL = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_PCT_OF_GROUPS_NAME_STR
# ticks
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_X_TICKS_ROTATION = 0 
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_X_AXIS_TICKS = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_Y_AXIS_TICKS = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_X_AXIS_TICK_LABELS = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_Y_AXIS_TICK_LABELS = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_Y_TICKS = np.arange(0, 110, 10)

# annotation text
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_THRESHOLD = 10
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_FONTSIZE = 15
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_COLOR = 'black'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_V_POS = 'center'
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_H_POS = 'center'

# facet grid remove inner plot elements
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_X_AXIS_LABELS = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_Y_AXIS_LABELS = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_X_AXIS_TICKS = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_Y_AXIS_TICKS = True
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

# strings
# group categories str
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_MOA_LABELS_ADD_EFFECT_SIZE_STR = False
OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_MOA_LABELS_CAPITALIZE = False

########################################################################################################################
### plot_omnibus_test_result_moa_confidence_interval_per_group_per_dataset options ###
########################################################################################################################

# plot name
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_NAME = 'omnibus_test_result_measure_association_confidence_interval'

# plot decorator
def omnibus_test_result_moa_confidence_interval_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI,
                           'axes.labelpad': 20}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# group inclusion
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_GROUP_INCLUSION = OmnibusTestResultMeasureAssociationConfIntGroupInclusion.SIGNIFICANT_GROUPS

# effect size field
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_EFFECT_SIZE_VALUE_FIELD = MeasureAssociationConfIntPlotFields.EFFECT_SIZE
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_EFFECT_SIZE_VALUE_LOWER_BOUND_FIELD = MeasureAssociationConfIntPlotFields.EFFECT_SIZE_CONF_INT_LOWER
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_EFFECT_SIZE_VALUE_UPPER_BOUND_FIELD = MeasureAssociationConfIntPlotFields.EFFECT_SIZE_CONF_INT_UPPER

# colors
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_COLOR_PALETTE = RESULT_AGGREGATION_COLOR_PALETTE
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_COLOR_ALPHA = 1
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_COLOR_SATURATION = 0.75

# moa strength color palette
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE = 'magma'
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_DESAT = 0.8
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_ALPHA = 0.6
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_NON_SIG_COLOR = None
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_NUMBER_COLORS = 1000
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_COLOR_INDEX_MIN = 300
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_COLOR_INDEX_MAX = 999

# legend config
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_LEGEND = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_COLOR = 'black'
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_LINEWIDTH = 1.5
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_LENGTH = 4
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_WIDTH = 2

# scatter config
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SCATTER_MARKER_SIZE = 100
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SCATTER_MARKER_EDGECOLOR = 'black'
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SCATTER_MARKER_EDGEWIDTH = 1.5

# errorbar config
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_CAP_SIZE = 8
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_CAP_THICKNESS = 1.5
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_BAR_COLOR = 'black'
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_BAR_LINEWIDTH = 1.5

# scatter/errorbar sort order
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_CONFIDENCE_INTERVALS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_ASCENDING = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_METRIC = SortMetric.MAX
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_SECOND_SORT_FIELD = MeasureAssociationConfIntPlotFields.EFFECT_SIZE_CONF_INT_UPPER

# spines
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_TOP = False
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_BOTTOM = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_LEFT = False
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_RIGHT = False
# axes
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_Y_AXIS_MARGIN = 0.1
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHAREX_PCT_RATIO = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHAREY_PCT_RATIO = False
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_GRID_LINE_AXIS = Axes.X_AXIS
# axes labels
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_X_AXIS_LABEL = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_Y_AXIS_LABEL = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_X_AXIS_LABEL = OmnibusTestResultMeasureAssociationConfIntXLabelKind.DYNAMIC # set x_labe dynamically or statically
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_Y_AXIS_LABEL = GROUP_FIELD_NAME_STR
# ticks
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_X_AXIS_TICKS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_Y_AXIS_TICKS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_X_AXIS_TICK_LABELS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_Y_AXIS_TICK_LABELS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_X_TICKS_PCT_RATIO = np.round(np.arange(0, 1.1, 0.1), 1)
# facet grid remove inner plot elements
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_X_AXIS_LABELS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_Y_AXIS_LABELS = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_X_AXIS_TICKS = False
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_Y_AXIS_TICKS = False
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_X_AXIS_TICK_LABELS = False
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_Y_AXIS_TICK_LABELS = False

# strings
# group categories str
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_MOA_LABELS_ADD_EFFECT_SIZE_STR = True
OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_MOA_LABELS_CAPITALIZE = False