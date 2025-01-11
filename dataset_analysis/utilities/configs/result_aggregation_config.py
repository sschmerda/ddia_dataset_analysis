from .general_config import *
from ..constants.constants import *
from ..standard_import import *
from ..plotting_functions import *

########################################################################################################################
### option enums ###
########################################################################################################################

class SequenceStatisticsPlotFields(Enum):
    SEQUENCE_LENGTH = LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR
    PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ = LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_NAME_STR
    PCT_REPEATED_LEARNING_ACTIVITIES = LEARNING_ACTIVITY_SEQUENCE_PCT_REPEATED_LEARNING_ACTIVITIES_NAME_STR
    MEAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    SEQUENCE_TYPE = LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    SEQUENCE_TYPE = LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    SEQUENCE_TYPE = LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    MEDIAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR
    SEQUENCE_TYPE = LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR

class SequenceStatisticsDistributionBoxplotSortMetric(Enum):
    MEAN = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC_MEAN
    MEDIAN = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC_MEDIAN
    MAX = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC_MAX
    MIN = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC_MIN

########################################################################################################################
### figure save options ###
########################################################################################################################

SAVE_FIGURE_DPI = 400
SAVE_FIGURE_IMAGE_FORMAT = 'png'
SAVE_FIGURE_BBOX_INCHES = 'tight'

########################################################################################################################
### sequence statistics options ###
########################################################################################################################

# fields to plot
SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST = [SequenceStatisticsPlotFields.SEQUENCE_LENGTH,
                                           SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ,
                                           SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES,
                                           SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE]

########################################################################################################################
### figure plot options ###
########################################################################################################################

RESULT_AGGREGATION_FIG_SIZE_WIDTH_INCH = 16
RESULT_AGGREGATION_FIG_SIZE_HEIGHT_INCH = 8
RESULT_AGGREGATION_FIG_SIZE_DPI = 300


########################################################################################################################
### facet grid options ###
########################################################################################################################

RESULT_AGGREGATION_FACET_GRID_N_COLUMNS = 3
# facet grid config
RESULT_AGGREGATION_FACET_GRID_ASPECT = 1
RESULT_AGGREGATION_FACET_GRID_HEIGHT = RESULT_AGGREGATION_FIG_SIZE_WIDTH_INCH/(RESULT_AGGREGATION_FACET_GRID_N_COLUMNS * RESULT_AGGREGATION_FACET_GRID_ASPECT)

########################################################################################################################
### plot_avg_sequence_statistics_per_group_per_dataset options ###
########################################################################################################################

# plot name
AVG_SEQUENCE_STATISTICS_PLOT_NAME = 'average_sequence_statistics_'

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
# color saturation
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_SATURATION = 0.75

             return func(*args, **kwargs)
    return wrapper

# boxplot color palette
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE = SEABORN_COLOR_PALETTE
# AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE ='vlag'

# boxplot config
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_OUTLIERS = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_MEANS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WIDTH = 0.8
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_LINE_WIDTH = 2
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WHISKERS = 1.5 
# color saturation
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_SATURATION = 0.75
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

# lineplot color palette
SUMMARY_SEQUENCE_STATISTICS_COLOR_PALETTE = SEABORN_COLOR_PALETTE 
# SUMMARY_SEQUENCE_STATISTICS_COLOR_PALETTE = ['black'] 

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

# plot decorator
def sequence_statistics_distribution_per_group_per_dataset_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': RESULT_AGGREGATION_FIG_SIZE_DPI}
# color saturation
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_SATURATION = 0.75

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

# boxplot color palette
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_PALETTE = SEABORN_COLOR_PALETTE

# boxplot config
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_OUTLIERS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_ORIENTATION = 'h'
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_MEANS = True
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WIDTH = 0.8
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LINE_WIDTH = 1
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WHISKERS = 1.5 
# color saturation
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_SATURATION = 0.75
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
SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC = SequenceStatisticsDistributionBoxplotSortMetric.MEDIAN
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