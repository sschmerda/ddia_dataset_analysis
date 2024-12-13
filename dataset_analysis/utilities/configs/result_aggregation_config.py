from .general_config import *
from ..constants.constants import *
from ..standard_import import *

########################################################################################################################
### option enums ###
########################################################################################################################

class AvgSequenceStatsPerGroupPerDatasetsPlotFields(Enum):
    SEQUENCE_LENGTH = LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR
    PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ = LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_NAME_STR
    PCT_REPEATED_LEARNING_ACTIVITIES = LEARNING_ACTIVITY_SEQUENCE_PCT_REPEATED_LEARNING_ACTIVITIES_NAME_STR
    MEAN_NORMALIZED_SEQUENCE_DISTANCE = LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR

########################################################################################################################
### figure save options ###
########################################################################################################################

SAVE_FIGURE_DPI = 400
SAVE_FIGURE_IMAGE_FORMAT = 'png'
SAVE_FIGURE_BBOX_INCHES = 'tight'

########################################################################################################################
### plot_avg_sequence_statistics_per_group_per_dataset options ###
########################################################################################################################

# fields to plot
AVG_SEQUENCE_STATS_PER_GROUP_PER_DATASE_FIELDS_TO_PLOT_LIST = [AvgSequenceStatsPerGroupPerDatasetsPlotFields.SEQUENCE_LENGTH,
                                                               AvgSequenceStatsPerGroupPerDatasetsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ,
                                                               AvgSequenceStatsPerGroupPerDatasetsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES,
                                                               AvgSequenceStatsPerGroupPerDatasetsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE]

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
        rc_context_dict = {'figure.dpi': 300,
                           'figure.figsize': (16,
                                              8),
                          'patch.edgecolor': 'black'}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='paper', font_scale=1.5, rc={}),\
             plt.rc_context(rc_context_dict):

            return func(*args, **kwargs)
    return wrapper

# boxplot config
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_OUTLIERS = False
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_MEANS = True
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WIDTH = 0.8
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WHISKERS = 1.5 
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_MARKER = {'marker':'o',
                                                                'markerfacecolor':'white', 
                                                                'markeredgecolor':'black',
                                                                'markersize': 10,
                                                                'zorder': 30}
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE='vlag'

# swarmplot config
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_SIZE = 8
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_COLOR = 'black'
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_ALPHA = 0.6
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_EDGECOLOR = 'gray'
AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_LINEWIDTH = 1