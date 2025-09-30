from ..standard_import import *
from ..constants import *
from ..constants.enums import *
from .general_config import *

########################################################################################################################
### pairplot option enums ###
########################################################################################################################

# relates sequence statistic to 
def return_plot_field_data_kind(plot_field: PairplotFieldsToPlot) -> Tuple[bool, bool]:
    match plot_field:
        case PairplotFieldsToPlot.PCT_REPEATED_LEARNING_ACTIVITIES:
            return (True, False)
        case PairplotFieldsToPlot.NUMBER_REPEATED_LEARNING_ACTIVITIES:
            return (False, False)
        case PairplotFieldsToPlot.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP:
            return (True, False)
        case PairplotFieldsToPlot.NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP:
            return (False, False)
        case PairplotFieldsToPlot.SEQUENCE_LENGTH:
            return (False, False)
        case PairplotFieldsToPlot.MEAN_SEQUENCE_DISTANCE:
            return (False, False)
        case PairplotFieldsToPlot.MEAN_NORMALIZED_SEQUENCE_DISTANCE:
            return (True, True)
        case _:
            raise ValueError(PAIRPLOT_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{plot_field}')

########################################################################################################################
###  fields to plot options ###
########################################################################################################################

PAIRPLOT_FIELDS_TO_PLOT_LIST = [PairplotFieldsToPlot.PCT_REPEATED_LEARNING_ACTIVITIES,
                                PairplotFieldsToPlot.NUMBER_REPEATED_LEARNING_ACTIVITIES,
                                PairplotFieldsToPlot.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP,
                                PairplotFieldsToPlot.NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP,
                                PairplotFieldsToPlot.SEQUENCE_LENGTH,
                                PairplotFieldsToPlot.MEAN_SEQUENCE_DISTANCE,
                                PairplotFieldsToPlot.MEAN_NORMALIZED_SEQUENCE_DISTANCE]

########################################################################################################################
###  pairplot plotting options ###
########################################################################################################################

PAIRPLOT_GROUP_FILTER = None
PAIRPLOT_CLUSTER_FILTER = None
PAIRPLOT_VARIABLE_SCALER = None #PairplotVariableScaler.ROBUST_SCALER
PAIRPLOT_VARIABLE_SCALER_BASE = PairplotGroupingVariable.GROUP
PAIRPLOT_ADD_LEGEND = True
PAIRPLOT_ADD_HEADER = True
PAIRPLOT_EXCLUDE_NON_CLUSTERED = False
PAIRPLOT_SET_AXIS_LIM_FOR_DTYPE = True 
PAIRPLOT_ADD_CENTRAL_TENDENCY_MARKERS = True
PAIRPLOT_ADD_CENTRAL_TENDENCY_MARKERS_PER_GROUPING_VARIABLE = True
PAIRPLOT_ADD_REGRESSION_LINE = True
PAIRPLOT_ADD_REGRESSION_LINE_PER_GROUPING_VARIABLE = True
PAIRPLOT_CONTINUOUS_EVAL_METRIC_ADD_SLOPE_IN_HEADER = True
PAIRPLOT_CATEGORICAL_EVAL_METRIC_ADD_SLOPE_IN_HEADER = False

########################################################################################################################
### facet grid sizing options ###
########################################################################################################################

PAIRPLOT_FACET_GRID_ASPECT = 1
PAIRPLOT_FACET_GRID_HEIGHT = 3
PAIRPLOT_FIG_SIZE_DPI = 100

########################################################################################################################
### style options ###
########################################################################################################################

def pairplot_decorator(func):
    """A decorator to apply temporary Seaborn settings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        axes_style_rc_dict = {'axes.grid': True,
                              'font.family': ['Arial']}
        rc_context_dict = {'figure.dpi': PAIRPLOT_FIG_SIZE_DPI}

        with sns.axes_style('ticks', rc=axes_style_rc_dict),\
             sns.plotting_context(context='notebook', font_scale=1, rc={}),\
             plt.rc_context(rc_context_dict):

             return func(*args, **kwargs)
    return wrapper

########################################################################################################################
### plot elements options ###
########################################################################################################################

PAIRPLOT_COLOR_PALETTE = SEABORN_COLOR_PALETTE

PAIRPLOT_SCATTER_POINT_SIZE = 25
PAIRPLOT_SCATTER_POINT_ALPHA = 0.8
PAIRPLOT_SCATTER_POINT_EDGECOLOR = 'black'

PAIRPLOT_STRIPPLOT_POINT_SIZE = 5
PAIRPLOT_STRIPPLOT_POINT_ALPHA = 0.8
PAIRPLOT_STRIPPLOT_POINT_LINEWIDTH = 0.35
PAIRPLOT_STRIPPLOT_POINT_EDGECOLOR = 'black'

PAIRPLOT_CENTRAL_TENDENCY_MARKER_SIZE_OUTER = 150
PAIRPLOT_CENTRAL_TENDENCY_MARKER_LINEWIDTH_OUTER = 1.8
PAIRPLOT_CENTRAL_TENDENCY_MARKER_EDGECOLOR_OUTER = 'white'
PAIRPLOT_CENTRAL_TENDENCY_MARKER_SIZE_INNER = 75
PAIRPLOT_CENTRAL_TENDENCY_MARKER_LINEWIDTH_INNER = 1.4
PAIRPLOT_CENTRAL_TENDENCY_MARKER_EDGECOLOR_INNER = 'black'
PAIRPLOT_CENTRAL_TENDENCY_MARKER_ALPHA = 1
PAIRPLOT_CENTRAL_TENDENCY_MARKER_COLOR = 'red'
PAIRPLOT_CENTRAL_TENDENCY_MARKER_KIND = '*'

PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_SIZE_OUTER = 40
PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_LINEWIDTH_OUTER = 1.8
PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_EDGECOLOR_OUTER = 'white'
PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_SIZE_INNER = 20
PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_LINEWIDTH_INNER = 1.4 
PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_EDGECOLOR_INNER = 'black'
PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_KIND = 'D'

PAIRPLOT_REGRESSION_LINE_NUMBER_OF_POINTS = 1000

PAIRPLOT_REGRESSION_LINE_LINEWIDTH = 2
PAIRPLOT_REGRESSION_LINE_LINE_STYLE = '-'
PAIRPLOT_REGRESSION_LINE_COLOR = 'red'
PAIRPLOT_REGRESSION_LINE_PROPORTION_AXIS_LIM = 0.03

PAIRPLOT_REGRESSION_LINE_PER_GROUPING_VARIABLE_LINEWIDTH = 1.5
PAIRPLOT_REGRESSION_LINE_PER_GROUPING_VARIABLE_LINE_STYLE = '--'
PAIRPLOT_REGRESSION_LINE_PER_GROUPING_VARIABLE_PROPORTION_AXIS_LIM = 0.03

########################################################################################################################
### subplot plot axes options ###
########################################################################################################################

PAIRPLOT_ADJUST_X_LABEL = True
PAIRPLOT_X_LABEL_ROTATION = 0
PAIRPLOT_X_LABEL_VERTICAL_PAD = 30
PAIRPLOT_X_LABEL_SPLIT_STRING = ' '
PAIRPLOT_X_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT = 3

PAIRPLOT_ADJUST_Y_LABEL = True
PAIRPLOT_Y_LABEL_ROTATION = 0
PAIRPLOT_Y_LABEL_RIGHT_PAD = 30
PAIRPLOT_Y_LABEL_SPLIT_STRING = ' '
PAIRPLOT_Y_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT = 3

########################################################################################################################
### subplot header options ###
########################################################################################################################

PAIRPLOT_HEADER_FONTSIZE = 15
PAIRPLOT_PEARSON_CORRELATION_AND_PARAMS_ROUND_DIGITS = 3
PAIRPLOT_P_VALUE_ROUND_DIGITS = 3

########################################################################################################################
### misc options ###
########################################################################################################################

PAIRPLOT_X_AXIS_OFFSET_IF_NO_X_VARIANCE = 0.000001