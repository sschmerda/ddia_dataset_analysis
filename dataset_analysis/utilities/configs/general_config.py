from ..standard_import import *

########################################################################################################################
### general ###
########################################################################################################################

# specify whether old pickle files should be deleted when new ones are being written to disk.
# this prevents files which are not being overwritten due to potential name changes to stick around.
DELETE_OLD_PICKLE_FILES = True

########################################################################################################################
### printing elements ###
########################################################################################################################

DASH_STRING = '-' * 100
STAR_STRING = '*' * 100

########################################################################################################################
### parallelization options ###
########################################################################################################################

# specify whether the computations should be done in parallel
PARALLELIZE_COMPUTATIONS = True

# specify the number of cores to be utilized
# all cores available will be utilized if value equals -1

NUMBER_OF_CORES = -1

########################################################################################################################
### randomness option ###
########################################################################################################################

RNG_SEED = 1

########################################################################################################################
### sequence distance libraries ###
########################################################################################################################
# function needs to take 2 strings as the first 2 arguments and return a numeric distance: seq_dist(string_1, string_2)

# from Levenshtein import distance as distance # second fastest implementation in python(mainly written in C)
from polyleven import levenshtein as distance # fastest implementation in python(mainly written in C)

# specify positional and keyword arguments for the distance function(if there are any)
SEQUENCE_DISTANCE_FUNCTION_ARGS = []
SEQUENCE_DISTANCE_FUNCTION_KWARGS = {}

# specify whether the group field(given the datasets has one) should be ignored when calculating sequence distances.
# sequence distances will be calculated over the entire length of a user's learning activities in the interactions dataframe.
SEQUENCE_DISTANCE_IGNORE_GROUPS = False

########################################################################################################################
### pandas options ###
########################################################################################################################

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)

########################################################################################################################
### seaborn options ###
########################################################################################################################

# plot dimensions
SEABORN_FIGURE_LEVEL_ASPECT_WIDE = 1.8
SEABORN_FIGURE_LEVEL_ASPECT_SQUARE = 1
SEABORN_FIGURE_LEVEL_HEIGHT_WIDE_SINGLE = 7
SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_SINGLE = 10
SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET = 3
SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_DISTANCE_MATRIX = 6
SEABORN_FIGURE_LEVEL_HEIGHT_CLUSTER_PARAMETER_TUNING = 18
SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_2_COL= 9

# number of facet grid columns
SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS = 5
SEABORN_SEQUENCE_FILTER_FACET_GRID_2_COL_N_COLUMNS = 2
SEABORN_SEQUENCE_FILTER_FACET_GRID_CLUSTERING_MONITORING_N_COLUMNS = 10 

# rug
SEABORN_RUG_PLOT_HEIGHT_PROPORTION_SINGLE = 0.025
SEABORN_RUG_PLOT_HEIGHT_PROPORTION_FACET = 0.025
SEABORN_RUG_PLOT_HEIGHT_PROPORTION_JOINTPLOT = 0.2
SEABORN_RUG_PLOT_ALPHA_SINGLE = 1
SEABORN_RUG_PLOT_ALPHA_FACET = 0.6
SEABORN_RUG_PLOT_ALPHA_JOINTPLOT = 1
SEABORN_RUG_PLOT_LINEWIDTH_JOINTPLOT = 3
SEABORN_RUG_PLOT_COLOR = 'orange'

# point/scatter
SEABORN_POINT_COLOR_RED = 'red'
SEABORN_POINT_COLOR_BLUE = 'blue'
SEABORN_POINT_COLOR_ORANGE = 'orange'
SEABORN_POINT_SIZE_SINGLE = 5
SEABORN_POINT_SIZE_FACET = 100 
SEABORN_POINT_SIZE_FACET_CLUSTER_2D = 20 
SEABORN_POINT_SIZE_JOINTPLOT = 200
SEABORN_POINT_ALPHA_SINGLE = 0.4
SEABORN_POINT_ALPHA_FACET = 0.4
SEABORN_POINT_ALPHA_FACET_CLUSTER_2D = 1
SEABORN_POINT_ALPHA_JOINTPLOT_PREPROCESSING = 0.4
SEABORN_POINT_ALPHA_JOINTPLOT_SEQ_STAT = 1
SEABORN_POINT_EDGECOLOR = 'black'
SEABORN_POINT_LINEWIDTH = 1

# line
SEABORN_LINE_WIDTH_FACET = 2
SEABORN_LINE_WIDTH_SINGLE = 3
SEABORN_LINE_WIDTH_JOINTPLOT = 3
SEABORN_LINE_COLOR_RED = 'red'
SEABORN_LINE_COLOR_ORANGE = 'orange'

# box
SEABORN_BOX_LINE_WIDTH_SINGLE = 1.5
SEABORN_BOX_LINE_WIDTH_FACET = 1.5

# marker
SEABORN_MARKER_ONE = '1'
SEABORN_MARKER_TWO = '2'
SEABORN_MARKER_COLOR_ORANGE = 'orange'
SEABORN_MARKER_COLOR_RED = 'red'
SEABORN_MARKER_COLOR_GREEN = 'limegreen'
SEABORN_MARKER_SIZE = 80 
SEABORN_MARKER_ALPHA = 1
SEABORN_MARKER_EDGECOLOR_BLACK = 'black'
SEABORN_MARKER_LINEWIDTH = 2

# heatmap
SEABORN_HEATMAP_CMAP = 'magma'
SEABORN_HEATMAP_ANNOTATION_FONTSIZE = 8
SEABORN_HEATMAP_LINEWIDTH = 0.5
SEABORN_HEATMAP_ANNOTATION_COLOR = 'lime'

# main plot element
SEABORN_PLOT_OBJECT_ALPHA = 0.6
SEABORN_DEFAULT_RGB_TUPLE = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)

# plot title
SEABORN_TITLE_FONT_SIZE = 20
SEABORN_SUPTITLE_HEIGHT_CM = 2

# facet grid 
FACET_GRID_SUBPLOTS_H_SPACE = 0.1
FACET_GRID_SUBPLOTS_H_SPACE_SQUARE_WITH_TITLE = 1.5

# color palette type
SEABORN_COLOR_PALETTE = 'husl'

# histogram pin width calculation formula
SEABORN_HISTOGRAM_BIN_CALC_METHOD = 'doane'

# set figsize
FIGURE_LENGTH = 16
FIGURE_HEIGHT = 8

# set dpi
FIGURE_DPI = 100
FIGURE_CLUSTERING_MONITORING_DPI = 50

sns.set_theme(context='notebook',
              style='darkgrid',
              palette='deep',
              rc = {'patch.edgecolor': 'black',
                    'figure.dpi': FIGURE_DPI,
                    'figure.figsize':(FIGURE_LENGTH,
                                      FIGURE_HEIGHT)})

# sns.set_context('notebook', rc={'font.size':20,'axes.titlesize':30,'axes.labelsize':20, 'xtick.labelsize':16, 'ytick.labelsize':16, 'legend.fontsize':20, 'legend.title_fontsize':20})   
marker_config = {'marker':'o',
                 'markerfacecolor':'white', 
                 'markeredgecolor':'black',
                 'markersize': 10,
                 'zorder': 30}
marker_config_eval_metric_mean = {'marker':'o',
                                  'markerfacecolor':'red', 
                                  'markeredgecolor':'black',
                                  'markersize': 6,
                                  'zorder': 30}

########################################################################################################################
### plotly options ###
########################################################################################################################

# parallel coordinates
PLOTLY_PARALLEL_COORDINATES_COLORSCALE = 'Plasma'
PLOTLY_PARALLEL_COORDINATES_FIGURE_WIDTH = 1600
PLOTLY_PARALLEL_COORDINATES_FIGURE_HEIGHT = 800

########################################################################################################################
### numpy options ###
########################################################################################################################

# turn off an unnecessary numpy warning
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

########################################################################################################################
### warnings options ###
########################################################################################################################

warnings.filterwarnings(action='ignore', category=FutureWarning) 