from ..standard_import import *
from .general_config import *

########################################################################################################################
### figure plot options ###
########################################################################################################################

PAIRPLOT_FIG_SIZE_DPI = 100
PAIRPLOT_COLOR_PALETTE = SEABORN_COLOR_PALETTE

########################################################################################################################
### plot_pairplot options ###
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