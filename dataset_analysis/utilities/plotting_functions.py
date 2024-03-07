from .standard_import import *
from .constants import *
from .config import *

def return_axis_limits(data: ArrayLike,
                       data_is_pct: bool,
                       pct: int = 5) -> tuple[float]:
    """Returns a tuple with lower and upper axis limits for given input data

    Parameters
    ----------
    data : ArrayLike
        An array of numeric data
    data_is_pct : bool
        A flag indicating whether data values are percentages. If true, the axis limits will be set to (-5, 105)
    pct : int, optional
        A pct of the max value in the data which defines upper and lower axis limits, by default 5

    Returns
    -------
    tuple[float]
        A tuple of axis limits
    """
    if data_is_pct:
        axis_upper_limit = 105
        axis_lower_limit = -5
    else:
        max_val = max(data)
        axis_upper_limit = float(max_val * (1 + pct / 100))
        axis_lower_limit = float((axis_upper_limit - max_val) * -1)

    return axis_lower_limit, axis_upper_limit

def return_color_palette(n_colors: int,
                         color_palette_name: str = SEABORN_COLOR_PALETTE,
                         **kwargs) -> List[tuple]:
    """Return a seaborn color palette

    Parameters
    ----------
    n_colors : int
        The number of colors to be calculated
    color_palette_name : str, optional
        A seaborn color palette name, by default SEABORN_COLOR_PALETTE

    Returns
    -------
    List[tuple]
        A seaborn color palette
    """
    col_palette = sns.color_palette(color_palette_name, 
                                    n_colors=n_colors,
                                    **kwargs)
    return col_palette

def plot_legend(data: ArrayLike) -> None:
    """Plots a legend for the unique values in data

    Parameters
    ----------
    data : ArrayLike
        An array of values for which a legend is being plotted

    Returns
    -------
    None
    """
    legend_labels = np.unique(data)
    n_labels = len(legend_labels)
    legend_handles = [plt.scatter([], [], color=color, lw=2, label=label) for color, label in zip(return_color_palette(n_colors=n_labels), legend_labels)]
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

def calculate_suptitle_position(g: FacetGrid,
                                cm_top_offset: Union[int, float]) -> float:
    """Calculate and return the suptitle positon for a facet grid object

    Parameters
    ----------
    g : FacetGrid
        A seaborn figure level facet grid object
    cm_top_offset : Union[int, float]
        The offset between the suptitle and the top row of the facet grid  

    Returns
    -------
    float
        The y location of the suptitle in figure coordinates
    """
    height_inches = g.figure.get_figheight()
    height_cm = height_inches * 2.54
    y_loc = (height_cm + cm_top_offset) / height_cm

    return y_loc

def set_facet_grid_column_number(data: pd.DataFrame,
                                 facet_var: str,
                                 preset_number_of_columns: int) -> int:
    """Determines the number of columns for the facet grid plot. It will be the minimum of preset_number_of_columns
    and the total number of subplots possible given the unique number of values in the facet_var field of data.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe used for plotting the facet grid
    facet_var : str
        The name of the variable whose values determine the number of subplots  
    preset_number_of_columns : int
        The preset number of columns of the facet grid

    Returns
    -------
    int
        The number of columns for the facet grid plot
    """    
    n_subplots = data[facet_var].nunique()

    n_columns = min(preset_number_of_columns, n_subplots)

    return n_columns

def plot_distribution(data: pd.DataFrame,
                      x_var: str,
                      label: str,
                      log_scale: bool):
    """Plot the distribution of a variable via boxplot-stripplot, kernel-density and histogram.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing the variable for which the distribution plots are being plotted
    x_var : str
        The dataframe field name of the variable for which the distribution plots are being plotted
    label : str
        The axis label of the variable for which the distribution plots are being plotted
    log_scale : bool
        A boolean indicating whether logarithmized axis should be applied
    pointsize : float
        The pointsize of the stripplot
    """
    # box and stripplot
    g=sns.catplot(data,
                  x=x_var,
                  kind='box',
                  height=SEABORN_FIGURE_LEVEL_HEIGHT_WIDE_SINGLE,
                  aspect=SEABORN_FIGURE_LEVEL_ASPECT_WIDE,
                  showmeans=True,
                  meanprops=marker_config)

    g.map_dataframe(sns.stripplot,
                    x=x_var,
                    s=SEABORN_POINT_SIZE_SINGLE, 
                    color=SEABORN_POINT_COLOR_RED,
                    alpha=SEABORN_POINT_ALPHA_SINGLE,
                    edgecolor=SEABORN_POINT_EDGECOLOR,
                    linewidth=SEABORN_POINT_LINEWIDTH)
    g.set(xlabel=label)
    axes = g.axes.flat
    for ax in axes:
        for patch in ax.patches:
            r, g, b, _ = patch.get_facecolor()
            if (r, g, b) != SEABORN_DEFAULT_RGB_TUPLE:
                patch.set_facecolor((*SEABORN_DEFAULT_RGB_TUPLE, SEABORN_PLOT_OBJECT_ALPHA))
    if log_scale:
            plt.xscale('log')
    plt.show(g);

    # histogram
    g = sns.displot(data=data, 
                    x=x_var, 
                    kind='hist',
                    stat='count',
                    height=SEABORN_FIGURE_LEVEL_HEIGHT_WIDE_SINGLE,
                    aspect=SEABORN_FIGURE_LEVEL_ASPECT_WIDE,
                    bins=SEABORN_HISTOGRAM_BIN_CALC_METHOD, 
                    kde=True,
                    alpha=SEABORN_PLOT_OBJECT_ALPHA)

    axes = g.axes.flat
    for ax in axes:
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
    g = sns.rugplot(data=data, 
                    x=x_var, 
                    color=SEABORN_RUG_PLOT_COLOR,
                    height=SEABORN_RUG_PLOT_HEIGHT_PROPORTION_SINGLE,
                    alpha=SEABORN_RUG_PLOT_ALPHA_SINGLE)
    g.set(xlabel=label)
    if log_scale:
            plt.xscale('log')
    plt.show(g);