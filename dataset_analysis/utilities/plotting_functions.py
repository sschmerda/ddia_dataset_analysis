from .standard_import import *
from .constants import *
from .config import *

def return_axis_limits(data: ArrayLike,
                       data_is_pct: bool,
                       pct_is_ratio: bool = False,
                       pct: int = 5) -> tuple[float]:
    """Returns a tuple with lower and upper axis limits for given input data

    Parameters
    ----------
    data : ArrayLike
        An array of numeric data
    data_is_pct : bool
        A flag indicating whether data values are percentages. If true, the axis limits will be set to (-5, 105)
    pct_is_ratio : bool, optional
        A flag indicating whether the percentage is bounded by the interval [0, 1]. Otherwise an interval of
        [0, 100] is being used. The parameter only has an effect if data_is_pct == True.
    pct : int, optional
        A pct of the max value in the data which defines upper and lower axis limits, by default 5

    Returns
    -------
    tuple[float]
        A tuple of axis limits
    """
    if data_is_pct:
        if pct_is_ratio:
            axis_upper_limit = 1.05
            axis_lower_limit = -0.05
        else:
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

def set_facet_grid_column_number(data: ArrayLike,
                                 preset_number_of_columns: int) -> int:
    """Determines the number of columns for the facet grid plot. It will be the minimum of preset_number_of_columns
    and the total number of subplots possible given the unique number of values in data.

    Parameters
    ----------
    data : ArrayLike
        An array containing values for grouping
    preset_number_of_columns : int
        The preset number of columns of the facet grid

    Returns
    -------
    int
        The number of columns for the facet grid plot
    """    
    n_subplots = len(np.unique(data))

    n_columns = min(preset_number_of_columns, n_subplots)

    return n_columns

def add_central_tendency_marker_per_facet(g: FacetGrid,
                                          sequence_stats_per_group_df: pd.DataFrame,
                                          col_field: str,
                                          x_var: str,
                                          y_var: str) -> FacetGrid:
    """Add a legend for central tendency markers to the input facet grid object

    Parameters
    ----------
    g : FacetGrid
        A seaborn figure level facet grid object
    sequence_stats_per_group_df : pd.DataFrame
        A dataframe containing statistics of learning_activity sequences over user entities grouped by group entities
    col_field : str
        A string indicating the variable used for faceting
    x_var : str
        The variable to be potted on the x axis
    y_var : str
        The variable to be potted on the y axis

    Returns
    -------
    FacetGrid
        The seaborn facet grid object with added legend
    """

    central_tendencies = sequence_stats_per_group_df.groupby(col_field)[[x_var, y_var]].agg([np.mean, np.median])

    for ax, group in zip(g.axes.flat, central_tendencies.index):

        x_mean = central_tendencies.loc[group][x_var]['mean']
        y_mean = central_tendencies.loc[group][y_var]['mean']

        x_median = central_tendencies.loc[group][x_var]['median']
        y_median = central_tendencies.loc[group][y_var]['median']
        
        marker_style_1 = MarkerStyle(marker=SEABORN_MARKER_ONE, fillstyle='none')
        marker_style_2 = MarkerStyle(marker=SEABORN_MARKER_TWO, fillstyle='none')
        sns.scatterplot(x=[x_mean], 
                        y=[y_mean], 
                        s=SEABORN_MARKER_SIZE,
                        linewidth=SEABORN_MARKER_LINEWIDTH,
                        alpha=SEABORN_MARKER_ALPHA,
                        color=SEABORN_MARKER_COLOR_RED,
                        label=LEARNING_ACTIVITY_SEQUENCE_MEAN_NAME_STR,
                        marker=marker_style_1,
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(x=[x_median], 
                        y=[y_median], 
                        s=SEABORN_MARKER_SIZE,
                        linewidth=SEABORN_MARKER_LINEWIDTH,
                        alpha=SEABORN_MARKER_ALPHA,
                        color=SEABORN_MARKER_COLOR_GREEN,
                        label=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        marker=marker_style_2,
                        legend=False,
                        zorder=100,
                        ax=ax)

    handles = [plt.scatter([], 
                           [], 
                           marker=marker_style_1, 
                           color=SEABORN_MARKER_COLOR_RED, 
                           linewidth=SEABORN_MARKER_LINEWIDTH, 
                           alpha=SEABORN_MARKER_ALPHA),
               plt.scatter([], 
                           [], 
                           marker=marker_style_2, 
                           color=SEABORN_MARKER_COLOR_GREEN, 
                           linewidth=SEABORN_MARKER_LINEWIDTH, 
                           alpha=SEABORN_MARKER_ALPHA)]

    labels = [LEARNING_ACTIVITY_SEQUENCE_MEAN_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR]

    g.add_legend(title=None, 
                 label_order=labels, 
                 legend_data={label: handle for label, handle in zip(labels, handles)}, 
                 frameon=True)

    return g

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
                    line_kws={'linewidth': SEABORN_LINE_WIDTH_SINGLE,
                              'alpha': SEABORN_PLOT_OBJECT_ALPHA},
                    alpha=SEABORN_PLOT_OBJECT_ALPHA)
    for ax in g.axes.flat:
        ax.lines[0].set_color(SEABORN_LINE_COLOR_RED)

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

def plot_stat_plot(sequence_stats_per_group_df: pd.DataFrame,
                   aggregated_statistics_df_long: pd.DataFrame,
                   statistic: str,
                   stat_is_pct: bool,
                   pct_is_ratio: bool,
                   boxplot_title: str,
                   pointplot_title: str,
                   ecdfplot_title: str) -> None:
    """Plot a boxplot, pointplot and ecdfplot of a sequence statistic over sequences per group. The plots are split by the
    use of 1. all sequences and 2. only unique sequences within a group.

    Parameters
    ----------
    sequence_stats_per_group_df : pd.DataFrame
        A dataframe containing statistics of learning_activity sequences over user entities grouped by group entities
    aggregated_statistics_df_long : pd.DataFrame
        A dataframe in long format containing an aggregated statistic of learning_activity sequences
    statistic : str
        The sequence statistic to be plotted
    stat_is_pct : bool
        A flag indicating whether the sequence statistic is a percentage
    pct_is_ratio : bool
        A flag indicating whether the percentage is expressed in the interval [0, 1]
    boxplot_title : str
        The title for the respective boxplot
    pointplot_title : str
        The title for the respective pointplot
    ecdfplot_title : str
        The title for the respective ecdfplot
    """
    axis_lim = return_axis_limits(sequence_stats_per_group_df[statistic],
                                  stat_is_pct,
                                  pct_is_ratio=pct_is_ratio)
    
    n_cols = set_facet_grid_column_number(sequence_stats_per_group_df[LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR],
                                          SEABORN_SEQUENCE_FILTER_FACET_GRID_2_COL_N_COLUMNS)

    n_groups = sequence_stats_per_group_df[GROUP_FIELD_NAME_STR].nunique()

    # boxplot
    g=sns.catplot(sequence_stats_per_group_df,
                  x=statistic,
                  y=GROUP_FIELD_NAME_STR,
                  col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                  col_wrap=n_cols, 
                  kind='box',
                  orient='h',
                  showmeans=True,
                  meanprops=marker_config,
                  palette=return_color_palette(n_groups),
                  height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_2_COL,
                  aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                  facet_kws=dict(sharex=True,
                                 sharey=True,
                                 xlim=axis_lim))
    # Create a custom legend
    plot_legend(sequence_stats_per_group_df[GROUP_FIELD_NAME_STR])
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.tight_layout()
    y_loc = calculate_suptitle_position(g,
                                        SEABORN_SUPTITLE_HEIGHT_CM)
    g.figure.suptitle(boxplot_title, 
                      fontsize=SEABORN_TITLE_FONT_SIZE,
                      y=y_loc)
    plt.show(g);

    # pointplot
    g=sns.catplot(aggregated_statistics_df_long,
                  x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                  y=statistic,
                  hue=GROUP_FIELD_NAME_STR,
                  col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                  col_wrap=n_cols, 
                  kind='point',
                  palette=return_color_palette(n_groups),
                  height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_2_COL,
                  aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                  legend=False,
                  facet_kws=dict(sharex=True,
                                 sharey=True,
                                 ylim=axis_lim))
    # Create a custom legend
    plot_legend(sequence_stats_per_group_df[GROUP_FIELD_NAME_STR])
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.tight_layout()
    y_loc = calculate_suptitle_position(g,
                                        SEABORN_SUPTITLE_HEIGHT_CM)
    g.figure.suptitle(pointplot_title, 
                      fontsize=SEABORN_TITLE_FONT_SIZE,
                      y=y_loc)
    plt.show(g);

    # ecdf plot
    g=sns.displot(sequence_stats_per_group_df,
                  x=statistic,
                  hue=GROUP_FIELD_NAME_STR,
                  col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                  col_wrap=n_cols, 
                  kind='ecdf',
                  palette=return_color_palette(n_groups),
                  height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_2_COL,
                  aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                  legend=False,
                  facet_kws=dict(sharex=True,
                                 sharey=True,
                                 xlim=axis_lim))
    # Create a custom legend
    plot_legend(sequence_stats_per_group_df[GROUP_FIELD_NAME_STR])
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.tight_layout()
    y_loc = calculate_suptitle_position(g,
                                        SEABORN_SUPTITLE_HEIGHT_CM)
    g.figure.suptitle(ecdfplot_title, 
                      fontsize=SEABORN_TITLE_FONT_SIZE,
                      y=y_loc)
    plt.show(g);

def plot_stat_scatter_plot_per_group(sequence_stats_per_group_df: pd.DataFrame,
                                     statistic_x: str,
                                     statistic_y: str,
                                     statistic_x_label: str,
                                     statistic_y_label: str,
                                     stat_x_is_pct: bool,
                                     stat_y_is_pct: bool,
                                     stat_x_pct_is_ratio: bool,
                                     stat_y_pct_is_ratio: bool,
                                     title: str,
                                     share_x: bool,
                                     share_y: bool) -> None:
    """Plot a scatterplot of a sequence statistic against the respective sequence frequency of a unique sequence
    per group.

    Parameters
    ----------
    sequence_stats_per_group_df : pd.DataFrame
        A dataframe containing statistics of learning_activity sequences over user entities grouped by group entities
    statistic_x : str
        The sequence statistic to be plotted on the x axis
    statistic_y : str
        The sequence statistic to be plotted on the y axis
    statistic_x_label : str
        The label used for the respective statistic on the x axis
    statistic_y_label : str
        The label used for the respective statistic on the y axis
    stat_x_is_pct : bool
        A flag indicating whether the sequence statistic on the x axis is a percentage
    stat_y_is_pct : bool
        A flag indicating whether the sequence statistic on the y axis is a percentage
    stat_x_pct_is_ratio : bool
        A flag indicating whether the percentage on the x axis is expressed in the interval [0, 1]
    stat_y_pct_is_ratio : bool
        A flag indicating whether the percentage on the y axis is expressed in the interval [0, 1]
    title : str
        The title of the plot
    share_x : bool
        A flag indicating whether the x axis should be shared across subplot
    share_y : bool
        A flag indicating whether the y axis should be shared across subplot
    """
    xlim = return_axis_limits(sequence_stats_per_group_df[statistic_x],
                              stat_x_is_pct,
                              stat_x_pct_is_ratio)  
    ylim = return_axis_limits(sequence_stats_per_group_df[statistic_y],
                              stat_y_is_pct,
                              stat_y_pct_is_ratio)
    
    n_cols = set_facet_grid_column_number(sequence_stats_per_group_df[GROUP_FIELD_NAME_STR],
                                          SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

    # relative sequence frequency %
    g=sns.relplot(sequence_stats_per_group_df,
                  x=statistic_x,
                  y=statistic_y,
                  col=GROUP_FIELD_NAME_STR,
                  col_wrap=n_cols,
                  kind='scatter',
                  height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                  aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                  s=SEABORN_POINT_SIZE_FACET,
                  alpha=SEABORN_POINT_ALPHA_FACET,
                  edgecolor=SEABORN_POINT_EDGECOLOR,
                  linewidth=SEABORN_POINT_LINEWIDTH,
                  facet_kws=dict(sharex=share_x,
                                  sharey=share_y,
                                  xlim=xlim,
                                  ylim=ylim))
    g.map_dataframe(sns.regplot,
                    x=statistic_x,
                    y=statistic_y,
                    scatter=False,
                    robust=True,
                    ci=None,
                    line_kws=dict(color=SEABORN_MARKER_COLOR_ORANGE,
                                  linewidth=SEABORN_LINE_WIDTH_FACET))
    g.set(xlabel=statistic_x_label,
          ylabel=statistic_y_label)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.tight_layout()
    g = add_central_tendency_marker_per_facet(g,
                                              sequence_stats_per_group_df,
                                              GROUP_FIELD_NAME_STR,
                                              statistic_x,
                                              statistic_y)
    y_loc = calculate_suptitle_position(g,
                                        SEABORN_SUPTITLE_HEIGHT_CM)
    g.figure.suptitle(title, 
                        fontsize=SEABORN_TITLE_FONT_SIZE,
                        y=y_loc)
    plt.show(g);

def plot_stat_hist_plot_per_group(sequence_stats_per_group_df: pd.DataFrame,
                                  statistic: str,
                                  stat_is_pct: bool,
                                  pct_is_ratio: bool,
                                  statistic_label: str,
                                  title: str,
                                  share_x: bool,
                                  share_y: bool) -> None:
    """Plot a histogram of a sequence statistic per group. The plots are split by the
    use of 1. all sequences and 2. only unique sequences within a group.

    Parameters
    ----------
    sequence_stats_per_group_df : pd.DataFrame
        A dataframe containing the merged unique learning activity sequence stats and all learning activity sequence
        stats dataframes 
    statistic : str
        The sequence statistic to be plotted
    stat_is_pct : bool
        A flag indicating whether the sequence statistic is a percentage
    pct_is_ratio : bool
        A flag indicating whether the percentage is expressed in the interval [0, 1]
    statistic_label : str
        The label used for the respective statistic
    title : str
        The plot title
    share_x : bool
        A flag indicating whether the x axis should be shared across subplot
    share_y : bool
        A flag indicating whether the y axis should be shared across subplot
    """
    xlim_plot = return_axis_limits(sequence_stats_per_group_df[statistic],
                                   stat_is_pct,
                                   pct_is_ratio)

    title = LEARNING_ACTIVITY_SEQUENCE_HISTOGRAM_TITLE_NAME_STR + '\n' + statistic
    g = sns.FacetGrid(sequence_stats_per_group_df,
                      col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                      row=GROUP_FIELD_NAME_STR, 
                      sharex=share_x, 
                      sharey=share_y,
                      height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET_2_COL,
                      aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                      xlim=xlim_plot)
    g.map_dataframe(sns.histplot, 
                    x=statistic, 
                    stat='count',
                    kde=True,
                    line_kws={'linewidth': SEABORN_LINE_WIDTH_FACET,
                              'alpha': SEABORN_PLOT_OBJECT_ALPHA},
                    bins=SEABORN_HISTOGRAM_BIN_CALC_METHOD,
                    alpha=SEABORN_PLOT_OBJECT_ALPHA)
    for ax in g.axes.flat:
        ax.lines[0].set_color(SEABORN_LINE_COLOR_RED)
    g.map_dataframe(sns.rugplot, 
                    x=statistic,
                    height=SEABORN_RUG_PLOT_HEIGHT_PROPORTION_FACET,
                    color=SEABORN_RUG_PLOT_COLOR,
                    expand_margins=True,
                    alpha=SEABORN_RUG_PLOT_ALPHA_FACET)
    g.set(xlabel=statistic_label) 
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.tight_layout()
    y_loc = calculate_suptitle_position(g,
                                        SEABORN_SUPTITLE_HEIGHT_CM)
    g.figure.suptitle(title, 
                      fontsize=SEABORN_TITLE_FONT_SIZE,
                      y=y_loc)
    plt.subplots_adjust(hspace=FACET_GRID_SUBPLOTS_H_SPACE)
    plt.show(g)