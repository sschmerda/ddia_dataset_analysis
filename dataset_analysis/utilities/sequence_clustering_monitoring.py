from .standard_import import *
from .constants import *
from .config import *
from .io_functions import *
from .plotting_functions import *

def plot_distance_matrix_field_value_distribution(path_within_pickle_directory_list: list[str],
                                                  filename: str,
                                                  groups: Union[list[int], None],
                                                  sample_size_pct: Union[int, None],
                                                  normalize_distance: bool,
                                                  use_unique_sequence_distances: bool) -> None:
    """Plot the value distribution of the fields in the distance matrix. (No single value should dominate in order for
    umap to work properly)


    Parameters
    ----------
    path_within_pickle_directory_list : str
        A list of path elements pointing to the subfolder of the pickle directory containing the sequence distance analytics objects for the respective dataset 
    filename : str
        The name given to the serialized sequence distance analytics object for the respective dataset
    groups : Union[list[int], None]
        The groups for which distance matrix field value distributions will be plotted. If None all groups will be included
    sample_size_pct : Union[int, None]
        If not None a subset of the fields in the distance matrix of size sample_size_pct * n_cols will be used for plotting
    normalize_distance : bool
        A boolean indicating whether the sequence distances are being normalized between 0 and 1
    use_unique_sequence_distances: bool
        A boolean indicating whether only unique sequences are being used as the basis for distance calculations
    """
    sequence_distance_analytics = pickle_read(path_within_pickle_directory_list,
                                              filename)

    if normalize_distance:
        distance_type = LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR 
    else:
        distance_type = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR 
    
    if use_unique_sequence_distances:
        field_type = SEQUENCE_ID_FIELD_NAME_STR
    else:
        field_type = USER_FIELD_NAME_STR

    if not groups:
        groups = sequence_distance_analytics.unique_learning_activity_sequence_stats_per_group[GROUP_FIELD_NAME_STR].unique()

    for group in groups:
        distance_matrix = (sequence_distance_analytics.return_sequence_distance_matrix_per_group(group, 
                                                                                                 normalize_distance, 
                                                                                                 use_unique_sequence_distances)
                                                                                                 [SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_FIELD_NAME_STR])
        if sample_size_pct:
            sample_size = round(distance_matrix.shape[1] * sample_size_pct / 100)
            fields = sorted(random.sample(tuple(distance_matrix.columns), sample_size))
            distance_matrix = distance_matrix.loc[:, fields]
        
        distance_matrix_long = pd.melt(distance_matrix,
                                       var_name=field_type,
                                       value_name=distance_type)

        share_x_axis = True
        if normalize_distance:
            axis_xlim = return_axis_limits(distance_matrix_long[distance_type],
                                           True,
                                           True)
        else:
            axis_xlim = return_axis_limits(distance_matrix_long[distance_type],
                                           False,
                                           False)
        
        n_cols = set_facet_grid_column_number(distance_matrix_long[field_type],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_CLUSTERING_MONITORING_N_COLUMNS)

        print('')
        print(STAR_STRING)
        print(STAR_STRING)
        print('')
        print(f'Sequence Distance Matrix for {GROUP_FIELD_NAME_STR} {group}:')
        print(f'Distribution of values per field')
        print('')
        print(f'Field Type: {field_type}')
        print(f'Field Value: {distance_type}')
        print('')
        with plt.rc_context({'figure.dpi': FIGURE_CLUSTERING_MONITORING_DPI}):
            g = sns.FacetGrid(distance_matrix_long,
                              col=field_type,
                              col_wrap=n_cols,
                              sharex=share_x_axis, 
                              sharey=False,
                              height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                              aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                              xlim=axis_xlim)

            g.map_dataframe(sns.histplot, 
                            x=distance_type, 
                            stat='count',
                            kde=True,
                            line_kws={'linewidth': SEABORN_LINE_WIDTH_FACET,
                                      'alpha': SEABORN_PLOT_OBJECT_ALPHA},
                            bins=SEABORN_HISTOGRAM_BIN_CALC_METHOD,
                            alpha=SEABORN_PLOT_OBJECT_ALPHA)
            for ax in g.axes.flat:
                ax.lines[0].set_color(SEABORN_LINE_COLOR_RED)
            g.map_dataframe(sns.rugplot, 
                            x=distance_type,
                            height=SEABORN_RUG_PLOT_HEIGHT_PROPORTION_FACET,
                            color=SEABORN_RUG_PLOT_COLOR,
                            expand_margins=True,
                            alpha=SEABORN_RUG_PLOT_ALPHA_FACET)
            g.set(xlabel=distance_type)
            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            plt.tight_layout()
            plt.show(g)