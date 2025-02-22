from .configs.general_config import *
from .configs.sequence_pairplot_config import *
from .constants.constants import *
from .standard_import import *
from .plotting_functions import *
from .result_tables import ResultTables
from .data_classes import *
from .validators import *

class LearningActivitySequencePairplot():
    """A class used for plotting group-wise pairplots of relevant variables"""

    def __init__(self, 
                 result_tables: ResultTables,
                 exclude_non_clustered: bool,
                 set_axis_lim_for_dtype: bool,
                 evaluation_metric_field: str,
                 evaluation_metric_is_categorical: bool,
                 evaluation_metric_is_pct: bool,
                 evaluation_metric_pct_is_ratio: bool) -> None:

        self.interactions: pd.DataFrame = result_tables.interactions.copy()
        self.learning_activity_sequence_stats_per_group: pd.DataFrame = result_tables.learning_activity_sequence_stats_per_group.copy()
        self.exclude_non_clustered: bool = exclude_non_clustered
        self.set_axis_lim_for_dtype: bool = set_axis_lim_for_dtype
        self.evaluation_metric_field: str = evaluation_metric_field
        self.evaluation_metric_is_categorical: bool = evaluation_metric_is_categorical
        self.evaluation_metric_is_pct: bool = evaluation_metric_is_pct
        self.evaluation_metric_pct_is_ratio: bool = evaluation_metric_pct_is_ratio

        self._fields = [GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR, SEQUENCE_ID_FIELD_NAME_STR]
        self._fields_to_plot = self._return_fields_to_plot(PAIRPLOT_FIELDS_TO_PLOT_LIST) + [self.evaluation_metric_field]
        self._n_plot_fields = len(self._fields_to_plot)
        self._fields_to_plot_axis_lim_cat = self._return_fields_to_plot_data_kind(PAIRPLOT_FIELDS_TO_PLOT_LIST) + [(self.evaluation_metric_is_pct, self.evaluation_metric_pct_is_ratio)]
        self._fields_to_keep = self._fields + self._fields_to_plot

        self._merge_fields = [GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, self.evaluation_metric_field]

        self.seq_stats_data: pd.DataFrame = self._generate_plotting_df(self.exclude_non_clustered)

    def return_sequence_stats_data(self,
                                   groups_filter: Iterable[int] | None,
                                   clusters_filter: Iterable[int] | None) -> pd.DataFrame:

        seq_stats_df = self._filter_seq_stats_df(self.seq_stats_data,
                                                 groups_filter,
                                                 clusters_filter)

        return seq_stats_df

    def return_variable_relationship(self,
                                     group: int,
                                     cluster: int | None,
                                     groups_filter: Iterable[int] | None,
                                     clusters_filter: Iterable[int] | None,
                                     x_var: str,
                                     y_var: str) -> PairplotData:

        seq_stats_df = self._filter_seq_stats_df(self.seq_stats_data,
                                                 groups_filter,
                                                 clusters_filter)

        variable_relationship_dict = self._return_variable_relationships_per_group_per_cluster(seq_stats_df,
                                                                                               self._fields_to_plot)

        return self._return_variable_relationship(variable_relationship_dict,
                                                  group,
                                                  cluster,
                                                  x_var,
                                                  y_var)
    
    @pairplot_decorator
    def plot_pairplot_seq_vars_per_cluster_for_each_group(self,
                                                          groups_filter: Iterable[int] | None,
                                                          clusters_filter: Iterable[int] | None,
                                                          add_legend: bool,
                                                          add_header: bool,
                                                          add_central_tendency_marker: bool,
                                                          add_central_tendency_marker_per_grouping_variable: bool,
                                                          add_regression_line: bool,
                                                          add_regression_line_per_grouping_variable: bool) -> None:

        print(DASH_STRING)
        print(PAIRPLOT_PER_GROUP_PER_CLUSTER_TITLE_STR)
        print(DASH_STRING)
        print('')

        color_palette_all_clust, grouping_variable_index_mapping = self._return_color_palette_and_index_mapping_df(self.seq_stats_data,
                                                                                                                   PairplotGroupingVariable.CLUSTER)

        seq_stats_df = self._filter_seq_stats_df(self.seq_stats_data,
                                                 groups_filter,
                                                 clusters_filter)
        variable_relationship_dict = self._return_variable_relationships_per_group_per_cluster(seq_stats_df,
                                                                                               self._fields_to_plot)

        for group, df in seq_stats_df.groupby(GROUP_FIELD_NAME_STR):

            print(STAR_STRING)
            print(f'{GROUP_FIELD_NAME_STR}: {group}')

            clusters = sorted(df[PairplotGroupingVariable.CLUSTER.value].unique())
            color_palette = [color_palette_all_clust[grouping_variable_index_mapping[i]] for i in clusters]

            central_tendencies_per_cluster_df = self._calculate_central_tendencies(df,
                                                                                   PairplotGroupingVariable.CLUSTER)
            central_tendencies_df = self._calculate_central_tendencies(df,
                                                                       None)
            #TODO: function for this
            if self.evaluation_metric_is_categorical:
                central_tendencies_per_cluster_df_categorical = self._calculate_central_tendencies_categorical_eval_metric(df,
                                                                                                                           PairplotGroupingVariable.CLUSTER)
                central_tendencies_df_categorical = self._calculate_central_tendencies_categorical_eval_metric(df,
                                                                                                               None)
            else:
                central_tendencies_per_cluster_df_categorical = None
                central_tendencies_df_categorical = None

            g = sns.pairplot(df,
                             vars=self._fields_to_plot,
                             hue=CLUSTER_FIELD_NAME_STR,
                             kind='scatter',
                             corner=False,
                             height=PAIRPLOT_FACET_GRID_HEIGHT,
                             aspect=PAIRPLOT_FACET_GRID_ASPECT,
                             palette=color_palette,
                             plot_kws=dict(s=PAIRPLOT_SCATTER_POINT_SIZE,
                                           alpha=PAIRPLOT_SCATTER_POINT_ALPHA,
                                           edgecolor=PAIRPLOT_SCATTER_POINT_EDGECOLOR))

            self._format_plot(g,
                              df,
                              variable_relationship_dict,
                              central_tendencies_df,
                              central_tendencies_per_cluster_df,
                              central_tendencies_df_categorical,
                              central_tendencies_per_cluster_df_categorical,
                              PairplotGroupingVariable.CLUSTER,
                              group,
                              None,
                              color_palette,
                              add_legend,
                              add_header,
                              add_central_tendency_marker,
                              add_central_tendency_marker_per_grouping_variable,
                              add_regression_line,
                              add_regression_line_per_grouping_variable)

            plt.show()

    @pairplot_decorator
    def plot_pairplot_seq_vars_per_group(self,
                                         groups_filter: Iterable[int] | None,
                                         clusters_filter: Iterable[int] | None,
                                         add_legend: bool,
                                         add_header: bool,
                                         add_central_tendency_marker: bool,
                                         add_central_tendency_marker_per_grouping_variable: bool,
                                         add_regression_line: bool,
                                         add_regression_line_per_grouping_variable: bool) -> None:

        print(DASH_STRING)
        print(PAIRPLOT_PER_GROUP_TITLE_STR)
        print(DASH_STRING)
        print('')

        color_palette_all_group, grouping_variable_index_mapping = self._return_color_palette_and_index_mapping_df(self.seq_stats_data,
                                                                                                                   PairplotGroupingVariable.GROUP)

        seq_stats_df = self._filter_seq_stats_df(self.seq_stats_data,
                                                 groups_filter,
                                                 clusters_filter)
        variable_relationship_dict = self._return_variable_relationships_per_group_per_cluster(seq_stats_df,
                                                                                               self._fields_to_plot)

        groups = sorted(seq_stats_df[PairplotGroupingVariable.GROUP.value].unique())
        color_palette = [color_palette_all_group[grouping_variable_index_mapping[i]] for i in groups]

        central_tendencies_per_group_df = self._calculate_central_tendencies(seq_stats_df,
                                                                             PairplotGroupingVariable.GROUP)
        central_tendencies_df = self._calculate_central_tendencies(seq_stats_df,
                                                                   None)
        if self.evaluation_metric_is_categorical:
            central_tendencies_per_group_df_categorical = self._calculate_central_tendencies_categorical_eval_metric(seq_stats_df,
                                                                                                                     PairplotGroupingVariable.GROUP)
            central_tendencies_df_categorical = self._calculate_central_tendencies_categorical_eval_metric(seq_stats_df,
                                                                                                           None)
        else:                                                            
            central_tendencies_per_group_df_categorical = None
            central_tendencies_df_categorical = None

        g = sns.pairplot(seq_stats_df,
                         vars=self._fields_to_plot,
                         hue=GROUP_FIELD_NAME_STR,
                         kind='scatter',
                         corner=False,
                         height=PAIRPLOT_FACET_GRID_HEIGHT,
                         aspect=PAIRPLOT_FACET_GRID_ASPECT,
                         palette=color_palette,
                         plot_kws=dict(s=PAIRPLOT_SCATTER_POINT_SIZE,
                                       alpha=PAIRPLOT_SCATTER_POINT_ALPHA,
                                       edgecolor=PAIRPLOT_SCATTER_POINT_EDGECOLOR))

        self._format_plot(g,
                          seq_stats_df,
                          variable_relationship_dict,
                          central_tendencies_df,
                          central_tendencies_per_group_df,
                          central_tendencies_df_categorical,
                          central_tendencies_per_group_df_categorical,
                          PairplotGroupingVariable.GROUP,
                          None,
                          None,
                          color_palette,
                          add_legend,
                          add_header,
                          add_central_tendency_marker,
                          add_central_tendency_marker_per_grouping_variable,
                          add_regression_line,
                          add_regression_line_per_grouping_variable)

        plt.show()
    
    def _format_plot(self,
                     g: FacetGrid,
                     plotting_df: pd.DataFrame,
                     variable_relationship_dict: dict,
                     central_tendencies_df: pd.DataFrame,
                     central_tendencies_per_grouping_var_df: pd.DataFrame,
                     central_tendencies_df_categorical: pd.DataFrame | None,
                     central_tendencies_per_grouping_var_df_categorical: pd.DataFrame | None,
                     grouping_variable: PairplotGroupingVariable,
                     group: int | None,
                     cluster: int | None,
                     color_palette: List[tuple],
                     add_legend: bool,
                     add_header: bool,
                     add_central_tendency_marker: bool,
                     add_central_tendency_marker_per_grouping_variable: bool,
                     add_regression_line: bool,
                     add_regression_line_per_grouping_variable: bool) -> None:

        for row, row_axes in enumerate(g.axes):
            for column, ax in enumerate(row_axes):

                # variable relationship header
                if add_header:
                    if self.evaluation_metric_is_categorical:
                        if (column != (self._n_plot_fields - 1)) and (row != (self._n_plot_fields - 1)) and (column != row):
                            self._add_variable_relation_header(variable_relationship_dict,
                                                               ax,
                                                               group,
                                                               cluster,
                                                               row,
                                                               column,
                                                               PAIRPLOT_CONTINUOUS_EVAL_METRIC_ADD_SLOPE_IN_HEADER)

                        if (column == (self._n_plot_fields - 1)) or (row == (self._n_plot_fields - 1)) and (column != row):
                            self._add_variable_relation_header(variable_relationship_dict,
                                                               ax,
                                                               group,
                                                               cluster,
                                                               row,
                                                               column,
                                                               PAIRPLOT_CATEGORICAL_EVAL_METRIC_ADD_SLOPE_IN_HEADER)

                    else:
                        if (column != row):
                            self._add_variable_relation_header(variable_relationship_dict,
                                                               ax,
                                                               group,
                                                               cluster,
                                                               row,
                                                               column,
                                                               PAIRPLOT_CONTINUOUS_EVAL_METRIC_ADD_SLOPE_IN_HEADER)

                # plots for categorical evaluation metric
                if self.evaluation_metric_is_categorical:
                    if (row == (self._n_plot_fields - 1)) and (column != row):
                        self._plot_stripplot(plotting_df,
                                             grouping_variable,
                                             self._fields_to_plot[column],
                                             self._fields_to_plot[-1],
                                             ax,
                                             color_palette,
                                             'h')

                    if (column == (self._n_plot_fields - 1)) and (column != row):
                        self._plot_stripplot(plotting_df,
                                             grouping_variable,
                                             self._fields_to_plot[-1],
                                             self._fields_to_plot[row],
                                             ax,
                                             color_palette,
                                             'v')

                # plot central tendency markers
                if self.evaluation_metric_is_categorical:
                    if (column != (self._n_plot_fields - 1)) and (row != (self._n_plot_fields - 1)) and (column != row):
                        if add_central_tendency_marker_per_grouping_variable:
                            self._add_central_tendency_marker_per_grouping_variable_scatter(central_tendencies_per_grouping_var_df,
                                                                                            grouping_variable,
                                                                                            self._fields_to_plot[column],
                                                                                            self._fields_to_plot[row],
                                                                                            ax,
                                                                                            color_palette)
                        if add_central_tendency_marker:
                            self._add_central_tendency_marker_scatter(central_tendencies_df,
                                                                      self._fields_to_plot[column],
                                                                      self._fields_to_plot[row],
                                                                      ax)

                    if (row == (self._n_plot_fields - 1)) and (column != row):
                        if add_central_tendency_marker_per_grouping_variable:
                            self._add_central_tendency_marker_per_grouping_variable_stripplot(central_tendencies_per_grouping_var_df_categorical,
                                                                                              grouping_variable,
                                                                                              self._fields_to_plot[column],
                                                                                              self._fields_to_plot[-1],
                                                                                              ax,
                                                                                              color_palette)
                        if add_central_tendency_marker:
                            self._add_central_tendency_marker_stripplot(central_tendencies_df_categorical,
                                                                        self._fields_to_plot[column],
                                                                        self._fields_to_plot[-1],
                                                                        ax)
                                                                    
                    if (column == (self._n_plot_fields - 1)) and (column != row):
                        if add_central_tendency_marker_per_grouping_variable:
                            self._add_central_tendency_marker_per_grouping_variable_stripplot(central_tendencies_per_grouping_var_df_categorical,
                                                                                              grouping_variable,
                                                                                              self._fields_to_plot[-1],
                                                                                              self._fields_to_plot[row],
                                                                                              ax,
                                                                                              color_palette)
                        if add_central_tendency_marker:
                            self._add_central_tendency_marker_stripplot(central_tendencies_df_categorical,
                                                                        self._fields_to_plot[-1],
                                                                        self._fields_to_plot[row],
                                                                        ax)
                                                                    

                else:
                    if (column != row):
                        if add_central_tendency_marker_per_grouping_variable:
                            self._add_central_tendency_marker_per_grouping_variable_scatter(central_tendencies_per_grouping_var_df,
                                                                                            grouping_variable,
                                                                                            self._fields_to_plot[column],
                                                                                            self._fields_to_plot[row],
                                                                                            ax,
                                                                                            color_palette)
                        if add_central_tendency_marker:
                            self._add_central_tendency_marker_scatter(central_tendencies_df,
                                                                      self._fields_to_plot[column],
                                                                      self._fields_to_plot[row],
                                                                      ax)
                # set axis limits
                if self.set_axis_lim_for_dtype:
                    if self.evaluation_metric_is_categorical:
                        if (column != (self._n_plot_fields - 1)) and (row != (self._n_plot_fields - 1)) and (column != row):
                            x_lim = return_axis_limits(plotting_df[self._fields_to_plot[column]],
                                                       self._fields_to_plot_axis_lim_cat[column][0],
                                                       self._fields_to_plot_axis_lim_cat[column][1])
                            y_lim = return_axis_limits(plotting_df[self._fields_to_plot[row]],
                                                       self._fields_to_plot_axis_lim_cat[row][0],
                                                       self._fields_to_plot_axis_lim_cat[row][1])
                            ax.set_xlim(x_lim)
                            ax.set_ylim(y_lim)

                        if (row == (self._n_plot_fields - 1)) and (column != row):
                            x_lim = return_axis_limits(plotting_df[self._fields_to_plot[column]],
                                                       self._fields_to_plot_axis_lim_cat[column][0],
                                                       self._fields_to_plot_axis_lim_cat[column][1])
                            y_lim = ax.get_ylim()
                            ax.set_xlim(x_lim)
                                                                        
                                                                        
                        if (column == (self._n_plot_fields - 1)) and (column != row):
                            x_lim = ax.get_xlim()
                            y_lim = return_axis_limits(plotting_df[self._fields_to_plot[row]],
                                                       self._fields_to_plot_axis_lim_cat[row][0],
                                                       self._fields_to_plot_axis_lim_cat[row][1])
                            ax.set_ylim(y_lim)
                                                                        

                    else:
                        if (column != row):
                            x_lim = return_axis_limits(plotting_df[self._fields_to_plot[column]],
                                                       self._fields_to_plot_axis_lim_cat[column][0],
                                                       self._fields_to_plot_axis_lim_cat[column][1])
                            y_lim = return_axis_limits(plotting_df[self._fields_to_plot[row]],
                                                       self._fields_to_plot_axis_lim_cat[row][0],
                                                       self._fields_to_plot_axis_lim_cat[row][1])
                            ax.set_xlim(x_lim)
                            ax.set_ylim(y_lim)
                else:
                    x_lim = ax.get_xlim()
                    y_lim = ax.get_ylim()

                # plot regression line
                if self.evaluation_metric_is_categorical:
                    if (column != (self._n_plot_fields - 1)) and (row != (self._n_plot_fields - 1)) and (column != row):
                        if add_regression_line_per_grouping_variable:
                            self._add_regression_line_per_grouping_variable(plotting_df,
                                                                            variable_relationship_dict,
                                                                            group,
                                                                            grouping_variable,
                                                                            row,
                                                                            column,
                                                                            ax,
                                                                            color_palette,
                                                                            x_lim,
                                                                            y_lim)

                        if add_regression_line:
                            self._add_regression_line(plotting_df,
                                                      variable_relationship_dict,
                                                      group,
                                                      row,
                                                      column,
                                                      ax,
                                                      x_lim,
                                                      y_lim)
                else:
                    if (column != row):
                        if add_regression_line_per_grouping_variable:
                            self._add_regression_line_per_grouping_variable(plotting_df,
                                                                            variable_relationship_dict,
                                                                            group,
                                                                            grouping_variable,
                                                                            row,
                                                                            column,
                                                                            ax,
                                                                            color_palette,
                                                                            x_lim,
                                                                            y_lim)
                        if add_regression_line:
                            self._add_regression_line(plotting_df,
                                                      variable_relationship_dict,
                                                      group,
                                                      row,
                                                      column,
                                                      ax,
                                                      x_lim,
                                                      y_lim)

                # adjust axis labels
                self._adjust_axis_labels(ax)

        sns.move_legend(g,
                        frameon=True,
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left',
                        markerscale=2)
        
        if not add_legend:
            g._legend.remove()

        plt.tight_layout()

    def _generate_plotting_df(self,
                              exclude_non_clustered: bool) -> pd.DataFrame:

        plotting_df_list = []
        for (_, _), df in self.learning_activity_sequence_stats_per_group.groupby([GROUP_FIELD_NAME_STR, SEQUENCE_ID_FIELD_NAME_STR]):
            df[USER_FIELD_NAME_STR] = list(df[LEARNING_ACTIVITY_SEQUENCE_USERS_NAME_STR].iloc[0])
            plotting_df_list.append(df)
        plotting_df = pd.concat(plotting_df_list).sort_values(by=[GROUP_FIELD_NAME_STR,
                                                                  LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR], 
                                                                  ascending=[True, False])
        plotting_df[CLUSTER_FIELD_NAME_STR] = plotting_df[CLUSTER_FIELD_NAME_STR].apply(lambda x: x[0])

        merge_df = self.interactions[self._merge_fields]
        evaluation_metric_merge_df = merge_df.loc[~merge_df.duplicated(), :]

        plotting_df = plotting_df.merge(evaluation_metric_merge_df, 
                                        on=[GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR], 
                                        how='left')

        plotting_df = plotting_df[self._fields_to_keep]

        if exclude_non_clustered:
            plotting_df = plotting_df.loc[plotting_df[CLUSTER_FIELD_NAME_STR] != -1, :]

        return plotting_df
    
    def _return_color_palette_and_index_mapping_df(self,
                                                   seq_stats_df: pd.DataFrame,
                                                   grouping_variable: PairplotGroupingVariable) -> Tuple[Iterable[Tuple[float]], dict]:

        grouping_variable_values = sorted(seq_stats_df.loc[seq_stats_df[grouping_variable.value] != -1, :][grouping_variable.value].unique())
        n_grouping_variable_values = len(grouping_variable_values)

        grouping_variable_index_mapping = dict(zip(grouping_variable_values, range(1, n_grouping_variable_values+1)))
        grouping_variable_index_mapping[-1] = 0

        color_palette = self._return_color_palette(n_grouping_variable_values,
                                                   PAIRPLOT_COLOR_PALETTE)
        color_palette.insert(0, (0,0,0))

        return (color_palette, grouping_variable_index_mapping)
    
    def _filter_seq_stats_df(self,
                             seq_stats_df: pd.DataFrame,
                             groups_filter: Iterable[int] | None,
                             clusters_filter: Iterable[int] | None) -> pd.DataFrame:

        seq_stats_df = seq_stats_df.copy()
        plotting_df_len = seq_stats_df.shape[0]
       
        group_filter = pd.Series([True] * plotting_df_len)
        cluster_filter = pd.Series([True] * plotting_df_len)
        
        if groups_filter is not None:
            check_iterable(groups_filter,
                           int)
            groups_filter = list(set(groups_filter))
            group_filter = seq_stats_df[GROUP_FIELD_NAME_STR].isin(groups_filter)
        
        if clusters_filter is not None:
            check_iterable(clusters_filter,
                           int)
            clusters_filter = list(set(clusters_filter))
            cluster_filter = seq_stats_df[CLUSTER_FIELD_NAME_STR].isin(clusters_filter)

        df_filter = group_filter & cluster_filter

        if sum(df_filter) == 0:
            raise ValueError(f'{PAIRPLOT_ERROR_FILTER_MISSPECIFICATION_NAME_STR}')

        seq_stats_df = seq_stats_df.loc[df_filter, :]

        return seq_stats_df

    def _plot_stripplot(self,
                        df: pd.DataFrame,
                        grouping_variable: PairplotGroupingVariable,
                        x_var: str,
                        y_var: str,
                        ax: matplotlib.axes.Axes,
                        palette: List[tuple],
                        orient: str) -> None:

        for collection in ax.collections:
            collection.remove()

        sns.stripplot(data=df, 
                      x=x_var,
                      y=y_var,
                      hue=grouping_variable.value,
                      orient=orient,
                      jitter=True,
                      palette=palette,
                      s=PAIRPLOT_STRIPPLOT_POINT_SIZE,
                      alpha=PAIRPLOT_STRIPPLOT_POINT_ALPHA,
                      linewidth=PAIRPLOT_STRIPPLOT_POINT_LINEWIDTH,
                      edgecolor=PAIRPLOT_STRIPPLOT_POINT_EDGECOLOR,
                      legend=False,
                      ax=ax)

    def _add_regression_line(self,
                             df: pd.DataFrame,
                             variable_relationship_dict: dict,
                             group: int | None,
                             row: int,
                             column: int,
                             ax: matplotlib.axes.Axes,
                             x_lim: Tuple[float, float],
                             y_lim: Tuple[float, float]) -> None:

        x_var = self._fields_to_plot[column]
        y_var = self._fields_to_plot[row]

        x_points = np.linspace(x_lim[0], 
                               x_lim[1], 
                               PAIRPLOT_REGRESSION_LINE_NUMBER_OF_POINTS)

        variable_relationship: PairplotData = self._return_variable_relationship(variable_relationship_dict,
                                                                                 group,
                                                                                 None,
                                                                                 x_var,
                                                                                 y_var)

        intercept = variable_relationship.intercept
        slope = variable_relationship.slope

        slope_is_none = slope is None
        slope_is_zero = slope == 0

        x_points_to_plot, y_hat_points_to_plot = self._return_reg_line_plotting_points(df,
                                                                                       x_var,
                                                                                       x_points,
                                                                                       y_lim,
                                                                                       intercept,
                                                                                       slope)

        plotting_df = pd.DataFrame({PAIRPLOT_X_VAR_FIELD_NAME_STR: x_points_to_plot,
                                    PAIRPLOT_Y_HAT_VAR_FIELD_NAME_STR: y_hat_points_to_plot})

        plotting_df = self._filter_df_by_axis_min_max_pct_lim(plotting_df,
                                                              PAIRPLOT_X_VAR_FIELD_NAME_STR,
                                                              PAIRPLOT_Y_HAT_VAR_FIELD_NAME_STR,
                                                              x_lim,
                                                              y_lim,
                                                              PAIRPLOT_REGRESSION_LINE_PROPORTION_AXIS_LIM,
                                                              slope_is_none,
                                                              slope_is_zero)
        
        sns.lineplot(data=plotting_df,
                     x=PAIRPLOT_X_VAR_FIELD_NAME_STR,
                     y=PAIRPLOT_Y_HAT_VAR_FIELD_NAME_STR,
                     errorbar=None,
                     legend=False,
                     color=PAIRPLOT_REGRESSION_LINE_COLOR,
                     linewidth=PAIRPLOT_REGRESSION_LINE_LINEWIDTH,
                     linestyle=PAIRPLOT_REGRESSION_LINE_LINE_STYLE,
                     antialiased=True,
                     zorder=500,
                     ax=ax)

    def _add_regression_line_per_grouping_variable(self,
                                                   df: pd.DataFrame,
                                                   variable_relationship_dict: dict,
                                                   group: int | None,
                                                   grouping_variable: PairplotGroupingVariable,
                                                   row: int,
                                                   column: int,
                                                   ax: matplotlib.axes.Axes,
                                                   palette: List[tuple],
                                                   x_lim: Tuple[float, float],
                                                   y_lim: Tuple[float, float]) -> None:

        x_var = self._fields_to_plot[column]
        y_var = self._fields_to_plot[row]

        x_points = np.linspace(x_lim[0], 
                               x_lim[1], 
                               PAIRPLOT_REGRESSION_LINE_NUMBER_OF_POINTS)

        plotting_df_list = []
        for grouping_variable_value, df_per_grouping_variable in df.groupby(grouping_variable.value):
            
            group_value, cluster_value = self._return_group_cluster_vals(grouping_variable,
                                                                         group,
                                                                         grouping_variable_value)

            variable_relationship: PairplotData = self._return_variable_relationship(variable_relationship_dict,
                                                                                     group_value,
                                                                                     cluster_value,
                                                                                     x_var,
                                                                                     y_var)

            intercept = variable_relationship.intercept
            slope = variable_relationship.slope

            slope_is_none = slope is None
            slope_is_zero = slope == 0

            x_points_to_plot, y_hat_points_to_plot = self._return_reg_line_plotting_points(df_per_grouping_variable,
                                                                                           x_var,
                                                                                           x_points,
                                                                                           y_lim,
                                                                                           intercept,
                                                                                           slope)

            plotting_df_per_grouping_variable = pd.DataFrame({PAIRPLOT_X_VAR_FIELD_NAME_STR: x_points_to_plot,
                                                              PAIRPLOT_Y_HAT_VAR_FIELD_NAME_STR: y_hat_points_to_plot,
                                                              grouping_variable.value: grouping_variable_value})

            plotting_df_per_grouping_variable = self._filter_df_by_axis_min_max_pct_lim(plotting_df_per_grouping_variable,
                                                                                        PAIRPLOT_X_VAR_FIELD_NAME_STR,
                                                                                        PAIRPLOT_Y_HAT_VAR_FIELD_NAME_STR,
                                                                                        x_lim,
                                                                                        y_lim,
                                                                                        PAIRPLOT_REGRESSION_LINE_PER_GROUPING_VARIABLE_PROPORTION_AXIS_LIM,
                                                                                        slope_is_none,
                                                                                        slope_is_zero)

            plotting_df_list.append(plotting_df_per_grouping_variable)
        
        plotting_df = pd.concat(plotting_df_list)

        sns.lineplot(data=plotting_df,
                     x=PAIRPLOT_X_VAR_FIELD_NAME_STR,
                     y=PAIRPLOT_Y_HAT_VAR_FIELD_NAME_STR,
                     hue=grouping_variable.value,
                     palette=palette,
                     errorbar=None,
                     legend=False,
                     linewidth=PAIRPLOT_REGRESSION_LINE_PER_GROUPING_VARIABLE_LINEWIDTH,
                     linestyle=PAIRPLOT_REGRESSION_LINE_PER_GROUPING_VARIABLE_LINE_STYLE,
                     antialiased=True,
                     zorder=500,
                     ax=ax)

    def _filter_df_by_axis_min_max_pct_lim(self,
                                           plotting_df: pd.DataFrame,
                                           x_var: str,
                                           y_var: str,
                                           x_lim: Tuple[float, float],
                                           y_lim: Tuple[float, float],
                                           proportion_ax_lim: float,
                                           slope_is_none: bool,
                                           slope_is_zero: bool) -> pd.DataFrame:

        x_min = x_lim[0] + x_lim[1] * proportion_ax_lim
        x_max = x_lim[1] - x_lim[1] * proportion_ax_lim

        y_min = y_lim[0] + y_lim[1] * proportion_ax_lim
        y_max = y_lim[1] - y_lim[1] * proportion_ax_lim

        x_min_filter = plotting_df[x_var] >= x_min 
        # x_max_filter = plotting_df[x_var] <= x_max 
        x_lim_filter = x_min_filter

        y_min_filter = plotting_df[y_var] >= y_min 
        # y_max_filter = plotting_df[y_var] <= y_max 
        y_lim_filter = y_min_filter

        if slope_is_none:
            axis_limits_filter =  y_lim_filter
        if slope_is_zero:
            axis_limits_filter =  x_lim_filter
        if not (slope_is_none or slope_is_zero):
            axis_limits_filter = x_lim_filter & y_lim_filter
    
        plotting_df = plotting_df.loc[axis_limits_filter, :]

        return plotting_df
    
    def _return_reg_line_plotting_points(self,
                                         df: pd.DataFrame,
                                         x_var: str,
                                         x_points: np.ndarray[float],
                                         y_lim: Tuple[float, float],
                                         intercept: float | None,
                                         slope: float | None) -> Tuple[np.ndarray[float | int], np.ndarray[float | int]]:

        if slope is None:
            x_value_array = np.unique(df[x_var])
            if len(x_value_array) != 1:
                raise ValueError(f'{PAIRPLOT_ERROR_NON_SLOPE_NO_UNIQUE_X_VALUE_NAME_STR}')
            else:
                x_value = x_value_array[0]

            x_points_to_plot = np.linspace(x_value, 
                                           x_value + PAIRPLOT_X_AXIS_OFFSET_IF_NO_X_VARIANCE, 
                                           PAIRPLOT_REGRESSION_LINE_NUMBER_OF_POINTS)
            y_hat_points_to_plot = np.linspace(y_lim[0], 
                                               y_lim[1], 
                                               PAIRPLOT_REGRESSION_LINE_NUMBER_OF_POINTS)
        else:
            x_points_to_plot = x_points
            y_hat_points_to_plot = intercept + slope * x_points
        
        return (x_points_to_plot, y_hat_points_to_plot)
    
    def _return_group_cluster_vals(self,
                                   grouping_field: PairplotGroupingVariable,
                                   group: int | None,
                                   grouping_var_value: int) -> Tuple[int | None, int | None]:

        match grouping_field:
            case PairplotGroupingVariable.GROUP:
                group_value = grouping_var_value
                cluster_value = None
            case PairplotGroupingVariable.CLUSTER:
                group_value = group
                cluster_value = grouping_var_value
        
        return group_value, cluster_value

    def _calculate_central_tendencies(self,
                                      seq_stats_data: pd.DataFrame,
                                      grouping_variable: PairplotGroupingVariable | None) -> pd.DataFrame:

        if grouping_variable:
            grouping_list = self._return_grouping_list(grouping_variable)

            central_tendencies_df = (seq_stats_data.groupby(grouping_list)[self._fields_to_plot]
                                                   .agg(np.mean)
                                                   .reset_index())
        else:
            central_tendencies_df = pd.DataFrame(seq_stats_data.agg(np.mean)).transpose()

        return central_tendencies_df

    def _calculate_central_tendencies_categorical_eval_metric(self,
                                                              seq_stats_data: pd.DataFrame,
                                                              grouping_variable: PairplotGroupingVariable | None) -> pd.DataFrame:

        if grouping_variable:
            grouping_list = self._return_grouping_list(grouping_variable) + [self.evaluation_metric_field]
        else:
            grouping_list = [self.evaluation_metric_field]

        central_tendencies_df_cat_eval_metric = (seq_stats_data.groupby(grouping_list)[self._fields_to_plot[:-1]]
                                                                .agg(np.mean)
                                                                .reset_index())

        return central_tendencies_df_cat_eval_metric
    
    def _return_grouping_list(self,
                              grouping_field: PairplotGroupingVariable) -> List[str]:

        match grouping_field:
            case PairplotGroupingVariable.GROUP:
                grouping_list = [PairplotGroupingVariable.GROUP.value]
            case PairplotGroupingVariable.CLUSTER:
                grouping_list = [PairplotGroupingVariable.GROUP.value, PairplotGroupingVariable.CLUSTER.value]

        return grouping_list

    def _add_central_tendency_marker_scatter(self,
                                             central_tendencies_df: pd.DataFrame,
                                             x_var: str,
                                             y_var: str,
                                             ax: matplotlib.axes.Axes) -> None:
        
        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        s=PAIRPLOT_CENTRAL_TENDENCY_MARKER_SIZE_OUTER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_MARKER_LINEWIDTH_OUTER,
                        alpha=PAIRPLOT_CENTRAL_TENDENCY_MARKER_ALPHA,
                        color=PAIRPLOT_CENTRAL_TENDENCY_MARKER_COLOR,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_MARKER_EDGECOLOR_OUTER,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_MARKER_KIND,
                        legend=False,
                        zorder=700,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        s=PAIRPLOT_CENTRAL_TENDENCY_MARKER_SIZE_INNER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_MARKER_LINEWIDTH_INNER,
                        alpha=PAIRPLOT_CENTRAL_TENDENCY_MARKER_ALPHA,
                        color=PAIRPLOT_CENTRAL_TENDENCY_MARKER_COLOR,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_MARKER_EDGECOLOR_INNER,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_MARKER_KIND,
                        legend=False,
                        zorder=700,
                        ax=ax)
    
    def _add_central_tendency_marker_per_grouping_variable_scatter(self,
                                                                   central_tendencies_df: pd.DataFrame,
                                                                   grouping_variable: PairplotGroupingVariable,
                                                                   x_var: str,
                                                                   y_var: str,
                                                                   ax: matplotlib.axes.Axes,
                                                                   palette: List[tuple]) -> None:
        
        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=grouping_variable.value, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_SIZE_OUTER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_EDGECOLOR_OUTER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_LINEWIDTH_OUTER,
                        legend=False,
                        zorder=600,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=grouping_variable.value, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_SIZE_INNER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_EDGECOLOR_INNER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_LINEWIDTH_INNER,
                        legend=False,
                        zorder=600,
                        ax=ax)

    def _add_central_tendency_marker_stripplot(self,
                                               central_tendencies_df: pd.DataFrame,
                                               x_var: str,
                                               y_var: str,
                                               ax: matplotlib.axes.Axes) -> None:

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        s=PAIRPLOT_CENTRAL_TENDENCY_MARKER_SIZE_OUTER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_MARKER_LINEWIDTH_OUTER,
                        alpha=PAIRPLOT_CENTRAL_TENDENCY_MARKER_ALPHA,
                        color=PAIRPLOT_CENTRAL_TENDENCY_MARKER_COLOR,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_MARKER_EDGECOLOR_OUTER,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_MARKER_KIND,
                        legend=False,
                        zorder=700,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        s=PAIRPLOT_CENTRAL_TENDENCY_MARKER_SIZE_INNER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_MARKER_LINEWIDTH_INNER,
                        alpha=PAIRPLOT_CENTRAL_TENDENCY_MARKER_ALPHA,
                        color=PAIRPLOT_CENTRAL_TENDENCY_MARKER_COLOR,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_MARKER_EDGECOLOR_INNER,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_MARKER_KIND,
                        legend=False,
                        zorder=700,
                        ax=ax)

    def _add_central_tendency_marker_per_grouping_variable_stripplot(self,
                                                                     central_tendencies_df: pd.DataFrame,
                                                                     grouping_variable: PairplotGroupingVariable,
                                                                     x_var: str,
                                                                     y_var: str,
                                                                     ax: matplotlib.axes.Axes,
                                                                     palette: List[tuple]) -> None:

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=grouping_variable.value, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_SIZE_OUTER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_EDGECOLOR_OUTER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_LINEWIDTH_OUTER,
                        legend=False,
                        zorder=600,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=grouping_variable.value, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_SIZE_INNER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_EDGECOLOR_INNER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_GROUPING_VARIABLE_MARKER_LINEWIDTH_INNER,
                        legend=False,
                        zorder=600,
                        ax=ax)

    def _adjust_axis_labels(self,
                            ax: matplotlib.axes.Axes) -> None:
        
        if PAIRPLOT_ADJUST_X_LABEL:

            y_label = ax.get_xlabel()
        
            y_label = self._transform_label(y_label,
                                            PAIRPLOT_X_LABEL_SPLIT_STRING,
                                            PAIRPLOT_X_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT)

            ax.set_xlabel(y_label, 
                            va='top',
                            ma='center',
                            rotation=PAIRPLOT_X_LABEL_ROTATION, 
                            labelpad=PAIRPLOT_X_LABEL_VERTICAL_PAD)  

        if PAIRPLOT_ADJUST_Y_LABEL:

            y_label = ax.get_ylabel()
        
            y_label = self._transform_label(y_label,
                                            PAIRPLOT_Y_LABEL_SPLIT_STRING,
                                            PAIRPLOT_Y_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT)

            ax.set_ylabel(y_label, 
                            ha='right',
                            ma='center',
                            rotation=PAIRPLOT_Y_LABEL_ROTATION, 
                            labelpad=PAIRPLOT_Y_LABEL_RIGHT_PAD)  

    def _transform_label(self,
                         label: str,
                         split_by_str: str,
                         words_per_line: int) -> str:

        n_words_label = len(label.split(split_by_str))

        if n_words_label > words_per_line:

            label_words = label.split(split_by_str)

            label = ''
            for n, word in enumerate(label_words):

                if n == 0:
                    label += word
                elif n % words_per_line == 0:
                    word = '\n' + word
                    label += word
                else:
                    word = split_by_str + word
                    label += word
        
        return label

    def _return_variable_relationship_df(self,
                                         variables_df: pd.DataFrame,
                                         fields: List[str]) -> pd.DataFrame:

        n_variables = len(fields)
        variables_df = variables_df[fields]

        variables_relationship_matrix = np.empty((n_variables, n_variables), dtype='object')

        for row in range(n_variables):
            for column in range(n_variables):
                x = variables_df.iloc[:, column]
                y = variables_df.iloc[:, row]

                if (len(np.unique(x)) == 1):

                    r = None
                    intercept = None
                    slope = None
                    p_value = None

                else:

                    result = sp.stats.linregress(x, y)
                    r = result.rvalue
                    intercept = result.intercept
                    slope = result.slope
                    p_value = result.pvalue

                results = PairplotData(r,
                                       intercept,
                                       slope,
                                       p_value)

                variables_relationship_matrix[row, column] = results

        return pd.DataFrame(variables_relationship_matrix, 
                            columns=fields, 
                            index=fields)

    def _return_variable_relationships_per_group_per_cluster(self,
                                                             seq_stats_df: pd.DataFrame,
                                                             fields: List[str]) -> Dict[Dict[str, PairplotData], Dict[str, Dict[str, PairplotData]]]:

        pairplot_variable_relationship_per_group = {}
        pairplot_variable_relationship_per_group_per_cluster = {}

        variables_df = seq_stats_df[fields]

        variable_relationship = self._return_variable_relationship_df(variables_df,
                                                                      fields)

        for group, df_per_group in seq_stats_df.groupby(GROUP_FIELD_NAME_STR):

            variables_df_per_group = df_per_group[fields]

            variable_relationship_per_group = self._return_variable_relationship_df(variables_df_per_group,
                                                                                    fields)

            pairplot_variable_relationship_per_group[group] = variable_relationship_per_group

            
            pairplot_variable_relationship_per_cluster = {}
            for cluster, df_per_group_per_clust in df_per_group.groupby(CLUSTER_FIELD_NAME_STR):

                variables_df_per_cluster = df_per_group_per_clust[fields]

                variable_relationship_per_group_per_cluster = self._return_variable_relationship_df(variables_df_per_cluster,
                                                                                                    fields)

                pairplot_variable_relationship_per_cluster[cluster] = variable_relationship_per_group_per_cluster

            pairplot_variable_relationship_per_group_per_cluster[group] = pairplot_variable_relationship_per_cluster

        return  {PAIRPLOT_VARIABLE_RELATIONSHIP: variable_relationship,
                 PAIRPLOT_VARIABLE_RELATIONSHIP_PER_GROUP: pairplot_variable_relationship_per_group,
                 PAIRPLOT_VARIABLE_RELATIONSHIP_PER_GROUP_PER_CLUSTER: pairplot_variable_relationship_per_group_per_cluster}

    def _return_variable_relationship(self,
                                      variable_relationship_dict: dict,
                                      group: int | None,
                                      cluster: int | None,
                                      x_var: str,
                                      y_var: str) -> PairplotData:
                                    
        if (group is None) and (cluster is None):
            variable_relationship: PairplotData = (variable_relationship_dict[PAIRPLOT_VARIABLE_RELATIONSHIP]
                                                                             .loc[y_var, x_var])

        elif (group is not None)  and (cluster is None):
            variable_relationship: PairplotData = (variable_relationship_dict[PAIRPLOT_VARIABLE_RELATIONSHIP_PER_GROUP]
                                                                             [group]
                                                                             .loc[y_var, x_var])
        elif (group is not None)  and (cluster is not None):
            variable_relationship: PairplotData = (variable_relationship_dict[PAIRPLOT_VARIABLE_RELATIONSHIP_PER_GROUP_PER_CLUSTER]
                                                                             [group]
                                                                             [cluster]
                                                                             .loc[y_var, x_var])
        else:
            raise ValueError(f'{PAIRPLOT_ERROR_RETURN_VARIABLE_RELATIONSHIP_NAME_STR}')
        
        return variable_relationship
    
    def _add_variable_relation_header(self,
                                      variable_relationship_dict: dict,
                                      ax: matplotlib.axes.Axes, 
                                      group: int | None,
                                      cluster: int | None,
                                      row: int,
                                      column: int,
                                      include_slope: bool) -> None:

        x_var = self._fields_to_plot[column]
        y_var = self._fields_to_plot[row]

        variable_relationship: PairplotData = self._return_variable_relationship(variable_relationship_dict,
                                                                                 group,
                                                                                 cluster,
                                                                                 x_var,
                                                                                 y_var)
        
        r = round(variable_relationship.corr, PAIRPLOT_PEARSON_CORRELATION_AND_PARAMS_ROUND_DIGITS)
        p = round(variable_relationship.p_value, PAIRPLOT_P_VALUE_ROUND_DIGITS)
        slope = round(variable_relationship.slope, PAIRPLOT_PEARSON_CORRELATION_AND_PARAMS_ROUND_DIGITS)
        intercept = round(variable_relationship.intercept, PAIRPLOT_PEARSON_CORRELATION_AND_PARAMS_ROUND_DIGITS)

        if include_slope:
            ax.set_title(f'r = {r}, slope = {slope}', fontsize=PAIRPLOT_HEADER_FONTSIZE)
        else:
            ax.set_title(f'r = {r}', fontsize=PAIRPLOT_HEADER_FONTSIZE)

    def _return_color_palette(self,
                              n_groups: int,
                              palette: str) -> Iterable[Tuple[float]]:

        color_palette = sns.color_palette(palette,
                                          n_colors=n_groups)
        return color_palette
    
    def _return_fields_to_plot(self,
                               fields_to_plot: List[PairplotFieldsToPlot]) -> List[str]:

        return [field.value for field in fields_to_plot]
    
    def _return_fields_to_plot_data_kind(self,
                                         fields_to_plot: List[PairplotFieldsToPlot]) -> List[Tuple[bool, bool]]:

        return [return_plot_field_data_kind(field) for field in fields_to_plot]