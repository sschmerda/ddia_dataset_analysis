from .configs.general_config import *
from .configs.sequence_pairplot_config import *
from .constants.constants import *
from .standard_import import *
from .plotting_functions import *
from .result_tables import ResultTables

class LearningActivitySequencePairplot():
    """A class used for plotting group-wise pairplots of relevant variables"""

    def __init__(self, 
                 result_tables: ResultTables,
                 exclude_non_clustered: bool,
                 set_axis_lim_for_dtype: bool,
                 add_central_tendency_markers: bool,
                 evaluation_metric_field: str,
                 evaluation_metric_is_categorical: bool,
                 evaluation_metric_is_pct: bool,
                 evaluation_metric_pct_is_ratio: bool) -> None:

        self.interactions: pd.DataFrame = result_tables.interactions.copy()
        self.learning_activity_sequence_stats_per_group: pd.DataFrame = result_tables.learning_activity_sequence_stats_per_group.copy()
        self.exclude_non_clustered: bool = exclude_non_clustered
        self.set_axis_lim_for_dtype: bool = set_axis_lim_for_dtype
        self.add_central_tendency_markers: bool = add_central_tendency_markers
        self.evaluation_metric_field: str = evaluation_metric_field
        self.evaluation_metric_is_categorical: bool = evaluation_metric_is_categorical
        self.evaluation_metric_is_pct: bool = evaluation_metric_is_pct
        self.evaluation_metric_pct_is_ratio: bool = evaluation_metric_pct_is_ratio

        self._fields = [GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR, SEQUENCE_ID_FIELD_NAME_STR]
        self._fields_to_plot = self._return_fields_to_plot(PAIRPLOT_FIELDS_TO_PLOT_LIST) + [self.evaluation_metric_field]
        self._fields_to_plot_axis_lim_cat = self._return_fields_to_plot_data_kind(PAIRPLOT_FIELDS_TO_PLOT_LIST) + [(self.evaluation_metric_is_pct, self.evaluation_metric_pct_is_ratio)]
        self._fields_to_keep = self._fields + self._fields_to_plot

        self._merge_fields = [GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, self.evaluation_metric_field]
    
    @pairplot_decorator
    def plot_pairplot(self) -> None:

        print(DASH_STRING)
        print(PAIRPLOT_TITLE_STR)
        print(DASH_STRING)
        print('')

        seq_stats_data = self._generate_plotting_df()

        n_clusters_all_groups = seq_stats_data.loc[seq_stats_data[CLUSTER_FIELD_NAME_STR] != -1, :][CLUSTER_FIELD_NAME_STR].nunique()

        for group, df in seq_stats_data.groupby(GROUP_FIELD_NAME_STR):

            print(STAR_STRING)
            print(f'{GROUP_FIELD_NAME_STR}: {group}')
            print(STAR_STRING)

            n_plot_fields = len(self._fields_to_plot)
            number_clusters_within_group = df.loc[df[CLUSTER_FIELD_NAME_STR] != -1, :][CLUSTER_FIELD_NAME_STR].nunique()

            if self.exclude_non_clustered:
                df = df.loc[df[CLUSTER_FIELD_NAME_STR] != -1, :]
                color_palette = self._return_color_palette(n_clusters_all_groups,
                                                           PAIRPLOT_COLOR_PALETTE)
                color_palette = color_palette[:number_clusters_within_group]
            else:
                if -1 in df[CLUSTER_FIELD_NAME_STR].unique():
                    color_palette = self._return_color_palette(n_clusters_all_groups,
                                                               PAIRPLOT_COLOR_PALETTE)
                    color_palette.insert(0, (0,0,0))
                    color_palette = color_palette[:number_clusters_within_group + 1]
                else:
                    color_palette = self._return_color_palette(n_clusters_all_groups,
                                                               PAIRPLOT_COLOR_PALETTE)
                    color_palette = color_palette[:number_clusters_within_group]


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

            central_tendencies_per_cluster_df = df.groupby(CLUSTER_FIELD_NAME_STR)[self._fields_to_plot].agg(np.mean).reset_index()
            central_tendencies_df = df.groupby(GROUP_FIELD_NAME_STR)[self._fields_to_plot].agg(np.mean).reset_index()
            if self.evaluation_metric_is_categorical:
                central_tendencies_per_cluster_df_categorical = df.groupby([CLUSTER_FIELD_NAME_STR, self.evaluation_metric_field])[self._fields_to_plot[:-1]].agg(np.mean).reset_index()
                central_tendencies_df_categorical = df.groupby([self.evaluation_metric_field])[self._fields_to_plot[:-1]].agg(np.mean).reset_index()

            for row, row_axes in enumerate(g.axes):
                for column, ax in enumerate(row_axes):

                    # rotate axis labels
                    self._rotate_axis_label(ax)

                    # correlation header
                    if self.evaluation_metric_is_categorical:
                        if (column != (n_plot_fields - 1)) and (row != (n_plot_fields - 1)) and (column != row):
                            self._add_correlation_header(df,
                                                         ax,
                                                         row,
                                                         column)

                    else:
                        if (column != row):
                            self._add_correlation_header(df,
                                                         ax,
                                                         row,
                                                         column)

                    # plots for categorical evaluation metric
                    if self.evaluation_metric_is_categorical:
                        if (row == (n_plot_fields - 1)) and (column != row):
                            self._plot_stripplot(df,
                                                 self._fields_to_plot[column],
                                                 self._fields_to_plot[-1],
                                                 ax,
                                                 color_palette,
                                                 'h')

                        if (column == (n_plot_fields - 1)) and (column != row):
                            self._plot_stripplot(df,
                                                 self._fields_to_plot[-1],
                                                 self._fields_to_plot[row],
                                                 ax,
                                                 color_palette,
                                                 'v')
                    # plot central tendency markers
                    if self.add_central_tendency_markers:
                        if self.evaluation_metric_is_categorical:
                            if (column != (n_plot_fields - 1)) and (row != (n_plot_fields - 1)) and (column != row):
                                self._add_central_tendency_marker_per_cluster_scatter(central_tendencies_per_cluster_df,
                                                                                    self._fields_to_plot[column],
                                                                                    self._fields_to_plot[row],
                                                                                    ax,
                                                                                    color_palette)
                                self._add_central_tendency_marker_scatter(central_tendencies_df,
                                                                        self._fields_to_plot[column],
                                                                        self._fields_to_plot[row],
                                                                        ax)

                            if (row == (n_plot_fields - 1)) and (column != row):
                                self._add_central_tendency_marker_per_cluster_stripplot(central_tendencies_per_cluster_df_categorical,
                                                                                        self._fields_to_plot[column],
                                                                                        self._fields_to_plot[-1],
                                                                                        ax,
                                                                                        color_palette)
                                self._add_central_tendency_marker_stripplot(central_tendencies_df_categorical,
                                                                            self._fields_to_plot[column],
                                                                            self._fields_to_plot[-1],
                                                                            ax)
                                                                            
                            if (column == (n_plot_fields - 1)) and (column != row):
                                self._add_central_tendency_marker_per_cluster_stripplot(central_tendencies_per_cluster_df_categorical,
                                                                                        self._fields_to_plot[-1],
                                                                                        self._fields_to_plot[row],
                                                                                        ax,
                                                                                        color_palette)
                                self._add_central_tendency_marker_stripplot(central_tendencies_df_categorical,
                                                                            self._fields_to_plot[-1],
                                                                            self._fields_to_plot[row],
                                                                            ax)
                                                                            

                        else:
                            if (column != row):
                                self._add_central_tendency_marker_per_cluster_scatter(central_tendencies_per_cluster_df,
                                                                                    self._fields_to_plot[column],
                                                                                    self._fields_to_plot[row],
                                                                                    ax,
                                                                                    color_palette)
                                self._add_central_tendency_marker_scatter(central_tendencies_df,
                                                                        self._fields_to_plot[column],
                                                                        self._fields_to_plot[row],
                                                                        ax)
                    # set axis limits
                    if self.set_axis_lim_for_dtype:
                        if self.evaluation_metric_is_categorical:
                            if (column != (n_plot_fields - 1)) and (row != (n_plot_fields - 1)) and (column != row):
                                x_lim = return_axis_limits(df[self._fields_to_plot[column]],
                                                           self._fields_to_plot_axis_lim_cat[column][0],
                                                           self._fields_to_plot_axis_lim_cat[column][1])
                                y_lim = return_axis_limits(df[self._fields_to_plot[row]],
                                                           self._fields_to_plot_axis_lim_cat[row][0],
                                                           self._fields_to_plot_axis_lim_cat[row][1])
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                            if (row == (n_plot_fields - 1)) and (column != row):
                                x_lim = return_axis_limits(df[self._fields_to_plot[column]],
                                                           self._fields_to_plot_axis_lim_cat[column][0],
                                                           self._fields_to_plot_axis_lim_cat[column][1])
                                ax.set_xlim(x_lim)
                                                                            
                                                                            
                            if (column == (n_plot_fields - 1)) and (column != row):
                                y_lim = return_axis_limits(df[self._fields_to_plot[row]],
                                                           self._fields_to_plot_axis_lim_cat[row][0],
                                                           self._fields_to_plot_axis_lim_cat[row][1])
                                ax.set_ylim(y_lim)
                                                                            

                        else:
                            if (column != row):
                                x_lim = return_axis_limits(df[self._fields_to_plot[column]],
                                                           self._fields_to_plot_axis_lim_cat[column][0],
                                                           self._fields_to_plot_axis_lim_cat[column][1])
                                y_lim = return_axis_limits(df[self._fields_to_plot[row]],
                                                           self._fields_to_plot_axis_lim_cat[row][0],
                                                           self._fields_to_plot_axis_lim_cat[row][1])
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

            sns.move_legend(g,
                            frameon=True,
                            bbox_to_anchor=(1, 0.5), 
                            loc='center left',
                            markerscale=2)

            plt.tight_layout()
            plt.show()

    def _generate_plotting_df(self) -> pd.DataFrame:

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

        return plotting_df
    
    def _plot_stripplot(self,
                        df: pd.DataFrame,
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
                      hue=CLUSTER_FIELD_NAME_STR,
                      orient=orient,
                      jitter=True,
                      palette=palette,
                      s=PAIRPLOT_STRIPPLOT_POINT_SIZE,
                      alpha=PAIRPLOT_STRIPPLOT_POINT_ALPHA,
                      linewidth=PAIRPLOT_STRIPPLOT_POINT_LINEWIDTH,
                      edgecolor=PAIRPLOT_STRIPPLOT_POINT_EDGECOLOR,
                      legend=False,
                      ax=ax)

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
                        zorder=100,
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
                        zorder=100,
                        ax=ax)
    
    def _add_central_tendency_marker_per_cluster_scatter(self,
                                                         central_tendencies_df: pd.DataFrame,
                                                         x_var: str,
                                                         y_var: str,
                                                         ax: matplotlib.axes.Axes,
                                                         palette: List[tuple]) -> None:
        
        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=CLUSTER_FIELD_NAME_STR, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_SIZE_OUTER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_EDGECOLOR_OUTER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_LINEWIDTH_OUTER,
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=CLUSTER_FIELD_NAME_STR, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_SIZE_INNER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_EDGECOLOR_INNER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_LINEWIDTH_INNER,
                        legend=False,
                        zorder=100,
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
                        zorder=100,
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
                        zorder=100,
                        ax=ax)

    def _add_central_tendency_marker_per_cluster_stripplot(self,
                                                           central_tendencies_df: pd.DataFrame,
                                                           x_var: str,
                                                           y_var: str,
                                                           ax: matplotlib.axes.Axes,
                                                           palette: List[tuple]) -> None:

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=CLUSTER_FIELD_NAME_STR, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_SIZE_OUTER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_EDGECOLOR_OUTER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_LINEWIDTH_OUTER,
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=CLUSTER_FIELD_NAME_STR, 
                        s=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_SIZE_INNER,
                        palette=palette,
                        marker=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_KIND,
                        edgecolor=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_EDGECOLOR_INNER,
                        linewidth=PAIRPLOT_CENTRAL_TENDENCY_PER_CLUSTER_MARKER_LINEWIDTH_INNER,
                        legend=False,
                        zorder=100,
                        ax=ax)
    
    def _rotate_axis_label(self,
                           ax: matplotlib.axes.Axes) -> None:

        ax.set_xlabel(ax.get_xlabel(), rotation=PAIRPLOT_X_LABEL_ROTATION, labelpad=PAIRPLOT_X_LABEL_PAD)  
        ax.set_ylabel(ax.get_ylabel(), rotation=PAIRPLOT_Y_LABEL_ROTATION, labelpad=PAIRPLOT_Y_LABEL_PAD)
    
    def _add_correlation_header(self,
                                df: pd.DataFrame,
                                ax: matplotlib.axes.Axes,
                                row: int,
                                column: int) -> None:

        x = df[self._fields_to_plot].iloc[:, row] 
        y = df[self._fields_to_plot].iloc[:, column] 

        result = sp.stats.pearsonr(x, y)
        r = round(result.statistic, PAIRPLOT_PEARSON_CORRELATION_ROUND_DIGITS)
        p = round(result.pvalue, PAIRPLOT_PEARSON_CORRELATION_P_VALUE_ROUND_DIGITS)

        ax.set_title(f'r = {r} p = {p}', fontsize=PAIRPLOT_HEADER_FONTSIZE)

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

