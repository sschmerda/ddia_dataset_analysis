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
        self._fields_to_plot = [LEARNING_ACTIVITY_SEQUENCE_PCT_REPEATED_LEARNING_ACTIVITIES_NAME_STR,
                                LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_NAME_STR,
                                LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_NAME_STR,
                                LEARNING_ACTIVITY_SEQUENCE_NUMBER_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ_NAME_STR,
                                LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQ_NAME_STR,
                                self.evaluation_metric_field]
        self._fields_to_plot_axis_lim_cat = [(True, False),
                                             (False, False),
                                             (True, False),
                                             (False, False),
                                             (False, False),
                                             (False, False),
                                             (True, True),
                                             (self.evaluation_metric_is_pct, self.evaluation_metric_pct_is_ratio)]
        self._fields_to_keep = self._fields + self._fields_to_plot

        self._merge_fields = [GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR, self.evaluation_metric_field]
    
    @pairplot_decorator
    def plot_pairplot(self) -> None:

        print(DASH_STRING)
        print(f'Pairplot of Analysis-Relevant Variables Subdivided by {CLUSTER_FIELD_NAME_STR}s for each {GROUP_FIELD_NAME_STR}')
        print(DASH_STRING)
        print('')

        seq_stats_data = self._generate_plotting_df()

        color_palette = self._return_color_palette(seq_stats_data,
                                                   SEABORN_COLOR_PALETTE)

        for group, df in seq_stats_data.groupby(GROUP_FIELD_NAME_STR):

            print(STAR_STRING)
            print(f'{GROUP_FIELD_NAME_STR}: {group}')
            print(STAR_STRING)

            n_plot_fields = len(self._fields_to_plot)

            color_palette_new = color_palette[:df[CLUSTER_FIELD_NAME_STR].nunique()]

            g = sns.pairplot(df,
                             vars=self._fields_to_plot,
                             hue=CLUSTER_FIELD_NAME_STR,
                             kind='scatter',
                             corner=False,
                             height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                             aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                             palette=color_palette_new,
                             plot_kws=dict(
                                            s=SEABORN_POINT_SIZE_FACET_CLUSTER_2D,
                                            alpha=SEABORN_POINT_ALPHA_FACET_CLUSTER_2D,
                                            edgecolor=SEABORN_POINT_EDGECOLOR))

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
                                                 color_palette_new,
                                                 'h')

                        if (column == (n_plot_fields - 1)) and (column != row):
                            self._plot_stripplot(df,
                                                 self._fields_to_plot[-1],
                                                 self._fields_to_plot[row],
                                                 ax,
                                                 color_palette_new,
                                                 'v')
                    # plot central tendency markers
                    if self.evaluation_metric_is_categorical:
                        if (column != (n_plot_fields - 1)) and (row != (n_plot_fields - 1)) and (column != row):
                            self._add_central_tendency_marker_per_cluster_scatter(central_tendencies_per_cluster_df,
                                                                                  self._fields_to_plot[column],
                                                                                  self._fields_to_plot[row],
                                                                                  ax,
                                                                                  color_palette_new)
                            self._add_central_tendency_marker_scatter(central_tendencies_df,
                                                                      self._fields_to_plot[column],
                                                                      self._fields_to_plot[row],
                                                                      ax)

                        if (row == (n_plot_fields - 1)) and (column != row):
                            self._add_central_tendency_marker_per_cluster_stripplot(central_tendencies_per_cluster_df_categorical,
                                                                                    self._fields_to_plot[column],
                                                                                    self._fields_to_plot[-1],
                                                                                    ax,
                                                                                    color_palette_new)
                            self._add_central_tendency_marker_stripplot(central_tendencies_df_categorical,
                                                                        self._fields_to_plot[column],
                                                                        self._fields_to_plot[-1],
                                                                        ax)
                                                                        
                        if (column == (n_plot_fields - 1)) and (column != row):
                            self._add_central_tendency_marker_per_cluster_stripplot(central_tendencies_per_cluster_df_categorical,
                                                                                    self._fields_to_plot[-1],
                                                                                    self._fields_to_plot[row],
                                                                                    ax,
                                                                                    color_palette_new)
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
                                                                                  color_palette_new)
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

    def _generate_plotting_df(self) -> None:

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

        if self.exclude_non_clustered:
            plotting_df = plotting_df[plotting_df[CLUSTER_FIELD_NAME_STR] != -1]

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
                      s=5,
                      alpha=SEABORN_POINT_ALPHA_FACET_CLUSTER_2D,
                      linewidth=0.35,
                      edgecolor=SEABORN_POINT_EDGECOLOR,
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
                        s=300,
                        linewidth=1.8,
                        alpha=SEABORN_MARKER_ALPHA,
                        color='red',
                        edgecolor='white',
                        marker='*',
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        s=150,
                        linewidth=1.4,
                        alpha=SEABORN_MARKER_ALPHA,
                        color='red',
                        edgecolor='black',
                        marker='*',
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
                        s=80,
                        palette=palette,
                        marker='D',
                        edgecolor='white',
                        linewidth=1.8,
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=CLUSTER_FIELD_NAME_STR, 
                        s=40,
                        palette=palette,
                        marker='D',
                        edgecolor='black',
                        linewidth=1.4,
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
                        s=300,
                        linewidth=1.8,
                        alpha=SEABORN_MARKER_ALPHA,
                        color='red',
                        edgecolor='white',
                        marker='*',
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        s=150,
                        linewidth=1.4,
                        alpha=SEABORN_MARKER_ALPHA,
                        color='red',
                        edgecolor='black',
                        marker='*',
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
                        s=80,
                        palette=palette,
                        marker='D',
                        edgecolor='white',
                        linewidth=1.8,
                        legend=False,
                        zorder=100,
                        ax=ax)

        sns.scatterplot(data=central_tendencies_df,
                        x=x_var, 
                        y=y_var,
                        hue=CLUSTER_FIELD_NAME_STR, 
                        s=40,
                        palette=palette,
                        marker='D',
                        edgecolor='black',
                        linewidth=1.4,
                        legend=False,
                        zorder=100,
                        ax=ax)
    
    def _rotate_axis_label(self,
                           ax: matplotlib.axes.Axes) -> None:

        ax.set_xlabel(ax.get_xlabel(), rotation=90)  
        ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=160)
    
    def _add_correlation_header(self,
                                df: pd.DataFrame,
                                ax: matplotlib.axes.Axes,
                                row: int,
                                column: int) -> None:

        x = df[self._fields_to_plot].iloc[:, row] 
        y = df[self._fields_to_plot].iloc[:, column] 

        result = sp.stats.pearsonr(x, y)
        r = round(result.statistic, 2)
        p = round(result.pvalue, 3)

        ax.set_title(f'r = {r} p = {p}', fontsize=15)

    def _return_color_palette(self,
                              data: pd.DataFrame,
                              palette: str) -> Iterable[Tuple[float]]:

        n_groups = data[CLUSTER_FIELD_NAME_STR].nunique()

        color_palette = sns.color_palette(palette,
                                          n_colors=n_groups)
        return color_palette