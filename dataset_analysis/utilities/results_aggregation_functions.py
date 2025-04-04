from .configs.result_aggregation_config import *
from .plotting_functions import *
from .html_style_functions import *

class AggregatedResults():
    """docstring for ClassName."""
    def __init__(self):

        self._path_to_result_tables = self._return_result_tables_paths()

        self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names = [strength_value + RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_COUNT_NAME_STR for strength_value in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES]
        self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names = [strength_value + RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_PCT_NAME_STR for strength_value in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES]
        self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names = [strength_value + RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_COUNT_PCT_NAME_STR for strength_value in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES]

    @avg_sequence_statistics_per_group_per_dataset_decorator
    def plot_avg_sequence_statistics_per_group_per_dataset(self) -> None:

        avg_sequence_stats_fields = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]
        avg_sequence_statistics_per_group_per_dataset = self._return_avg_sequence_stats_per_group_per_dataset_df(avg_sequence_stats_fields,
                                                                                                                 False)

        avg_unique_sequence_freq_stats_fields = [field.value for field in UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST]
        avg_unique_sequence_frequency_statistics_per_group_per_dataset = self._return_avg_sequence_stats_per_group_per_dataset_df(avg_unique_sequence_freq_stats_fields,
                                                                                                                                  True)

        fields_to_plot = SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST + UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST

        for field in fields_to_plot:

            avg_method_str = AVG_SEQUENCE_STATISTICS_AVERAGING_METHOD.value.capitalize() + ' '
            x_label = avg_method_str + field.value + AVG_SEQUENCE_STATISTICS_X_LABEL_SUFFIX

            print(STAR_STRING)
            print(x_label)
            print(STAR_STRING)

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                    data = avg_sequence_statistics_per_group_per_dataset
                    x_axis_ticks = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_RAW
                    x_axis_log_scale = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_RAW
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    False,
                                                    False,
                                                    is_log_scale=x_axis_log_scale)

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                     SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = avg_sequence_statistics_per_group_per_dataset
                    x_axis_ticks = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_PCT
                    x_axis_log_scale = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_PCT
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    True,
                                                    False,
                                                    is_log_scale=x_axis_log_scale)


                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = avg_sequence_statistics_per_group_per_dataset
                    x_axis_ticks = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_PCT_RATIO
                    x_axis_log_scale = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_PCT_RATIO
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    True,
                                                    True,
                                                    is_log_scale=x_axis_log_scale)

                case UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY:

                    data = avg_unique_sequence_frequency_statistics_per_group_per_dataset
                    x_axis_ticks = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_RAW
                    x_axis_log_scale = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_RAW
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    False,
                                                    False,
                                                    is_log_scale=x_axis_log_scale)

                case UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY:

                    data = avg_unique_sequence_frequency_statistics_per_group_per_dataset
                    x_axis_ticks = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_PCT
                    x_axis_log_scale = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_PCT
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    True,
                                                    False,
                                                    is_log_scale=x_axis_log_scale)

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{UniqueSequenceFrequencyStatisticsPlotFields.__name__}')

            color_dict = self._return_color_per_dataset(data,
                                                        ColorPaletteAggregationLevel.DATASET,
                                                        AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_PALETTE,
                                                        AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_ALPHA,
                                                        AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_SATURATION)

            if AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_BOXES:
                data = self._sort_groups_by_metric(data,
                                                   field,
                                                   AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_METRIC,
                                                   SequenceStatisticsDistributionSortingEntity.DATASET,
                                                   AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_ASCENDING)
            colors = self._return_colors(data,
                                         ColorPaletteAggregationLevel.DATASET,
                                         None,
                                         color_dict)
            
            # boxplot
            g = sns.boxplot(data, 
                            x=field.value, 
                            y=DATASET_NAME_FIELD_NAME_STR, 
                            hue=DATASET_NAME_FIELD_NAME_STR,
                            showfliers=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_OUTLIERS,
                            linewidth=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_LINE_WIDTH,
                            width=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WIDTH,
                            whis=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WHISKERS,
                            showmeans=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_MEANS,
                            meanprops=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_MARKER)
            # strip or swarmplot
            g = sns.swarmplot(data, 
                              x=field.value, 
                              y=DATASET_NAME_FIELD_NAME_STR, 
                              size=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_SIZE, 
                              color=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_COLOR,
                              alpha=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_ALPHA,
                              edgecolor=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_EDGECOLOR,
                              linewidth=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_LINEWIDTH)

            self._set_axis_labels(g,
                                  AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_X_AXIS_LABEL,
                                  AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_Y_AXIS_LABEL,
                                  x_label,
                                  AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_Y_AXIS_LABEL)

            self._set_axis_ticks(g,
                                 AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_X_AXIS_TICKS,
                                 AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_Y_AXIS_TICKS,
                                 AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_X_AXIS_TICK_LABELS,
                                 AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_PLOT_Y_AXIS_TICK_LABELS,
                                 x_axis_ticks,
                                 None)

            if x_axis_log_scale:
                self._set_log_scale_axes(g,
                                         Axes.X_AXIS)
            g.grid(True,
                   axis=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_GRID_LINE_AXIS.value,
                   which='both')

            for box, col in zip(g.patches, colors):
                box.set_facecolor(col)
            
            g.set(xlim=(x_axis_lim))

            plt.tight_layout()
            title = AVG_SEQUENCE_STATISTICS_PLOT_NAME + field.value
            self._save_figure(title)
            plt.show(g);

    @summary_sequence_statistics_per_group_per_dataset_decorator
    def plot_summary_sequence_statistics_per_group_per_dataset(self) -> None:

        sequence_summary_stats_fields = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]
        sequence_summary_stats_per_group_per_dataset = self._return_summary_sequence_stats_per_group_per_dataset_df(sequence_summary_stats_fields,
                                                                                                                    False)

        unique_sequence_freq_summary_stats_fields = [field.value for field in UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST]
        unique_sequence_frequency_summary_stats_per_group_per_dataset = self._return_summary_sequence_stats_per_group_per_dataset_df(unique_sequence_freq_summary_stats_fields,
                                                                                                                                     True)

        fields_to_plot = SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST + UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST

        for field in fields_to_plot:

            print(STAR_STRING)
            print(field.value)
            print(STAR_STRING)

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                    data = sequence_summary_stats_per_group_per_dataset
                    y_axis_ticks = SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW
                    y_axis_log_scale = SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_RAW

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                     SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = sequence_summary_stats_per_group_per_dataset
                    y_axis_ticks = SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_PCT
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT
                    y_axis_log_scale = SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_PCT

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = sequence_summary_stats_per_group_per_dataset
                    y_axis_ticks = SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_PCT_RATIO
                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT_RATIO
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT_RATIO
                    y_axis_log_scale = SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_PCT_RATIO

                case UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_summary_stats_per_group_per_dataset
                    y_axis_ticks = SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW
                    y_axis_log_scale = SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_RAW

                case UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_summary_stats_per_group_per_dataset
                    y_axis_ticks = SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_PCT
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT
                    y_axis_log_scale = SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_PCT

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{UniqueSequenceFrequencyStatisticsPlotFields.__name__}')

            n_cols = set_facet_grid_column_number(data[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            color_dict = self._return_color_per_group_per_dataset(data,
                                                                  ColorPaletteAggregationLevel.GROUP,
                                                                  SUMMARY_SEQUENCE_STATISTICS_COLOR_PALETTE,
                                                                  SUMMARY_SEQUENCE_STATISTICS_COLOR_ALPHA,
                                                                  SUMMARY_SEQUENCE_STATISTICS_COLOR_SATURATION)

            # TODO: set new statistcs variable?
            g = sns.relplot(data,
                            x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                            y=field.value,
                            hue=GROUP_FIELD_NAME_STR,
                            col=DATASET_NAME_FIELD_NAME_STR,
                            col_wrap=n_cols,
                            height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                            aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                            kind=SUMMARY_SEQUENCE_STATISTICS_KIND,
                            legend=SUMMARY_SEQUENCE_STATISTICS_PLOT_LEGEND,
                            linewidth=SUMMARY_SEQUENCE_STATISTICS_LINE_WIDTH,
                            markers=SUMMARY_SEQUENCE_STATISTICS_PLOT_MARKERS,
                            marker=SUMMARY_SEQUENCE_STATISTICS_MARKER_TYPE,
                            markersize=SUMMARY_SEQUENCE_STATISTICS_MARKER_SIZE,
                            markerfacecolor=SUMMARY_SEQUENCE_STATISTICS_MARKER_FACECOLOR,
                            markeredgecolor=SUMMARY_SEQUENCE_STATISTICS_MARKER_EDGECOLOR,
                            markeredgewidth=SUMMARY_SEQUENCE_STATISTICS_MARKER_EDGEWIDTH,
                            alpha=SUMMARY_SEQUENCE_STATISTICS_MARKER_ALPHA,
                            facet_kws=dict(sharex=share_x,
                                           sharey=share_y))

            g.set_titles('{col_name}') 

            if share_y:
                y_axis_lim = return_axis_limits(data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio,
                                                is_log_scale=y_axis_log_scale)
                g.set(ylim=y_axis_lim)
            
            for ax, (dataset_name, facet_data) in zip(g.axes.flat, data.groupby(DATASET_NAME_FIELD_NAME_STR)):

                self._set_axis_labels(ax,
                                      SUMMARY_SEQUENCE_STATISTICS_PLOT_X_AXIS_LABEL,
                                      SUMMARY_SEQUENCE_STATISTICS_PLOT_Y_AXIS_LABEL,
                                      SUMMARY_SEQUENCE_STATISTICS_X_AXIS_LABEL,
                                      SUMMARY_SEQUENCE_STATISTICS_Y_AXIS_LABEL)

                self._set_axis_ticks(ax,
                                     SUMMARY_SEQUENCE_STATISTICS_PLOT_X_AXIS_TICKS,
                                     SUMMARY_SEQUENCE_STATISTICS_PLOT_Y_AXIS_TICKS,
                                     SUMMARY_SEQUENCE_STATISTICS_PLOT_X_AXIS_TICK_LABELS,
                                     SUMMARY_SEQUENCE_STATISTICS_PLOT_Y_AXIS_TICK_LABELS,
                                     None,
                                     y_axis_ticks)

                if y_axis_log_scale:
                    self._set_log_scale_axes(ax,
                                             Axes.Y_AXIS)
                ax.grid(True,
                        axis=SUMMARY_SEQUENCE_STATISTICS_GRID_LINE_AXIS.value,
                        which='both')

                if not share_y:
                    y_axis_lim = return_axis_limits(facet_data[field.value],
                                                    statistic_is_pct,
                                                    statistic_is_ratio,
                                                    is_log_scale=y_axis_log_scale)
                    ax.set_ylim(*y_axis_lim)
                
                colors = self._return_colors(facet_data,
                                             ColorPaletteAggregationLevel.GROUP,
                                             dataset_name,
                                             color_dict)
                
                for line, color in zip(ax.lines, colors):
                    line.set_color(color)
                
                ax.spines['top'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_TOP)
                ax.spines['bottom'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_LEFT)
                ax.spines['right'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_RIGHT)

            n_groups = data[DATASET_NAME_FIELD_NAME_STR].nunique()
            self._remove_inner_plot_elements_grid(g,
                                                  n_groups,
                                                  n_cols,
                                                  SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_X_AXIS_LABELS,
                                                  SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_Y_AXIS_LABELS,
                                                  SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_X_AXIS_TICKS,
                                                  SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_Y_AXIS_TICKS,
                                                  SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                                  SUMMARY_SEQUENCE_STATISTICS_REMOVE_INNER_Y_AXIS_TICK_LABELS)

            plt.tight_layout()
            title = SUMMARY_SEQUENCE_STATISTICS_PLOT_NAME + field.value
            self._save_figure(title)
            plt.show(g);

    @sequence_statistics_distribution_boxplot_per_group_per_dataset_decorator
    def plot_sequence_statistics_distribution_boxplot_per_group_per_dataset(self,
                                                                            include_unique_sequences: bool) -> None:

        if include_unique_sequences:
            sequence_stats_fields = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]
            sequence_statistics_per_group_per_dataset = self._return_sequence_statistics_distribution_non_unique_unique_split_per_group_per_dataset_df(sequence_stats_fields)

            fields_to_plot = SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST

        else:
            sequence_stats_fields = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]
            sequence_statistics_per_group_per_dataset = self._return_sequence_statistics_distribution_per_group_per_dataset_df(sequence_stats_fields,
                                                                                                                               False)

            unique_sequence_freq_stats_fields = [field.value for field in UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST]
            unique_sequence_frequency_statistics_per_group_per_dataset = self._return_sequence_statistics_distribution_per_group_per_dataset_df(unique_sequence_freq_stats_fields,
                                                                                                                                                True)

            fields_to_plot = SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST + UNIQUE_SEQUENCE_FREQUENCY_STATISTICS_FIELDS_TO_PLOT_LIST

        for field in fields_to_plot:

            print(STAR_STRING)
            print(field.value)
            print(STAR_STRING)

            if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                print(f'{BOXPLOT_IS_SORTED_STR}{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC.value}')
            else:
                print(f'{BOXPLOT_IS_SORTED_STR}{GROUP_FIELD_NAME_STR}{BOXPLOT_GROUP_NUMBER_STR}')
            print('')

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_RAW
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_RAW

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_PCT
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_PCT
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_PCT


                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_PCT_RATIO
                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_PCT_RATIO
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_PCT_RATIO
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_PCT_RATIO
                
                case UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_RAW
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_RAW

                case UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_PCT
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_PCT
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_PCT

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{UniqueSequenceFrequencyStatisticsPlotFields.__name__}')

            n_cols = set_facet_grid_column_number(data[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            color_dict = self._return_color_per_group_per_dataset(data,
                                                                  ColorPaletteAggregationLevel.GROUP,
                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_PALETTE,
                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_ALPHA,
                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_COLOR_SATURATION)

            def plot_boxplot(data, **kwargs):

                if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                    data = self._sort_groups_by_metric(data,
                                                       field,
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC,
                                                       SequenceStatisticsDistributionSortingEntity.GROUP,
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_ASCENDING)

                if include_unique_sequences:
                    hue_var = LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR
                else:
                    hue_var = GROUP_FIELD_NAME_STR

                sns.boxplot(
                            data=data, 
                            x=field.value,
                            y=GROUP_FIELD_NAME_STR,
                            hue=hue_var,
                            orient=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_ORIENTATION,
                            palette=RESULT_AGGREGATION_COLOR_PALETTE,
                            showfliers=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_OUTLIERS,
                            linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LINE_WIDTH,
                            width=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WIDTH,
                            whis=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_BOX_WHISKERS,
                            showmeans=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_MEANS,
                            meanprops=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_MARKER,
                            flierprops=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_OUTLIER_MARKER,
                            **kwargs)

            g = sns.FacetGrid(data,
                              col=DATASET_NAME_FIELD_NAME_STR,
                              col_wrap=n_cols,
                              height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                              aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                              sharex=share_x,
                              sharey=share_y)

            g.map_dataframe(plot_boxplot)

            g.set_titles('{col_name}') 

            if share_x:
                x_axis_lim = return_axis_limits(data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio,
                                                is_log_scale=x_axis_log_scale)
                g.set(xlim=x_axis_lim)

            if include_unique_sequences:
                g.add_legend(
                            title=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                            frameon=True,
                            bbox_to_anchor=(0.98, 0.5), 
                            loc='center left')

            for ax, (dataset_name, facet_data) in zip(g.axes.flat, data.groupby(DATASET_NAME_FIELD_NAME_STR)):

                self._set_axis_labels(ax,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_X_AXIS_LABEL,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_Y_AXIS_LABEL,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_AXIS_LABEL,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_Y_AXIS_LABEL)

                self._set_axis_ticks(ax,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_X_AXIS_TICKS,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_Y_AXIS_TICKS,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_X_AXIS_TICK_LABELS,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_Y_AXIS_TICK_LABELS,
                                     x_axis_ticks,
                                     None)

                if x_axis_log_scale:
                    self._set_log_scale_axes(ax,
                                             Axes.X_AXIS)
                ax.grid(True,
                        axis=SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_GRID_LINE_AXIS.value,
                        which='both')

                if not share_x:
                    x_axis_lim = return_axis_limits(facet_data[field.value],
                                                    statistic_is_pct,
                                                    statistic_is_ratio,
                                                    is_log_scale=x_axis_log_scale)
                    ax.set_xlim(*x_axis_lim)

                if include_unique_sequences:
                    pass
                else:

                    if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:

                        facet_data = self._sort_groups_by_metric(facet_data,
                                                                 field,
                                                                 SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC,
                                                                 SequenceStatisticsDistributionSortingEntity.GROUP,
                                                                 SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_ASCENDING)
                    colors = self._return_colors(facet_data,
                                                 ColorPaletteAggregationLevel.GROUP,
                                                 dataset_name,
                                                 color_dict)

                    for box, col in zip(ax.patches, colors):
                        box.set_facecolor(col)

                ax.spines['top'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_TOP)
                ax.spines['bottom'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_LEFT)
                ax.spines['right'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHOW_RIGHT)

            n_groups = data[DATASET_NAME_FIELD_NAME_STR].nunique()
            self._remove_inner_plot_elements_grid(g,
                                                  n_groups,
                                                  n_cols,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_X_AXIS_LABELS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_Y_AXIS_LABELS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_X_AXIS_TICKS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_Y_AXIS_TICKS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS)
            
            plt.tight_layout()
            if include_unique_sequences:
                title = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_NON_UNIQUE_UNIQUE_SPLIT_PLOT_NAME
            else:
                title = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_NAME
            title += field.value
            self._save_figure(title)
            plt.show(g);

    @sequence_statistics_distribution_ridgeplot_per_group_per_dataset_decorator
    def plot_sequence_statistics_distribution_ridgeplot_per_group_per_dataset(self) -> None:

        sequence_stats_fields = [field.value for field in SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST]
        sequence_statistics_per_group_per_dataset = self._return_sequence_statistics_distribution_per_group_per_dataset_df(sequence_stats_fields,
                                                                                                                           False)

        fields_to_plot = SEQUENCE_STATISTICS_FIELDS_TO_PLOT_LIST

        for field in fields_to_plot:

            print(STAR_STRING)
            print(field.value)
            print(STAR_STRING)

            if SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_BOXES:
                print(f'{RIDGEPLOT_IS_SORTED_STR}{SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_METRIC.value}')
            else:
                print(f'{RIDGEPLOT_IS_SORTED_STR}{GROUP_FIELD_NAME_STR}{RIDGEPLOT_GROUP_NUMBER_STR}')
            print('')

            match field:
                case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_RAW
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_RAW

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_PCT
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_PCT
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_PCT

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_PCT_RATIO
                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_PCT_RATIO
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_PCT_RATIO
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_PCT_RATIO

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SequenceStatisticsPlotFields.__name__}')

            n_cols = set_facet_grid_column_number(data[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            color_dict = self._return_color_per_group_per_dataset(data,
                                                                  ColorPaletteAggregationLevel.GROUP,
                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_COLOR_PALETTE,
                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_COLOR_ALPHA,
                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_COLOR_SATURATION)

            def plot_ridgeplot(data, **kwargs):

                dataset_name = data[DATASET_NAME_FIELD_NAME_STR].iloc[0]

                if SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_BOXES:
                    data = self._sort_groups_by_metric(data,
                                                       field,
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_METRIC,
                                                       SequenceStatisticsDistributionSortingEntity.GROUP,
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_ASCENDING)

                colors = self._return_colors(data,
                                             ColorPaletteAggregationLevel.GROUP,
                                             dataset_name,
                                             color_dict)

                n_groups = data[GROUP_FIELD_NAME_STR].nunique()
                shift_value = 0

                y_axis_ticks = np.arange(0, 
                                         -n_groups*SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT, 
                                         -SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT)
                y_axis_ticks_labels = data[GROUP_FIELD_NAME_STR].unique()

                zorder = 10000

                for (group, df), color in zip(data.groupby(GROUP_FIELD_NAME_STR), colors): 

                    # conf int
                    mean_value = df[field.value].mean()
                    confidence = 0.95
                    n = len(df)
                    mean_value = df[field.value].mean()
                    std_err = sp.stats.sem(df[field.value])  # Standard error: std / sqrt(n)
                    deg_fred = n - 1
                    ci = sp.stats.t.interval(confidence, deg_fred, loc=mean_value, scale=std_err)
                    ci_lower = ci[0] 
                    ci_upper = ci[1]
                    #

                    x_min = min(df[field.value])
                    x_max = max(df[field.value])
                    x_lower_bound = x_min - 0
                    x_upper_bound = x_max + 0
                    kde = sp.stats.gaussian_kde(df[field.value], 
                                                bw_method=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BANDWIDTH_METHOD)
                    x_vals = np.linspace(x_lower_bound, 
                                         x_upper_bound, 
                                         10000)
                    y_vals = kde(x_vals)
                    y_vals = y_vals / np.max(y_vals) # normalization to density of 1: y_vals = y_vals / np.trapz(y_vals, x_vals)

                    # ensure that borders of the densities are drawn at the min and max value of the data
                    y_vals[0] = 0
                    y_vals[-1] = 0
                    
                    # shift the y values for correct position on y axis
                    y_vals_shifted = y_vals + shift_value

                    # kde
                    sns.lineplot(x=x_vals, 
                                 y=y_vals_shifted,
                                 color=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_COLOR,
                                 linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_LINEWIDTH,
                                 zorder=zorder)
                    sns.lineplot(x=x_vals, 
                                 y=y_vals_shifted,
                                 color=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_COLOR,
                                 linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_LINEWIDTH,
                                 zorder=zorder+0.5)
                    if SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INCLUDE_KDE_BOTTOM_LINE:
                        sns.lineplot(x=x_vals, 
                                     y=np.zeros_like(x_vals) + shift_value,
                                     color=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_COLOR,
                                     linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_LINEWIDTH,
                                     zorder=zorder+0.5)
                    plt.fill_between(x_vals, 
                                     y_vals_shifted, 
                                     shift_value, 
                                     color=color, 
                                     zorder=zorder)

                    # conf int
                    sns.lineplot(x=[ci_lower, ci_upper],
                                 y=[shift_value, shift_value],
                                 legend=False,
                                 color=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_OUTER_LINEPLOT_COLOR,
                                 linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_OUTER_LINEPLOT_LINEWIDTH,
                                 zorder=100_000)
                    sns.lineplot(x=[ci_lower, ci_upper],
                                 y=[shift_value, shift_value],
                                 legend=False,
                                 color=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_INNER_LINEPLOT_COLOR,
                                 linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_INNER_LINEPLOT_LINEWIDTH,
                                 zorder=100_001)

                    sns.scatterplot(x=[mean_value],
                                    y=[shift_value],
                                    legend=False,
                                    color=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SCATTER_COLOR,
                                    s=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SCATTER_SIZE,
                                    linewidth=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SCATTER_LINEWIDTH,
                                    edgecolor=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SCATTER_EDGECOLOR,
                                    zorder=100_002)

                    shift_value -= SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT 
                    zorder += 1

                plt.yticks(y_axis_ticks, 
                           labels=y_axis_ticks_labels)

            g = sns.FacetGrid(data,
                              col=DATASET_NAME_FIELD_NAME_STR,
                              col_wrap=n_cols,
                              height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                              aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                              sharex=share_x,
                              sharey=share_y)

            g.map_dataframe(plot_ridgeplot)

            g.set_titles('{col_name}') 

            if share_x:
                x_axis_lim = return_axis_limits(data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio,
                                                is_log_scale=x_axis_log_scale)
                g.set(xlim=x_axis_lim)

            for ax, (dataset_name, facet_data) in zip(g.axes.flat, data.groupby(DATASET_NAME_FIELD_NAME_STR)):

                self._set_axis_labels(ax,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_X_AXIS_LABEL,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_Y_AXIS_LABEL,
                                      field.value,
                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_LABEL)

                self._set_axis_ticks(ax,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_X_AXIS_TICKS,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_Y_AXIS_TICKS,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_X_AXIS_TICK_LABELS,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_Y_AXIS_TICK_LABELS,
                                     x_axis_ticks,
                                     None)

                if x_axis_log_scale:
                    self._set_log_scale_axes(ax,
                                             Axes.X_AXIS)
                ax.grid(True,
                        axis=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_GRID_LINE_AXIS.value,
                        which='both')

                if not share_x:
                    x_axis_lim = return_axis_limits(facet_data[field.value],
                                                    statistic_is_pct,
                                                    statistic_is_ratio,
                                                    is_log_scale=x_axis_log_scale)
                    ax.set_xlim(*x_axis_lim)

                ax.spines['top'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_TOP)
                ax.spines['bottom'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_LEFT)
                ax.spines['right'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHOW_RIGHT)

            n_groups = data[DATASET_NAME_FIELD_NAME_STR].nunique()
            self._remove_inner_plot_elements_grid(g,
                                                  n_groups,
                                                  n_cols,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_X_AXIS_LABELS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_Y_AXIS_LABELS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_X_AXIS_TICKS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_Y_AXIS_TICKS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS)

            plt.tight_layout()
            title = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_PLOT_NAME + field.value
            self._save_figure(title)
            plt.show(g);

    @sequence_count_per_group_per_dataset_decorator
    def plot_sequence_count_per_group_per_dataset(self) -> None:

        sequence_count_per_group_per_dataset = self._return_sequence_count_per_group_per_dataset_df()

        print(STAR_STRING)
        print(f'{LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR} against {LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR}')
        print(STAR_STRING)

        n_cols = set_facet_grid_column_number(sequence_count_per_group_per_dataset[DATASET_NAME_FIELD_NAME_STR],
                                              RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

        color_dict = self._return_color_per_group_per_dataset(sequence_count_per_group_per_dataset,
                                                              ColorPaletteAggregationLevel.GROUP,
                                                              SEQUENCE_COUNT_COLOR_PALETTE,
                                                              SEQUENCE_COUNT_COLOR_ALPHA,
                                                              SEQUENCE_COUNT_COLOR_SATURATION)

        g = sns.relplot(sequence_count_per_group_per_dataset,
                        x=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
                        y=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
                        hue=GROUP_FIELD_NAME_STR,
                        col=DATASET_NAME_FIELD_NAME_STR,
                        col_wrap=n_cols,
                        height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                        aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                        kind=SEQUENCE_COUNT_KIND,
                        legend=SEQUENCE_COUNT_PLOT_LEGEND,
                        s=SEQUENCE_COUNT_MARKER_SIZE,
                        facecolor=SEQUENCE_COUNT_MARKER_FACECOLOR,
                        edgecolor=SEQUENCE_COUNT_MARKER_EDGECOLOR,
                        linewidth=SEQUENCE_COUNT_MARKER_EDGEWIDTH,
                        zorder=101,
                        facet_kws=dict(sharex=SEQUENCE_COUNT_SHAREX,
                                       sharey=SEQUENCE_COUNT_SHAREX)
                    )

        g.set_titles('{col_name}') 

        if SEQUENCE_COUNT_SHAREX:
            x_axis_lim = return_axis_limits(sequence_count_per_group_per_dataset[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR],
                                            False,
                                            False,
                                            is_log_scale=SEQUENCE_COUNT_LOG_SCALE_X_RAW)
            g.set(xlim=x_axis_lim,
                  ylim=x_axis_lim)


        for ax, (dataset_name, facet_data) in zip(g.axes.flat, sequence_count_per_group_per_dataset.groupby(DATASET_NAME_FIELD_NAME_STR)):

            self._set_axis_labels(ax,
                                  SEQUENCE_COUNT_PLOT_X_AXIS_LABEL,
                                  SEQUENCE_COUNT_PLOT_Y_AXIS_LABEL,
                                  SEQUENCE_COUNT_X_AXIS_LABEL,
                                  SEQUENCE_COUNT_Y_AXIS_LABEL)

            self._set_axis_ticks(ax,
                                 SEQUENCE_COUNT_PLOT_X_AXIS_TICKS,
                                 SEQUENCE_COUNT_PLOT_Y_AXIS_TICKS,
                                 SEQUENCE_COUNT_PLOT_X_AXIS_TICK_LABELS,
                                 SEQUENCE_COUNT_PLOT_Y_AXIS_TICK_LABELS,
                                 SEQUENCE_COUNT_X_TICKS_RAW,
                                 SEQUENCE_COUNT_Y_TICKS_RAW)

            if not SEQUENCE_COUNT_SHAREX:
                x_axis_lim = return_axis_limits(facet_data[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR],
                                                False,
                                                False,
                                                is_log_scale=SEQUENCE_COUNT_LOG_SCALE_X_RAW)
                ax.set_xlim(*x_axis_lim)
                ax.set_ylim(*x_axis_lim)

            ax.axline(xy1=(0,0), 
                      slope=1, 
                      color=SEQUENCE_COUNT_45_DEGREE_LINE_COLOR, 
                      linewidth=SEQUENCE_COUNT_45_DEGREE_LINE_WIDTH, 
                      zorder=100)

            colors = self._return_colors(facet_data,
                                         ColorPaletteAggregationLevel.GROUP,
                                         dataset_name,
                                         color_dict)

            collection = ax.collections[0]
            collection.set_facecolor(colors)

            ax.spines['top'].set_visible(SEQUENCE_COUNT_SHOW_TOP)
            ax.spines['bottom'].set_visible(SEQUENCE_COUNT_SHOW_BOTTOM)
            ax.spines['left'].set_visible(SEQUENCE_COUNT_SHOW_LEFT)
            ax.spines['right'].set_visible(SEQUENCE_COUNT_SHOW_RIGHT)

        for ax in g.axes.flat:

            if SEQUENCE_COUNT_LOG_SCALE_X_RAW:
                self._set_log_scale_axes(ax,
                                         Axes.BOTH)
            ax.grid(True,
                    axis=SEQUENCE_COUNT_GRID_LINE_AXIS.value,
                    which='both')

        n_groups = sequence_count_per_group_per_dataset[DATASET_NAME_FIELD_NAME_STR].nunique()
        self._remove_inner_plot_elements_grid(g,
                                              n_groups,
                                              n_cols,
                                              SEQUENCE_COUNT_REMOVE_INNER_X_AXIS_LABELS,
                                              SEQUENCE_COUNT_REMOVE_INNER_Y_AXIS_LABELS,
                                              SEQUENCE_COUNT_REMOVE_INNER_X_AXIS_TICKS,
                                              SEQUENCE_COUNT_REMOVE_INNER_Y_AXIS_TICKS,
                                              SEQUENCE_COUNT_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                              SEQUENCE_COUNT_REMOVE_INNER_Y_AXIS_TICK_LABELS)

        plt.tight_layout()
        self._save_figure(SEQUENCE_COUNT_PLOT_NAME)
        plt.show(g);

    @aggregated_omnibus_test_result_per_dataset_stacked_barplot_decorator
    def plot_aggregated_omnibus_test_result_per_dataset_stacked_barplot(self) -> None:

        # plotting data 
        pct_df, count_df = self._return_aggregated_omnibus_test_result_per_dataset_plotting_pct_df_count_df()

        # use fields name appropriate for plotting
        pct_df.columns = OMNIBUS_TEST_RESULT_STACKED_BARPLOT_GROUP_CATEGORIES
        count_df.columns = OMNIBUS_TEST_RESULT_STACKED_BARPLOT_GROUP_CATEGORIES

        pct_df = pct_df.sort_index(ascending=False)
        count_df = count_df.sort_index(ascending=False)
        pct_matrix = pct_df.to_numpy()
        count_matrix = count_df.to_numpy()

        x_axis_lim = return_axis_limits(None,
                                        True,
                                        False)
        x_axis_ticks = np.arange(0, 110, 10)

        # color palette
        index = OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_INDEX
        color_palette = sns.color_palette(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE,
                                          n_colors=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_NUMBER_COLORS,
                                          desat=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_DESAT)
        color_palette = [color_palette[i] for i in index]
        color_palette.insert(0, OMNIBUS_TEST_RESULT_STACKED_BARPLOT_NON_SIG_COLOR)
        cmap = ListedColormap(color_palette)

        # plot
        ax = pct_df.plot.barh(stacked=True, 
                              colormap=cmap,
                              edgecolor=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_EDGECOLOR,
                              linewidth=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_LINEWIDTH,
                              width=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_WIDTH,
                              alpha=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_ALPHA)

        # pct/count text
        pct_matrix = pct_df.to_numpy().transpose().flatten()
        count_matrix = count_matrix.transpose().flatten()
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            height = p.get_height()
            x_pos = p.get_x() + width / 2
            y_pos = p.get_y() + height / 2

            pct = pct_matrix[i] 
            count = count_matrix[i]

            if pct >= OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_THRESHOLD:
                ax.text(x_pos, 
                        y_pos, 
                        f'{pct:.1f}% (n={count})', 
                        ha=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_H_POS, 
                        va=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_V_POS, 
                        fontsize=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_FONTSIZE, 
                        color=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_ANNOTATION_TEXT_COLOR)

        # settings
        plt.grid(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_GRID_LINES_VERTICAL, 
                 axis='x')
        plt.grid(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_GRID_LINES_HORIZONTAL, 
                 axis='y')
        plt.xlim(x_axis_lim)
        plt.xticks(x_axis_ticks)
        plt.xlabel(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_LABEL)
        plt.ylabel('')
        plt.legend(bbox_to_anchor=(0.5, 1.02),
                   loc='lower center',
                   borderaxespad=0,
                   frameon=False,
                   ncol=pct_df.shape[1])
        plt.tight_layout()
        self._save_figure(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_NAME)
        plt.show(ax);

    @aggregated_omnibus_test_result_per_dataset_grouped_barplot_decorator
    def plot_aggregated_omnibus_test_result_per_dataset_grouped_barplot(self) -> None:

        # plotting data 
        pct_df, count_df = self._return_aggregated_omnibus_test_result_per_dataset_plotting_pct_df_count_df()

        # use fields name appropriate for plotting
        pct_df.columns = OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_GROUP_CATEGORIES
        count_df.columns = OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_GROUP_CATEGORIES

        pct_df_long = pd.melt(pct_df.reset_index(), 
                              id_vars=DATASET_NAME_FIELD_NAME_STR, 
                              var_name=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR,
                              value_name=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_PCT_FIELD_NAME_STR)
        pct_df_long = pct_df_long.sort_values(by=DATASET_NAME_FIELD_NAME_STR)

        count_df_long = pd.melt(count_df.reset_index(), 
                                id_vars=DATASET_NAME_FIELD_NAME_STR, 
                                var_name=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR,
                                value_name=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_COUNT_FIELD_NAME_STR)
        count_df_long = count_df_long.sort_values(by=DATASET_NAME_FIELD_NAME_STR)

        pct_count_df_long = pd.concat([pct_df_long[[DATASET_NAME_FIELD_NAME_STR, RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_PCT_FIELD_NAME_STR]],
                                       count_df_long[[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_COUNT_FIELD_NAME_STR]]],
                                       axis=1)

        y_axis_lim = return_axis_limits(None,
                                        True,
                                        False)
        y_axis_ticks = np.arange(0, 110, 10)

        # color palette
        index = OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_INDEX
        color_palette = sns.color_palette(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE,
                                          n_colors=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_NUMBER_COLORS,
                                          desat=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_DESAT)
        color_palette = [color_palette[i] for i in index]
        color_palette.insert(0, OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_NON_SIG_COLOR)

        n_cols = set_facet_grid_column_number(pct_df_long[DATASET_NAME_FIELD_NAME_STR],
                                              RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

        g = sns.catplot(data=pct_df_long,
                        x=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR,
                        y=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_PCT_FIELD_NAME_STR,
                        hue=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR,
                        col=DATASET_NAME_FIELD_NAME_STR,
                        col_wrap=n_cols,
                        height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                        aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                        legend=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_LEGEND,
                        orient=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ORIENTATION,
                        kind=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_KIND,
                        palette=color_palette,
                        alpha=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ALPHA,
                        width=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_WIDTH,
                        edgecolor=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_EDGECOLOR,
                        linewidth=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_EDGEWIDTH,
                        sharex=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHAREX,
                        sharey=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHAREY)

        g.set_titles('{col_name}') 

        g.set(ylim=y_axis_lim,
              yticks=y_axis_ticks,
              ylabel=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_Y_LABEL,
              xlabel=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_X_LABEL)
        
        g.set_xticklabels(rotation=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_X_TICKS_ROTATION)

        for ax, (df_name, df) in zip(g.axes.flat, pct_count_df_long.groupby(DATASET_NAME_FIELD_NAME_STR)):

            # set spines
            ax.spines['top'].set_visible(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_TOP)
            ax.spines['bottom'].set_visible(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_BOTTOM)
            ax.spines['left'].set_visible(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_LEFT)
            ax.spines['right'].set_visible(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHOW_RIGHT)


            # add annotiation text
            pct_array = df[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_PCT_FIELD_NAME_STR].to_numpy()
            count_array = df[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_COUNT_FIELD_NAME_STR].to_numpy()
            for i, p in enumerate(ax.patches):
                if p.get_height() > 0:
                    width = p.get_width()
                    height = p.get_height()
                    x_pos = p.get_x() + width / 2
                    y_pos = p.get_y() + height / 2

                    pct = pct_array[i]
                    count = count_array[i]

                    if pct >= OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_THRESHOLD:
                        ax.text(x_pos, 
                                y_pos, 
                                f'{pct:.1f}%\n(n={count})', 
                                ha=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_H_POS, 
                                va=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_V_POS, 
                                fontsize=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_FONTSIZE, 
                                color=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ANNOTATION_TEXT_COLOR)

        # legend
        sns.move_legend(g, 
                        "lower center",
                        bbox_to_anchor=(.5, 1), 
                        title=None, 
                        frameon=False, 
                        ncol=pct_df_long[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR].nunique())

        #TODO: adjust
        # for handle in g.legend.legendHandles:  
        #     handle.set_height(15)
        #     handle.set_width(30)
        #     handle.set_edgecolor("black")
        #     handle.set_linewidth(1.5) 
        
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.tight_layout()
        self._save_figure(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_NAME)
        plt.show(g);

    def display_summary_statistics_result(self,
                                          apply_table_styling: bool) -> None:
        """Displays the summary statistics result html table.
        """
        summary_statistics_result_df = self._return_summary_statistics_result_df()
        summary_statistics_result_df = summary_statistics_result_df.set_index(DATASET_NAME_FIELD_NAME_STR)
        summary_statistics_result_df.index.name = None

        soup = BeautifulSoup(summary_statistics_result_df.to_html(), "html.parser")

        if apply_table_styling:
            # standardized table formatting
            apply_html_table_formatting(soup)
        
        display(Markdown(soup.prettify()))

    def display_sequence_summary_statistics_result(self,
                                                   apply_table_styling: bool) -> None:
        """Displays the sequence summary statistics result html table.
        """
        sequence_summary_statistics_result_df = self._return_sequence_summary_statistics_result_df()
        sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.set_index(DATASET_NAME_FIELD_NAME_STR)
        sequence_summary_statistics_result_df.index.name = None

        soup = BeautifulSoup(sequence_summary_statistics_result_df.to_html(), "html.parser")

        if apply_table_styling:
            # standardized table formatting
            apply_html_table_formatting(soup)
        
        display(Markdown(soup.prettify()))

    def display_available_fields_result(self,
                                        apply_table_styling: bool) -> None:
        """Displays the available fields result html table.
        """
        available_fields_result_df = self._return_available_fields_result_df()

        available_fields_df = available_fields_result_df.copy()
        available_fields_df = available_fields_df.set_index(FIELD_STR)

        soup = BeautifulSoup(available_fields_df.to_html(), "html.parser")

        # move Field header value to correct position
        th_elements = soup.select('thead th')
        for th in th_elements[::-1]:
            if th.string == FIELD_STR:
                th.string = ''
        th = soup.select_one('thead th')
        if th:
            th.append(FIELD_STR)
        
        remove_empty_html_table_row(soup)
    
        if apply_table_styling:
            # standardized table formatting
            apply_html_table_formatting(soup)

        display(Markdown(soup.prettify()))

    def display_score_is_correct_relationship_result(self,
                                                     apply_table_styling: bool) -> None:
        """Displays the sequence summary statistics result html table.
        """
        score_is_corrcet_relationship_result_df = self._return_score_is_correct_relationship_result_df()
        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.set_index([DATASET_NAME_FIELD_NAME_STR, FIELD_STR])
        score_is_corrcet_relationship_result_df.index.names = [None, None]

        score_is_corrcet_relationship_result_df = score_is_corrcet_relationship_result_df.copy()

        soup = BeautifulSoup(score_is_corrcet_relationship_result_df.to_html(), "html.parser")

        if apply_table_styling:
            # standardized table formatting
            apply_html_table_formatting(soup)

        display(Markdown(soup.prettify()))

    def display_aggregated_omnibus_test_per_dataset_result(self,
                                                           apply_table_styling: bool) -> None:

        aggregated_omnibus_test_result_per_dataset = self._return_aggregated_omnibus_test_result_per_dataset_df()
        aggregated_omnibus_test_result_per_dataset_display = self._return_aggregated_omnibus_test_result_per_dataset_df_display(aggregated_omnibus_test_result_per_dataset)

        aggregated_omnibus_test_result_per_dataset_display = aggregated_omnibus_test_result_per_dataset_display.set_index(DATASET_NAME_FIELD_NAME_STR)
        aggregated_omnibus_test_result_per_dataset_display.index.name = None

        soup = BeautifulSoup(aggregated_omnibus_test_result_per_dataset_display.to_html(), "html.parser")

        if apply_table_styling:
            # standardized table formatting
            apply_html_table_formatting(soup,
                                        style_booleans=False)
        
        display(Markdown(soup.prettify()))

    def print_latex_summary_statistics_result(self):
        """Prints the summary statistics result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            summary_statistics_result_df = self._return_summary_statistics_result_df()
            summary_statistics_result_df = summary_statistics_result_df.set_index(DATASET_NAME_FIELD_NAME_STR)
            summary_statistics_result_df.index.name = None
            print(summary_statistics_result_df.style.format(precision=2)
                                                    .format_index(axis=1, escape='latex')
                                                    .format_index(axis=0, escape='latex')
                                                    .to_latex()
                                                    .replace('<NA>', '-')
                                                    .replace('nan', '-'))

    def print_latex_sequence_summary_statistics_result(self):
        """Prints the sequence summary statistics result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            sequence_summary_statistics_result_df = self._return_sequence_summary_statistics_result_df()
            sequence_summary_statistics_result_df = sequence_summary_statistics_result_df.set_index(DATASET_NAME_FIELD_NAME_STR)
            sequence_summary_statistics_result_df.index.name = None
            print(sequence_summary_statistics_result_df.style.format(precision=2)
                                                    .format_index(axis=1, escape='latex')
                                                    .format_index(axis=0, escape='latex')
                                                    .to_latex()
                                                    .replace('<NA>', '-')
                                                    .replace('nan', '-'))

    def print_latex_available_fields_result(self):
        """Prints the available fields result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            available_fields_result_df = self._return_available_fields_result_df()

            available_fields_result_df = available_fields_result_df.copy()
            available_fields_result_df = available_fields_result_df.set_index(FIELD_STR)
            print(available_fields_result_df.style.format_index(axis=1, escape='latex')
                                                    .format_index(axis=0, escape='latex')
                                    #  .hide(axis=0)
                                     .to_latex()
                                     .replace('\multicolumn{4}{r}', '\multicolumn{4}{c}'))

    def print_latex_score_is_correct_relationship_result(self):
        """Prints the score is_correct relationship result latex table.
        """
        with pd.option_context('max_colwidth', 1000):
            package_str = '\\usepackage{multirow} \n'
            score_is_correct_relationship_df = self._return_score_is_correct_relationship_result_df()
            score_is_correct_relationship_df = score_is_correct_relationship_df.set_index([DATASET_NAME_FIELD_NAME_STR, FIELD_STR])
            score_is_correct_relationship_df.index.names = [None, None]
            print(package_str + score_is_correct_relationship_df.style.format(precision=2)
                                                                      .format_index(axis=1, escape='latex')
                                                                      .format_index(axis=0, escape='latex')
                                                                      .to_latex()
                                                                      .replace('<NA>', '-')
                                                                      .replace('nan', '-'))

    def print_latex_aggregated_omnibus_test_result_per_dataset(self) -> None:
        """Prints the summary statistics result latex table.
        """
        with pd.option_context('max_colwidth', 1000):

            aggregated_omnibus_test_result_per_dataset = self._return_aggregated_omnibus_test_result_per_dataset_df()
            aggregated_omnibus_test_result_per_dataset_display = self._return_aggregated_omnibus_test_result_per_dataset_df_display(aggregated_omnibus_test_result_per_dataset)

            aggregated_omnibus_test_result_per_dataset_display = aggregated_omnibus_test_result_per_dataset_display.set_index(DATASET_NAME_FIELD_NAME_STR)
            aggregated_omnibus_test_result_per_dataset_display.index.name = None
            print(aggregated_omnibus_test_result_per_dataset_display.style.format(precision=2)
                                                    .format_index(axis=1, escape='latex')
                                                    .format_index(axis=0, escape='latex')
                                                    .to_latex()
                                                    .replace('<NA>', '-')
                                                    .replace('nan', '-'))

    def _return_result_tables_paths(self) -> list[str]:

        path_to_result_tables_dir = Path(PATH_TO_PICKLED_OBJECTS_FOLDER) / PATH_TO_RESULT_TABLES_PICKLE_FOLDER
        extension = '.pickle'
        path_to_result_tables = [file for file in path_to_result_tables_dir.rglob(f'*{extension}')]

        return path_to_result_tables
    
    def _return_avg_sequence_stats_per_group_per_dataset_df(self,
                                                            fields_to_plot: list[str],
                                                            use_unique_sequences: bool) -> pd.DataFrame:

        field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, *fields_to_plot]

        avg_learning_activity_sequence_stats_per_group_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                if use_unique_sequences:
                    learning_activity_sequence_stats_per_group = pickle.load(f).unique_learning_activity_sequence_stats_per_group[field_list]
                else:
                    learning_activity_sequence_stats_per_group = pickle.load(f).learning_activity_sequence_stats_per_group[field_list]
            
            avg_learning_activity_sequence_stats_per_group = (learning_activity_sequence_stats_per_group
                                                              .groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])
                                                              [fields_to_plot].agg(AVG_SEQUENCE_STATISTICS_AVERAGING_METHOD.value))

            avg_learning_activity_sequence_stats_per_group = avg_learning_activity_sequence_stats_per_group.reset_index()

            avg_learning_activity_sequence_stats_per_group_df_list.append(avg_learning_activity_sequence_stats_per_group)

        avg_learning_activity_sequence_stats_per_group_per_dataset = pd.concat(avg_learning_activity_sequence_stats_per_group_df_list)
        avg_learning_activity_sequence_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)
    
        return avg_learning_activity_sequence_stats_per_group_per_dataset 

    def _return_summary_sequence_stats_per_group_per_dataset_df(self,
                                                                fields_to_plot: list[str],
                                                                use_unique_sequences: bool) -> pd.DataFrame:

        # helper functions for quartiles 
        def first_quartile(array):
            return np.quantile(array, 0.25)
        def third_quartile(array):
            return np.quantile(array, 0.75)

        field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, *fields_to_plot]

        summary_stats_per_group_list= []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                if use_unique_sequences:
                    learning_activity_sequence_stats_per_group = pickle.load(f).unique_learning_activity_sequence_stats_per_group[field_list]
                else:
                    learning_activity_sequence_stats_per_group = pickle.load(f).learning_activity_sequence_stats_per_group[field_list]

                summary_statistic_per_field_long_list = []
                for field in fields_to_plot:

                    summary_statistic_per_field = (learning_activity_sequence_stats_per_group
                                                   .groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])[field]
                                                   .agg([min, max, np.median, first_quartile, third_quartile])
                                                   .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                    'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                    'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                                    'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                    'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})
                                                   .reset_index())

                    summary_field_list = [DATASET_NAME_FIELD_NAME_STR, 
                                          GROUP_FIELD_NAME_STR, 
                                          LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                          LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                          LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                          LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                          LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]

                    summary_statistic_per_field_long = pd.melt(summary_statistic_per_field[summary_field_list], 
                                                               id_vars=[DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR],
                                                               var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                               value_name=field)
                    sort_list = [DATASET_NAME_FIELD_NAME_STR, 
                                 GROUP_FIELD_NAME_STR]
                    summary_statistic_per_field_long.sort_values(by=sort_list,
                                                                 inplace=True)

                    summary_statistic_per_field_long_list.append(summary_statistic_per_field_long)
            
            merge_field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR]
            summary_stats_per_group = reduce(lambda left, right: pd.merge(left=left, 
                                                                          right=right, 
                                                                          how='inner', 
                                                                          on=merge_field_list), summary_statistic_per_field_long_list)
            summary_stats_per_group_list.append(summary_stats_per_group)

        summary_stats_per_group_per_dataset = pd.concat(summary_stats_per_group_list)
        summary_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)

        return summary_stats_per_group_per_dataset

    def _return_sequence_statistics_distribution_per_group_per_dataset_df(self,
                                                                          fields_to_plot: list[str],
                                                                          use_unique_sequences: bool) -> pd.DataFrame:

        field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, *fields_to_plot]

        learning_activity_sequence_stats_per_group_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                if use_unique_sequences:
                    learning_activity_sequence_stats_per_group = pickle.load(f).unique_learning_activity_sequence_stats_per_group[field_list]
                else:
                    learning_activity_sequence_stats_per_group = pickle.load(f).learning_activity_sequence_stats_per_group[field_list]
            
            learning_activity_sequence_stats_per_group_df_list.append(learning_activity_sequence_stats_per_group)

        summary_stats_per_group_per_dataset = pd.concat(learning_activity_sequence_stats_per_group_df_list)
        summary_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)

        return summary_stats_per_group_per_dataset

    def _return_sequence_statistics_distribution_non_unique_unique_split_per_group_per_dataset_df(self,
                                                                                                  fields_to_plot: list[str]) -> pd.DataFrame:

        field_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR, *fields_to_plot]

        learning_activity_sequence_stats_per_group_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                result_tables = pickle.load(f)
                learning_activity_sequence_stats_per_group = result_tables.learning_activity_sequence_stats_per_group[field_list]
                unique_learning_activity_sequence_stats_per_group = result_tables.unique_learning_activity_sequence_stats_per_group[field_list]
            
            learning_activity_sequence_stats_per_group_df_list.extend([learning_activity_sequence_stats_per_group,
                                                                       unique_learning_activity_sequence_stats_per_group])

        summary_stats_per_group_per_dataset = pd.concat(learning_activity_sequence_stats_per_group_df_list)
        sort_list = [DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR]
        summary_stats_per_group_per_dataset.sort_values(sort_list, inplace=True)

        return summary_stats_per_group_per_dataset

    def _return_sequence_count_per_group_per_dataset_df(self) -> pd.DataFrame:

        field_list = [DATASET_NAME_FIELD_NAME_STR, 
                      GROUP_FIELD_NAME_STR, 
                      LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
                      LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR]
        
        sequence_count_per_group_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                unique_learning_activity_sequence_stats_per_group = pickle.load(f).unique_learning_activity_sequence_stats_per_group[field_list]

            sequence_count_per_group = (unique_learning_activity_sequence_stats_per_group.groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])[field_list]
                                                                                         .head(1)
                                                                                         .reset_index(drop=True))
            
            sequence_count_per_group_df_list.append(sequence_count_per_group)

        sequence_count_per_group_per_dataset = pd.concat(sequence_count_per_group_df_list)
        sequence_count_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)

        return sequence_count_per_group_per_dataset

    def _return_summary_statistics_result_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains summary statistics of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing summary statistics of the analysed datasets
        """        
        summary_statistics_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                summary_statistics_df = pickle.load(f).summary_statistics_df

            summary_statistics_df_list.append(summary_statistics_df)
            
        summary_statistics_per_dataset_df = pd.concat(summary_statistics_df_list,
                                                      axis=1,
                                                      join='inner')

        summary_statistics_per_dataset_df = summary_statistics_per_dataset_df.transpose()

        # typecast to integer type which also can take on NAs
        idx = [N_ROWS_STR,
               N_UNIQUE_USERS_STR, 
               N_UNIQUE_GROUPS_STR, 
               N_UNIQUE_LEARNING_ACTIVITIES_STR]
        typecast_dict = {i: 'Int64' for i in idx}
        summary_statistics_per_dataset_df = summary_statistics_per_dataset_df.astype(typecast_dict)
        summary_statistics_per_dataset_df = summary_statistics_per_dataset_df.reset_index(names=DATASET_NAME_FIELD_NAME_STR)
        summary_statistics_per_dataset_df = summary_statistics_per_dataset_df.sort_values(by=DATASET_NAME_FIELD_NAME_STR)

        return summary_statistics_per_dataset_df

    def _return_sequence_summary_statistics_result_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains sequence summary statistics of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing sequence summary statistics of the analysed datasets
        """        
        sequence_summary_statistics_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                sequence_summary_statistics_df = pickle.load(f).sequence_statistics_df

            sequence_summary_statistics_df_list.append(sequence_summary_statistics_df)

        sequence_summary_statistics_per_dataset_df = pd.concat(sequence_summary_statistics_df_list,
                                                               axis=1,
                                                               join='inner')

        sequence_summary_statistics_per_dataset_df = sequence_summary_statistics_per_dataset_df.transpose()

        # typecast to integer type which also can take on NAs
        idx = [N_SEQUENCES_STR,
               N_UNIQUE_SEQUENCES_STR]
        typecast_dict = {i: 'Int64' for i in idx}
        sequence_summary_statistics_per_dataset_df = sequence_summary_statistics_per_dataset_df.astype(typecast_dict)
        sequence_summary_statistics_per_dataset_df = sequence_summary_statistics_per_dataset_df.reset_index(names=DATASET_NAME_FIELD_NAME_STR)
        sequence_summary_statistics_per_dataset_df = sequence_summary_statistics_per_dataset_df.sort_values(by=DATASET_NAME_FIELD_NAME_STR,
                                                                                                            ignore_index=True)

        return sequence_summary_statistics_per_dataset_df

    def _return_available_fields_result_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains information about field availability of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing field availability information about the analysed datasets
        """        
        available_fields_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                available_fields_df = pickle.load(f).available_fields_df

            available_fields_df_list.append(available_fields_df)

        def set_index(df: pd.DataFrame) -> pd.DataFrame:
            df = df.set_index(FIELD_STR)
            df.columns = df.columns.droplevel(0)
            return df

        available_fields_df_list = map(set_index, available_fields_df_list)

        availbable_fields_per_dataset_df = pd.concat(available_fields_df_list,
                                                     axis=1,
                                                     join='inner')
        availbable_fields_per_dataset_df = availbable_fields_per_dataset_df.sort_index(axis=1)

        availbable_fields_per_dataset_df.columns = pd.MultiIndex.from_product([[IS_AVAILABLE_STR], availbable_fields_per_dataset_df.columns])
        availbable_fields_per_dataset_df = availbable_fields_per_dataset_df.reset_index()

        return availbable_fields_per_dataset_df

    def _return_score_is_correct_relationship_result_df(self) -> pd.DataFrame:
        """Returns a dataframe which contains information about the relationship between the score and the is_correct fields of the analysed datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing information about the relationship between the score and the is_correct fields of the analysed datasets.
        """        
        score_is_correct_relationship_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                score_is_correct_relationship_df = pickle.load(f).score_is_correct_rel_df

            score_is_correct_relationship_df_list.append(score_is_correct_relationship_df)

        score_is_correct_relationship_per_dataset_df = pd.concat(score_is_correct_relationship_df_list,
                                                                 axis=1,
                                                                 join='inner')

        score_is_correct_relationship_per_dataset_df = score_is_correct_relationship_per_dataset_df.transpose()

        score_is_correct_relationship_per_dataset_df = score_is_correct_relationship_per_dataset_df.reset_index(names=[DATASET_NAME_FIELD_NAME_STR, 
                                                                                                                       FIELD_STR])
        score_is_correct_relationship_per_dataset_df = score_is_correct_relationship_per_dataset_df.sort_values(by=DATASET_NAME_FIELD_NAME_STR,
                                                                                                                ignore_index=True)

        return score_is_correct_relationship_per_dataset_df
    
    def _return_aggregated_omnibus_test_result_per_dataset_df(self) -> pd.DataFrame:

        aggregated_omnibus_test_result_per_dataset_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                omnibus_test_result_per_group = pickle.load(f).omnibus_test_result_df
            
            eval_metric_is_categorical = omnibus_test_result_per_group[OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR].iloc[0]

            p_val_is_significant_field_name, moa_strength_guidelines = self._return_result_aggregation_omnibus_test_fields(eval_metric_is_categorical)

            fields = [DATASET_NAME_FIELD_NAME_STR,
                      GROUP_FIELD_NAME_STR,
                      OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR,
                      OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR,
                      p_val_is_significant_field_name,
                      *moa_strength_guidelines]

            omnibus_test_result_per_group = omnibus_test_result_per_group[fields]
            aggregation_dict = self._return_result_aggregation_omnibus_test_aggregation_dict(p_val_is_significant_field_name)

            agg_results = (omnibus_test_result_per_group.groupby(DATASET_NAME_FIELD_NAME_STR)
                                                        .agg(**aggregation_dict)
                                                        .reset_index())

            moa_strength_counts_pcts = []
            for moa_strength_guideline, moa_strength_calc_base in zip(moa_strength_guidelines, OmnibusTestResultMeasureAssociationStrengthCalculationBase):

                moa_calculation_base_suffix = self._return_measure_association_strength_calculation_base_suffix_str(moa_strength_calc_base)

                moa_strength_counter = self._return_moa_strength_counter()
                for _, value in omnibus_test_result_per_group.query(f'{p_val_is_significant_field_name}==True')[moa_strength_guideline].iteritems():
                    moa_strength_counter[value] += 1

                agg_moa_strength = pd.DataFrame(moa_strength_counter, 
                                                index=pd.Index([0]))
                agg_moa_strength.columns = [strength_count + moa_calculation_base_suffix for strength_count in self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names]

                n_groups = omnibus_test_result_per_group.shape[0]
                agg_moa_strength_pct = agg_moa_strength.copy() / n_groups * 100
                agg_moa_strength_pct.columns = [strength_pct + moa_calculation_base_suffix  for strength_pct in self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names]
                
                moa_strength_counts_pcts.extend([agg_moa_strength, agg_moa_strength_pct])

            agg_df_list = [agg_results] + moa_strength_counts_pcts
            agg_moa_strength = pd.concat(agg_df_list, 
                                         axis=1, 
                                         ignore_index=False)

            aggregated_omnibus_test_result_per_dataset_df_list.append(agg_moa_strength)


        aggregated_omnibus_test_result_per_dataset_df = pd.concat(aggregated_omnibus_test_result_per_dataset_df_list, 
                                                                  ignore_index=True)
        aggregated_omnibus_test_result_per_dataset_df.sort_values(by=DATASET_NAME_FIELD_NAME_STR, 
                                                                  inplace=True, 
                                                                  ignore_index=True)

        return aggregated_omnibus_test_result_per_dataset_df
    
    def _return_aggregated_omnibus_test_result_per_dataset_plotting_pct_df_count_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        aggregated_omnibus_test_result_per_dataset = self._return_aggregated_omnibus_test_result_per_dataset_df()

        index = aggregated_omnibus_test_result_per_dataset[DATASET_NAME_FIELD_NAME_STR]

        moa_strength_count_fields = self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names.copy()
        moa_strength_count_fields.insert(0, RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_NO_SIGNIFICANT_P_VALUE)

        moa_strength_pct_fields = self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names.copy()
        moa_strength_pct_fields.insert(0, RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_PCT_OF_GROUPS_NO_SIGNIFICANT_P_VALUE)
        
        non_sig = (aggregated_omnibus_test_result_per_dataset[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS] - 
                   aggregated_omnibus_test_result_per_dataset[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE])
        aggregated_omnibus_test_result_per_dataset[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_NO_SIGNIFICANT_P_VALUE] = non_sig

        count_matrix = aggregated_omnibus_test_result_per_dataset.copy()[moa_strength_count_fields].to_numpy()
        pct_matrix = count_matrix / count_matrix.sum(axis=1).reshape(count_matrix.shape[0], 1) * 100


        pct_df = pd.DataFrame(pct_matrix, 
                              columns=moa_strength_pct_fields, 
                              index=index)
        pct_df = pct_df.sort_index(ascending=True)

        count_df = pd.DataFrame(count_matrix, 
                                columns=moa_strength_count_fields, 
                                index=index)
        count_df = count_df.sort_index(ascending=True)

        return pct_df, count_df

    def _return_aggregated_omnibus_test_result_per_dataset_df_display(self,
                                                                      aggregated_omnibus_test_result_per_dataset: pd.DataFrame) -> pd.DataFrame:

        aggregated_omnibus_test_result_per_dataset = aggregated_omnibus_test_result_per_dataset.round(decimals=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_ROUND_DECIMAL_POINTS)

        # combine number and pct of groups with significant p-value
        agg_number_of_groups_sig_p_value = aggregated_omnibus_test_result_per_dataset[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE]
        agg_pct_of_groups_sig_p_value = aggregated_omnibus_test_result_per_dataset[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_PCT_OF_GROUPS_SIGNIFICANT_P_VALUE]

        agg_number_pct_of_groups_sig_p_value_field_name = [RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_PCT_OF_GROUPS_COMBINED_SIGNIFICANT_P_VALUE]
        agg_number_pct_of_groups_sig_p_value = self._combine_count_and_pct(agg_number_of_groups_sig_p_value,
                                                                           agg_pct_of_groups_sig_p_value,
                                                                           agg_number_pct_of_groups_sig_p_value_field_name)
        # combine the moa strength number and pct of groups with significant p-value
        agg_moa_strength_count_pct_combined_list = []
        for moa_strength_calc_base in OmnibusTestResultMeasureAssociationStrengthCalculationBase: #TODO: make optional in config

            moa_calculation_base_suffix = self._return_measure_association_strength_calculation_base_suffix_str(moa_strength_calc_base)

            moa_strength_count_fields = [strength_count + moa_calculation_base_suffix for strength_count in self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names]
            moa_strength_pct_fields = [strength_pct + moa_calculation_base_suffix for strength_pct in self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names]

            agg_moa_strength_count = aggregated_omnibus_test_result_per_dataset[moa_strength_count_fields]
            agg_moa_strength_pct = aggregated_omnibus_test_result_per_dataset[moa_strength_pct_fields]

            agg_moa_strength_count_pct_combined = self._combine_count_and_pct(agg_moa_strength_count,
                                                                              agg_moa_strength_pct,
                                                                              self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names)

            agg_moa_strength_count_pct_combined_list.append(agg_moa_strength_count_pct_combined)

        ##
        def sum_data(array_first: np.ndarray,
                     array_second: np.ndarray) -> np.ndarray:

            return array_first + np.full_like(array_first, ' - ') + array_second

        index = agg_moa_strength_count_pct_combined_list[0].copy().index
        agg_moa_strength_count_pct_combined_list = [df.values for df in agg_moa_strength_count_pct_combined_list]

        agg_moa_strength_count_pct_combined = reduce(sum_data, agg_moa_strength_count_pct_combined_list)

        agg_moa_strength_count_pct_combined = pd.DataFrame(agg_moa_strength_count_pct_combined,
                                                           index=index)
        agg_moa_strength_count_pct_combined.columns = self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names

        ##

        aggregation_fields = [DATASET_NAME_FIELD_NAME_STR, 
                              OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR, 
                              OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR, 
                              RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS]

        aggregated_omnibus_test_result_per_dataset_display = pd.concat([aggregated_omnibus_test_result_per_dataset[aggregation_fields], 
                                                                        agg_number_pct_of_groups_sig_p_value,
                                                                        agg_moa_strength_count_pct_combined],
                                                                        axis=1)

        # set column names for displaying
        result_aggregation_omnibus_test_result_display_field_list = [DATASET_NAME_FIELD_NAME_STR,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EVAlUATION_FIELD_IS_CATEGORICAL_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_EVAlUATION_FIELD_TYPE_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_VERY_SMALL_EFFECT_SIZE_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_SMALL_EFFECT_SIZE_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_MEDIUM_EFFECT_SIZE_DISPLAY_FIELD,
                                                                     RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE_LARGE_EFFECT_SIZE_DISPLAY_FIELD]

        aggregated_omnibus_test_result_per_dataset_display.columns = result_aggregation_omnibus_test_result_display_field_list

        return aggregated_omnibus_test_result_per_dataset_display
    
    def _sort_groups_by_metric(self,
                               data: pd.DataFrame,
                               sequence_statistic: SequenceStatisticsPlotFields | UniqueSequenceFrequencyStatisticsPlotFields,
                               seq_stat_dist_sort_metric: SequenceStatisticsDistributionSortMetric,
                               sorting_entity: SequenceStatisticsDistributionSortingEntity,
                               ascending: bool) -> pd.DataFrame:
        """
        Sorts the data by sequence statistic. Used for sorting boxplots and ridgeplots. 
        """

        data = data.copy()

        match seq_stat_dist_sort_metric:
            case SequenceStatisticsDistributionSortMetric.MEAN:
                sort_metric = np.mean

            case SequenceStatisticsDistributionSortMetric.MEDIAN:
                sort_metric = np.median

            case SequenceStatisticsDistributionSortMetric.MAX:
                sort_metric = np.max

            case SequenceStatisticsDistributionSortMetric.MIN:
                sort_metric = np.min

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{seq_stat_dist_sort_metric.__name__}')
        
        match sorting_entity:
            case SequenceStatisticsDistributionSortingEntity.GROUP:
                sorting_field = GROUP_FIELD_NAME_STR

            case SequenceStatisticsDistributionSortingEntity.DATASET:
                sorting_field = DATASET_NAME_FIELD_NAME_STR

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{sorting_entity.__name__}')

        sort_metric_values = data.groupby(sorting_field)[sequence_statistic.value]\
                                 .agg(sort_metric)\
                                 .reset_index()

        sort_metric_values.rename({sequence_statistic.value: seq_stat_dist_sort_metric.name}, 
                                  axis=1, 
                                  inplace=True)

        sort_metric_values.sort_values(by=[seq_stat_dist_sort_metric.name], inplace=True, ascending=ascending)

        data[sorting_field] = pd.Categorical(data[sorting_field], categories=sort_metric_values[sorting_field], ordered=True)
        data.sort_values(by=sorting_field, inplace=True)

        return data

    def _return_moa_strength_counter(self) -> defaultdict[str, int]:

        moa_strength_counter = defaultdict(int)
        for moa_str in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES:
            moa_strength_counter[moa_str]

        return moa_strength_counter
    
    def _return_result_aggregation_omnibus_test_fields(self,
                                                       eval_metric_is_categorical: bool) -> Tuple[str, Tuple[str, str, str]]:

        if RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_CORRECT_P_VALUES:
            p_val_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND.value + OMNIBUS_TESTS_PVAL_CORRECTED_FIELD_NAME_STR + RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_CORRECTION_METHOD.value
        else:
            p_val_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND.value

        p_val_is_significant_field_name = p_val_field_name + OMNIBUS_TESTS_PVAL_IS_SIGNIFICANT_FIELD_NAME_STR

        moa_strength_guidelines = []
        for moa_strength_calc_base in OmnibusTestResultMeasureAssociationStrengthCalculationBase:

            moa_calculation_base_suffix = self._return_measure_association_strength_calculation_base_suffix_str(moa_strength_calc_base)

            if eval_metric_is_categorical:

                moa_strength_guideline = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY.value + 
                                          '_' + 
                                          RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_CONTINGENCY.value + 
                                          OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_FIELD_NAME_STR + 
                                          moa_calculation_base_suffix)

            else:
                moa_strength_guideline = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV.value + 
                                          '_' + 
                                          RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_AOV.value + 
                                          OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_FIELD_NAME_STR + 
                                          moa_calculation_base_suffix)

            moa_strength_guidelines.append(moa_strength_guideline)

        return (p_val_is_significant_field_name, tuple(moa_strength_guidelines))
    
    def _return_measure_association_strength_calculation_base_suffix_str(self,
                                                                         moa_strength_calc_base : OmnibusTestResultMeasureAssociationStrengthCalculationBase) -> str:

        match moa_strength_calc_base:
            case OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_VALUE:
                return ''

            case OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_CONF_INT_LOWER_BOUND:
                return OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_CONF_INT_LOWER_FIELD_NAME_STR

            case OmnibusTestResultMeasureAssociationStrengthCalculationBase.MOA_CONF_INT_UPPER_BOUND:
                return OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_CONF_INT_UPPER_FIELD_NAME_STR

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{OmnibusTestResultMeasureAssociationStrengthCalculationBase.__name__}')

    
    def _return_result_aggregation_omnibus_test_aggregation_dict(self,
                                                                 p_val_is_significant_field_name) -> dict[str, int | float]:

        aggregation_dict = {
                            f'{OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR}': pd.NamedAgg(column=OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR, aggfunc=lambda x: x[0]),
                            f'{OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR}': pd.NamedAgg(column=OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR, aggfunc=lambda x: x[0]),
                            f'{RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS}': pd.NamedAgg(column=GROUP_FIELD_NAME_STR, aggfunc=len),
                            f'{RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_NUMBER_OF_GROUPS_SIGNIFICANT_P_VALUE}': pd.NamedAgg(column=p_val_is_significant_field_name, aggfunc=sum),
                            f'{RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_PCT_OF_GROUPS_SIGNIFICANT_P_VALUE}': pd.NamedAgg(column=p_val_is_significant_field_name, aggfunc=lambda x: np.mean(x)*100),
                           }

        return aggregation_dict

    def _combine_count_and_pct(self,
                               count_data: pd.Series | pd.DataFrame,
                               pct_data: pd.Series | pd.DataFrame,
                               new_field_names: list[str]) -> pd.DataFrame:
        
        index = count_data.copy().index
        count_data = count_data.copy().astype(str).values
        pct_data = pct_data.copy().astype(str).values

        count_pct_combined_data = count_data + ' (' + pct_data + ')'
        count_pct_combined_data_df = pd.DataFrame(count_pct_combined_data,
                                                  index=index)
        count_pct_combined_data_df.columns = new_field_names

        return count_pct_combined_data_df
    
    def _save_figure(self,
                     file_name):

        file_name = '_'.join(file_name.lower().split(' '))

        plt.savefig(f'{PATH_TO_RESULT_PLOTS}/{file_name}.{SAVE_FIGURE_IMAGE_FORMAT}', 
                    dpi=SAVE_FIGURE_DPI,
                    format=SAVE_FIGURE_IMAGE_FORMAT,
                    bbox_inches=SAVE_FIGURE_BBOX_INCHES)

    def _set_axis_labels(self,
                         ax: matplotlib.axes.Axes,
                         plot_x_label: bool,
                         plot_y_label: bool,
                         x_label: str | None,
                         y_label: str | None) -> None:

        if plot_x_label:
            if x_label is not None:
                ax.set_xlabel(x_label)
        else:
            ax.set_xlabel('')

        if plot_y_label:
            if y_label is not None:
                ax.set_ylabel(y_label)
        else:
            ax.set_ylabel('')
    
    def _set_axis_ticks(self,
                        ax: matplotlib.axes.Axes,
                        plot_x_ticks: bool,
                        plot_y_ticks: bool,
                        plot_x_tick_labels: bool,
                        plot_y_tick_labels: bool,
                        x_axis_ticks: NDArray[np.number] | None,
                        y_axis_ticks: NDArray[np.number] | None) -> None:

        ax.tick_params(axis='x', 
                       which='both',
                       bottom=plot_x_ticks,
                       top=False,
                       labelbottom=plot_x_tick_labels,    
                       labeltop=False)

        ax.tick_params(axis='y', 
                       which='both',
                       left=plot_y_ticks,
                       right=False,
                       labelleft=plot_y_tick_labels,    
                       labelright=False)

        if x_axis_ticks is not None:
            ax.set_xticks(x_axis_ticks)
            ax.set_xticklabels([f'{tick}' for tick in x_axis_ticks])

        if y_axis_ticks is not None:
            ax.set_yticks(y_axis_ticks)
            ax.set_yticklabels([f'{tick}' for tick in y_axis_ticks])

    def _remove_inner_plot_elements_grid(self,
                                         g: FacetGrid,
                                         n_groups: int,
                                         n_cols: int,
                                         remove_inner_x_labels: bool,
                                         remove_inner_y_labels: bool,
                                         remove_inner_x_ticks: bool,
                                         remove_inner_y_ticks: bool,
                                         remove_inner_x_tick_labels: bool,
                                         remove_inner_y_tick_labels: bool) -> None:

        index_last_row_in_grid = (n_groups - 1) // n_cols
        for n, ax in enumerate(g.axes.flatten()):
            if (n % n_cols != 0):

                if remove_inner_y_labels:
                    ax.set_ylabel('')

                plt.tick_params(axis='y', 
                                which='both',
                                left=not remove_inner_y_ticks,
                                right=False,
                                labelleft=not remove_inner_y_tick_labels,    
                                labelright=False)

            if ((n // n_groups) != index_last_row_in_grid):

                if remove_inner_x_labels:
                    ax.set_xlabel('')
  
                plt.tick_params(axis='x', 
                                which='both',
                                bottom=not remove_inner_x_ticks,
                                top=False,
                                labelbottom=not remove_inner_x_tick_labels,    
                                labeltop=False)
    
    def _set_log_scale_axes(self,
                            ax: matplotlib.axes.Axes,
                            axis: Axes) -> None:
    
        match axis:
            case Axes.X_AXIS:
                ax.set(xscale='log')

                ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None))
                ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10))

                ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
                ax.xaxis.set_minor_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))

            case Axes.Y_AXIS:
                ax.set(yscale='log')

                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None))
                ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10))

                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
                ax.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))

            case Axes.BOTH:
                ax.set(xscale='log')
                ax.set(yscale='log')

                ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None))
                ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10))

                ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
                ax.xaxis.set_minor_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))

                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None))
                ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10))

                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
                ax.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{Axes.__name__}')
    
    def _return_color_palette(self,
                              data: pd.DataFrame,
                              color_palette_agg_level: ColorPaletteAggregationLevel,
                              palette: str,
                              alpha: float,
                              saturation: float) -> Iterable[Tuple[float]]:

        n_groups = data[color_palette_agg_level.value].nunique()

        color_palette = sns.color_palette(palette,
                                          desat=saturation, 
                                          n_colors=n_groups)
        color_palette = [col + (alpha,) for col in color_palette]

        return color_palette

    def _return_color_per_group_per_dataset(self,
                                            data: pd.DataFrame,
                                            color_palette_agg_level: ColorPaletteAggregationLevel,
                                            palette: str,
                                            alpha: float,
                                            saturation: float) -> Dict[str, Dict[int, Tuple[float]]]:

        color_dict = {}
        for dataset, df in data.groupby(DATASET_NAME_FIELD_NAME_STR):

            color_palette = self._return_color_palette(df,
                                                       color_palette_agg_level,
                                                       palette,
                                                       alpha,
                                                       saturation)

            groups = np.unique(df[GROUP_FIELD_NAME_STR])
            color_dict[dataset] = dict(zip(groups, color_palette))

        return color_dict

    def _return_color_per_dataset(self,
                                  data: pd.DataFrame,
                                  color_palette_agg_level: ColorPaletteAggregationLevel,
                                  palette: str,
                                  alpha: float,
                                  saturation: float) -> Dict[str, Tuple[float]]:

        color_palette = self._return_color_palette(data,
                                                   color_palette_agg_level,
                                                   palette,
                                                   alpha,
                                                   saturation)

        datasets = np.unique(data[DATASET_NAME_FIELD_NAME_STR])
        color_dict = dict(zip(datasets, color_palette))

        return color_dict
    
    def _return_colors(self,
                       data: pd.DataFrame,
                       color_palette_agg_level: ColorPaletteAggregationLevel,
                       dataset_name: str | None,
                       color_dict: Dict[str, Dict[int, Tuple[float]]] | Dict[str, Tuple[float]]) -> List[Tuple[float]]:

        match color_palette_agg_level:
            case ColorPaletteAggregationLevel.GROUP:
                color_dict = color_dict[dataset_name]

            case ColorPaletteAggregationLevel.DATASET:
                pass

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ColorPaletteAggregationLevel.__name__}')

        keys = data[color_palette_agg_level.value].unique()
        colors = [color_dict[key] for key in keys]

        return colors