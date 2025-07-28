from .configs.result_aggregation_config import *
from .configs.conf_int_config import SequenceType, ConfIntResultFields, ConfIntEstimator
from .data_classes import *
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

                case SequenceStatisticsPlotFields.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES:

                    data = avg_sequence_statistics_per_group_per_dataset
                    x_axis_ticks = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_X_TICKS_RAW
                    x_axis_log_scale = AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_LOG_SCALE_X_RAW
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    False,
                                                    False,
                                                    is_log_scale=x_axis_log_scale)

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES:

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
                                                   [field],
                                                   AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SORT_METRIC,
                                                   SortingEntity.DATASET,
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

                case SequenceStatisticsPlotFields.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES:

                    data = sequence_summary_stats_per_group_per_dataset
                    y_axis_ticks = SUMMARY_SEQUENCE_STATISTICS_Y_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW
                    y_axis_log_scale = SUMMARY_SEQUENCE_STATISTICS_LOG_SCALE_Y_RAW

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES:

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

                case SequenceStatisticsPlotFields.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_X_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SHAREY_RAW
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_LOG_SCALE_X_RAW

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES:

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
                                                       [field],
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC,
                                                       SortingEntity.GROUP,
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
                                                                 [field],
                                                                 SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC,
                                                                 SortingEntity.GROUP,
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
                    data_range_limits = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_RAW

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_PCT
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_PCT
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_PCT
                    data_range_limits = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_PCT

                case SequenceStatisticsPlotFields.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_RAW
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_RAW
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_RAW
                    data_range_limits = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_RAW

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES:

                    data = sequence_statistics_per_group_per_dataset
                    x_axis_ticks = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_X_TICKS_PCT_RATIO
                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREX_PCT_RATIO
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SHAREY_PCT_RATIO
                    x_axis_log_scale = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_LOG_SCALE_X_PCT_RATIO
                    data_range_limits = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_DATA_RANGE_LIMITS_PCT_RATIO

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
                                                       [field],
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_METRIC,
                                                       SortingEntity.GROUP,
                                                       SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_SORT_ASCENDING)

                colors = self._return_colors(data,
                                             ColorPaletteAggregationLevel.GROUP,
                                             dataset_name,
                                             color_dict)

                n_groups = data[GROUP_FIELD_NAME_STR].nunique()

                y_axis_ticks = np.arange(0, 
                                         -n_groups*SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT, 
                                         -SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT)
                y_axis_ticks_labels = data[GROUP_FIELD_NAME_STR].unique()

                shift_value = 0
                zorder = 10000

                for (group, df), color in zip(data.groupby(GROUP_FIELD_NAME_STR), colors): 

                    field_data = df[field.value].values

                    self._plot_kde(field_data,
                                   field,
                                   data_range_limits,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_COLOR,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_LINEWIDTH,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_OUTER_LINEPLOT_ALPHA,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_COLOR,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_LINEWIDTH,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INNER_LINEPLOT_ALPHA,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_COLOR,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_LINEWIDTH,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOTTOM_LINEPLOT_ALPHA,
                                   color,
                                   None,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_INCLUDE_KDE_BOTTOM_LINE,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_APPLY_BOUNDARY_REFLECTION,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BANDWIDTH_METHOD,
                                   SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BANDWIDTH_CUT,
                                   shift_value,
                                   zorder)

                    shift_value -= SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_Y_AXIS_TICK_SHIFT_INCREMENT 
                    zorder += 1


                ax = plt.gca()
                shift_value = 0
                zorder = 10000

                for (group, df) in data.groupby(GROUP_FIELD_NAME_STR): 

                    match SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_RANGE_CONF_INT_KIND:
                        case RangeIqrConfIntKind.BOX:
                            self._kde_draw_iqr_range_conf_int_box(ax,
                                                                  dataset_name,
                                                                  group,
                                                                  df,
                                                                  field,
                                                                  shift_value,
                                                                  zorder)
                        
                        case RangeIqrConfIntKind.NONE:
                            pass

                        case _:
                            raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{RangeIqrConfIntKind.__name__}')
                    
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

    @sequence_statistics_distribution_ridgeplot_mockup_decorator
    def plot_sequence_statistics_distribution_ridgeplot_mockup(self) -> None:
        
        mock_up_data = self._return_mock_up_dist_data(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_MU,
                                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_SIMGA,
                                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_RANGE_LIMITS,
                                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_N_POINTS)

        shift_value = 0
        zorder = 10000

        self._plot_kde_mockup(mock_up_data.x_values,
                              mock_up_data.y_values,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_OUTER_LINEPLOT_COLOR,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_OUTER_LINEPLOT_LINEWIDTH,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_OUTER_LINEPLOT_ALPHA,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_INNER_LINEPLOT_COLOR,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_INNER_LINEPLOT_LINEWIDTH,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_INNER_LINEPLOT_ALPHA,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_BOTTOM_LINEPLOT_COLOR,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_BOTTOM_LINEPLOT_LINEWIDTH,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_BOTTOM_LINEPLOT_ALPHA,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_FILL_COLOR,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_FILL_ALPHA,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_INCLUDE_KDE_BOTTOM_LINE,
                              shift_value,
                              zorder)

        ax = plt.gca()

        box_height_iqr = self._kde_get_line_width_in_data_coordinates(ax,
                                                                      RESULT_AGGREGATION_FIG_SIZE_DPI,
                                                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_IQR_BOX_HEIGHT_IN_LINEWIDTH)
        box_height_range = self._kde_get_line_width_in_data_coordinates(ax,
                                                                        RESULT_AGGREGATION_FIG_SIZE_DPI,
                                                                        SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_RANGE_BOX_HEIGHT_IN_LINEWIDTH)

        iqr_box_data = self._return_iqr_box_data_mockup(mock_up_data.mu,
                                                        mock_up_data.sigma, 
                                                        box_height_iqr,
                                                        shift_value)
        range_box_data = self._return_range_box_data_mockup(mock_up_data.mu,
                                                            mock_up_data.sigma,
                                                            SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_RANGE_LOWER_QUANTILE,
                                                            SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_RANGE_UPPER_QUANTILE,
                                                            box_height_range,
                                                            shift_value)
        self._plot_iqr_range(ax,
                             iqr_box_data,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_IQR_BOX_EDGE_LINEWIDTH,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_IQR_BOX_EDGECOLOR,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_IQR_BOX_FACECOLOR,
                             zorder+0.6)

        self._plot_iqr_range(ax,
                             range_box_data,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_RANGE_BOX_EDGE_LINEWIDTH,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_RANGE_BOX_EDGECOLOR,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_RANGE_BOX_FACECOLOR,
                             zorder+0.5)

        seq_stat_conf_int_res_mean = self._return_conf_int_mock_up(mock_up_data.mu,
                                                                   iqr_box_data.box_width,
                                                                   ConfIntEstimator.MEAN)

        single_conf_int_box_data_mean = self._return_single_conf_int_box_data(seq_stat_conf_int_res_mean,
                                                                              box_height_iqr,
                                                                              shift_value)

        seq_stat_conf_int_res_median = self._return_conf_int_mock_up(mock_up_data.mu,
                                                                     iqr_box_data.box_width,
                                                                     ConfIntEstimator.MEDIAN)

        single_conf_int_box_data_median = self._return_single_conf_int_box_data(seq_stat_conf_int_res_median,
                                                                                box_height_iqr,
                                                                                shift_value)

        dual_conf_int_box_data = self._return_dual_conf_int_box_data(seq_stat_conf_int_res_mean,
                                                                     seq_stat_conf_int_res_median,
                                                                     box_height_iqr,
                                                                     shift_value)

        # confidence interval
        match SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_KIND:
            case ConfidenceIntervalKind.MEAN:
                self._plot_single_conf_int(ax,
                                           single_conf_int_box_data_mean,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_EDGE_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_UPPER_BOX_FACECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_ALPHA,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_COLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_SINGLE_BOX_SCATTER_SIZE,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_SINGLE_BOX_SCATTER_MARKER,
                                           zorder)

            case ConfidenceIntervalKind.MEDIAN:
                self._plot_single_conf_int(ax,
                                           single_conf_int_box_data_median,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_EDGE_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_LOWER_BOX_FACECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_ALPHA,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_COLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_SINGLE_BOX_SCATTER_SIZE,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_SINGLE_BOX_SCATTER_MARKER,
                                           zorder)

            case ConfidenceIntervalKind.BOTH:
                self._plot_dual_conf_int(ax,
                                         dual_conf_int_box_data,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_EDGE_LINEWIDTH,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_EDGECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_UPPER_BOX_FACECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_LOWER_BOX_FACECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_ALPHA,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_COLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_DUAL_BOX_SCATTER_SIZE,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_LINEWIDTH,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_BOX_SCATTER_EDGECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_DUAL_UPPER_BOX_SCATTER_MARKER,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_DUAL_LOWER_BOX_SCATTER_MARKER,
                                         zorder)

            case ConfidenceIntervalKind.NONE:
                pass

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ConfidenceIntervalKind.__name__}')

        # annotations
        self._add_mock_up_annotations(ax,
                                      single_conf_int_box_data_mean,
                                      single_conf_int_box_data_median,
                                      dual_conf_int_box_data,
                                      mock_up_data,
                                      iqr_box_data,
                                      range_box_data,
                                      zorder)

        self._set_axis_labels(ax,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_X_AXIS_LABEL,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_Y_AXIS_LABEL,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_X_AXIS_LABEL,
                              SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_Y_AXIS_LABEL)

        self._set_axis_ticks(ax,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_X_AXIS_TICKS,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_Y_AXIS_TICKS,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_X_AXIS_TICK_LABELS,
                             SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_Y_AXIS_TICK_LABELS,
                             None,
                             None)

        ax.grid(False,
                axis=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_GRID_LINE_AXIS.value,
                which='both')

        ax.spines['top'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_SHOW_TOP)
        ax.spines['bottom'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_SHOW_BOTTOM)
        ax.spines['left'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_SHOW_LEFT)
        ax.spines['right'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_SHOW_RIGHT)

        ax.margins(x=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_X_AXIS_MARGIN, 
                   y=SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_Y_AXIS_MARGIN)

        plt.tight_layout()
        self._save_figure(SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_PLOT_NAME)
        plt.show(ax);

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

    @cluster_size_per_group_per_dataset_categorical_scatter_decorator
    def plot_cluster_size_per_group_per_dataset_categorical_scatter(self) -> None:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)

            cluster_size_per_group = self._return_cluster_size_per_group_per_dataset_df()

            color_dict = self._return_color_per_group_per_dataset(cluster_size_per_group,
                                                                  ColorPaletteAggregationLevel.GROUP,
                                                                  CLUSTER_SIZE_SCATTERPLOT_COLOR_PALETTE,
                                                                  CLUSTER_SIZE_SCATTERPLOT_COLOR_ALPHA,
                                                                  CLUSTER_SIZE_SCATTERPLOT_COLOR_SATURATION)

            statistic_is_pct = False 
            statistic_is_ratio = False
            share_x = CLUSTER_SIZE_SCATTERPLOT_SHAREX_RAW
            share_y = CLUSTER_SIZE_SCATTERPLOT_SHAREY_RAW
            x_axis_log_scale = CLUSTER_SIZE_SCATTERPLOT_LOG_SCALE_X_RAW

            n_cols = set_facet_grid_column_number(cluster_size_per_group[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            def plot_cluster_size_scatter(data, **kwargs):

                dataset_name = data[DATASET_NAME_FIELD_NAME_STR].iloc[0]
                if CLUSTER_SIZE_SCATTERPLOT_SORT_BOXES:
                    data = self._sort_groups_by_metric(data,
                                                       [CLUSTER_SIZE_SCATTERPLOT_SORT_FIELD],
                                                       CLUSTER_SIZE_SCATTERPLOT_SORT_METRIC,
                                                       SortingEntity.GROUP,
                                                       CLUSTER_SIZE_SCATTERPLOT_SORT_ASCENDING)

                colors = self._return_colors(data,
                                             ColorPaletteAggregationLevel.GROUP,
                                             dataset_name,
                                             color_dict)

                data_clustered = data.loc[data[CLUSTER_FIELD_NAME_STR]!=-1, :]
                data_non_clustered = data.loc[data[CLUSTER_FIELD_NAME_STR]==-1, :]

                match CLUSTER_SIZE_SCATTERPLOT_KIND:
                    case ClusterSizeScatterPlotKind.STRIPPLOT:
                        sns.stripplot(data_clustered,
                                      x=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR,
                                      y=GROUP_FIELD_NAME_STR,
                                      hue=GROUP_FIELD_NAME_STR,
                                      orient='h',
                                      palette=colors,
                                      s=CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE,
                                      edgecolor=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR,
                                      marker=CLUSTER_SIZE_SCATTERPLOT_MARKER_KIND,
                                      linewidth=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGEWIDTH,
                                      jitter=CLUSTER_SIZE_SCATTERPLOT_MARKER_JITTER,
                                      zorder=2)

                        sns.stripplot(data_non_clustered,
                                      x=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR,
                                      y=GROUP_FIELD_NAME_STR,
                                      orient='h',
                                      color=CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_COLOR,
                                      s=CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE,
                                      edgecolor=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR,
                                      marker=CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_KIND,
                                      linewidth=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGEWIDTH,
                                      jitter=CLUSTER_SIZE_SCATTERPLOT_MARKER_JITTER,
                                      zorder=1)

                    case ClusterSizeScatterPlotKind.SWARMPLOT:
                        sns.swarmplot(data_clustered,
                                      x=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR,
                                      y=GROUP_FIELD_NAME_STR,
                                      hue=GROUP_FIELD_NAME_STR,
                                      orient='h',
                                      palette=colors,
                                      s=CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE,
                                      edgecolor=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR,
                                      marker=CLUSTER_SIZE_SCATTERPLOT_MARKER_KIND,
                                      linewidth=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGEWIDTH,
                                      dodge=False,
                                      zorder=2)

                        sns.swarmplot(data_non_clustered,
                                      x=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR,
                                      y=GROUP_FIELD_NAME_STR,
                                      orient='h',
                                      color=CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_COLOR,
                                      s=CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE,
                                      edgecolor=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR,
                                      marker=CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_KIND,
                                      linewidth=CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGEWIDTH,
                                      dodge=False,
                                      zorder=1)

                    case _:
                        raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ClusterSizeScatterPlotKind.__name__}')

                plt.margins(y=0.05)

            g = sns.FacetGrid(cluster_size_per_group,
                              col=DATASET_NAME_FIELD_NAME_STR,
                              col_wrap=n_cols,
                              height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                              aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                              sharex=share_x,
                              sharey=share_y)

            g.map_dataframe(plot_cluster_size_scatter)

            g.set_titles('{col_name}') 

            if share_x:
                x_axis_lim = return_axis_limits(cluster_size_per_group[RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR],
                                                statistic_is_pct,
                                                statistic_is_ratio,
                                                is_log_scale=x_axis_log_scale)
                g.set(xlim=x_axis_lim)

            for ax, (dataset_name, facet_data) in zip(g.axes.flat, cluster_size_per_group.groupby(DATASET_NAME_FIELD_NAME_STR)):

                self._set_axis_labels(ax,
                                      CLUSTER_SIZE_SCATTERPLOT_PLOT_X_AXIS_LABEL,
                                      CLUSTER_SIZE_SCATTERPLOT_PLOT_Y_AXIS_LABEL,
                                      CLUSTER_SIZE_SCATTERPLOT_X_AXIS_LABEL,
                                      CLUSTER_SIZE_SCATTERPLOT_Y_AXIS_LABEL)

                self._set_axis_ticks(ax,
                                     CLUSTER_SIZE_SCATTERPLOT_PLOT_X_AXIS_TICKS,
                                     CLUSTER_SIZE_SCATTERPLOT_PLOT_Y_AXIS_TICKS,
                                     CLUSTER_SIZE_SCATTERPLOT_PLOT_X_AXIS_TICK_LABELS,
                                     CLUSTER_SIZE_SCATTERPLOT_PLOT_Y_AXIS_TICK_LABELS,
                                     None,
                                     None)

                if x_axis_log_scale:
                    self._set_log_scale_axes(ax,
                                             Axes.X_AXIS)
                ax.grid(True,
                        axis=CLUSTER_SIZE_SCATTERPLOT_GRID_LINE_AXIS.value,
                        which='both')

                if not share_x:
                    x_axis_lim = return_axis_limits(facet_data[RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR],
                                                    statistic_is_pct,
                                                    statistic_is_ratio,
                                                    is_log_scale=x_axis_log_scale)
                    ax.set_xlim(*x_axis_lim)


                if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:

                    facet_data = self._sort_groups_by_metric(facet_data,
                                                             [CLUSTER_SIZE_SCATTERPLOT_SORT_FIELD],
                                                             CLUSTER_SIZE_SCATTERPLOT_SORT_METRIC,
                                                             SortingEntity.GROUP,
                                                             CLUSTER_SIZE_SCATTERPLOT_SORT_ASCENDING)

                ax.spines['top'].set_visible(CLUSTER_SIZE_SCATTERPLOT_SHOW_TOP)
                ax.spines['bottom'].set_visible(CLUSTER_SIZE_SCATTERPLOT_SHOW_BOTTOM)
                ax.spines['left'].set_visible(CLUSTER_SIZE_SCATTERPLOT_SHOW_LEFT)
                ax.spines['right'].set_visible(CLUSTER_SIZE_SCATTERPLOT_SHOW_RIGHT)

            n_groups = cluster_size_per_group[DATASET_NAME_FIELD_NAME_STR].nunique()
            self._remove_inner_plot_elements_grid(g,
                                                  n_groups,
                                                  n_cols,
                                                  CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_X_AXIS_LABELS,
                                                  CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_Y_AXIS_LABELS,
                                                  CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_X_AXIS_TICKS,
                                                  CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_Y_AXIS_TICKS,
                                                  CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                                  CLUSTER_SIZE_SCATTERPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS)

            # plot the legend with matching markers
            if CLUSTER_SIZE_SCATTERPLOT_PLOT_LEGEND:
                self._plot_cluster_size_scatter_legend(g,
                                                       CLUSTER_SIZE_SCATTERPLOT_MARKER_KIND,
                                                       CLUSTER_SIZE_SCATTERPLOT_NON_CLUSTERED_MARKER_KIND,
                                                       RESULT_AGGREGATION_CLUSTER_SIZE_LEGEND_CLUSTERED_SEQUENCES_STR,
                                                       RESULT_AGGREGATION_CLUSTER_SIZE_LEGEND_NON_CLUSTERED_SEQUENCES_STR,
                                                       CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR,
                                                       CLUSTER_SIZE_SCATTERPLOT_MARKER_EDGECOLOR,
                                                       CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE,
                                                       CLUSTER_SIZE_SCATTERPLOT_MARKER_SIZE)
            
            plt.tight_layout()
            self._save_figure(CLUSTER_SIZE_SCATTERPLOT_NAME)
            plt.show(g);

    @cluster_size_per_group_per_dataset_lineplot_decorator
    def plot_cluster_size_per_group_per_dataset_lineplot(self) -> None:
        
        cluster_size_per_group = self._return_cluster_size_per_group_per_dataset_df()
        cluster_size_per_group_sorted_and_ranked = self._return_cluster_size_per_group_per_dataset_sorted_and_ranked_df(cluster_size_per_group)

        color_dict = self._return_color_per_group_per_dataset(cluster_size_per_group_sorted_and_ranked,
                                                              ColorPaletteAggregationLevel.GROUP,
                                                              CLUSTER_SIZE_LINEPLOT_COLOR_PALETTE,
                                                              CLUSTER_SIZE_LINEPLOT_COLOR_ALPHA,
                                                              CLUSTER_SIZE_LINEPLOT_COLOR_SATURATION)

        statistic_is_pct = False 
        statistic_is_ratio = False
        share_x = CLUSTER_SIZE_LINEPLOT_SHAREX_RAW
        share_y = CLUSTER_SIZE_LINEPLOT_SHAREY_RAW
        x_axis_log_scale = CLUSTER_SIZE_LINEPLOT_LOG_SCALE_X_RAW

        n_cols = set_facet_grid_column_number(cluster_size_per_group_sorted_and_ranked[DATASET_NAME_FIELD_NAME_STR],
                                              RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)


            
        def plot_cluster_size_lineplot(data, **kwargs):

            dataset_name = data[DATASET_NAME_FIELD_NAME_STR].iloc[0]
            colors = self._return_colors(data,
                                         ColorPaletteAggregationLevel.GROUP,
                                         dataset_name,
                                         color_dict)

            sns.lineplot(data,
                         x=RESULT_AGGREGATION_CLUSTER_SIZE_CLUSTER_RANK_FIELD_NAME_STR,
                         y=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR,
                         hue=GROUP_FIELD_NAME_STR,
                         legend=CLUSTER_SIZE_LINEPLOT_PLOT_LEGEND,
                         palette=colors,
                         linewidth=CLUSTER_SIZE_LINEPLOT_LINE_WIDTH) 

            sns.scatterplot(data,
                            x=RESULT_AGGREGATION_CLUSTER_SIZE_CLUSTER_RANK_FIELD_NAME_STR, 
                            y=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR, 
                            hue=GROUP_FIELD_NAME_STR,
                            palette=colors,
                            marker=CLUSTER_SIZE_LINEPLOT_MARKER_KIND,
                            s=CLUSTER_SIZE_LINEPLOT_MARKER_SIZE,
                            edgecolor=CLUSTER_SIZE_LINEPLOT_MARKER_EDGECOLOR,
                            linewidth=CLUSTER_SIZE_LINEPLOT_MARKER_EDGEWIDTH,
                            alpha=CLUSTER_SIZE_LINEPLOT_MARKER_ALPHA,
                            zorder=3)

        g = sns.FacetGrid(cluster_size_per_group_sorted_and_ranked,
                          col=DATASET_NAME_FIELD_NAME_STR,
                          col_wrap=n_cols,
                          height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                          aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                          sharex=share_x,
                          sharey=share_y)

        g.map_dataframe(plot_cluster_size_lineplot)

        g.set_titles('{col_name}') 

        if share_y:
            y_axis_lim = return_axis_limits(cluster_size_per_group_sorted_and_ranked[RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR],
                                            statistic_is_pct,
                                            statistic_is_ratio,
                                            is_log_scale=x_axis_log_scale)
            g.set(ylim=y_axis_lim)

        for ax, (dataset_name, facet_data) in zip(g.axes.flat, cluster_size_per_group_sorted_and_ranked.groupby(DATASET_NAME_FIELD_NAME_STR)):

            self._set_axis_labels(ax,
                                  CLUSTER_SIZE_LINEPLOT_PLOT_X_AXIS_LABEL,
                                  CLUSTER_SIZE_LINEPLOT_PLOT_Y_AXIS_LABEL,
                                  CLUSTER_SIZE_LINEPLOT_X_AXIS_LABEL,
                                  CLUSTER_SIZE_LINEPLOT_Y_AXIS_LABEL)

            self._set_axis_ticks(ax,
                                 CLUSTER_SIZE_LINEPLOT_PLOT_X_AXIS_TICKS,
                                 CLUSTER_SIZE_LINEPLOT_PLOT_Y_AXIS_TICKS,
                                 CLUSTER_SIZE_LINEPLOT_PLOT_X_AXIS_TICK_LABELS,
                                 CLUSTER_SIZE_LINEPLOT_PLOT_Y_AXIS_TICK_LABELS,
                                 None,
                                 None)

            if x_axis_log_scale:
                self._set_log_scale_axes(ax,
                                         Axes.Y_AXIS)
            ax.grid(True,
                    axis=CLUSTER_SIZE_LINEPLOT_GRID_LINE_AXIS.value,
                    which='both')

            if not share_x:
                y_axis_lim = return_axis_limits(facet_data[RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR],
                                                statistic_is_pct,
                                                statistic_is_ratio,
                                                is_log_scale=x_axis_log_scale)
                ax.set_ylim(*y_axis_lim)

            ax.spines['top'].set_visible(CLUSTER_SIZE_LINEPLOT_SHOW_TOP)
            ax.spines['bottom'].set_visible(CLUSTER_SIZE_LINEPLOT_SHOW_BOTTOM)
            ax.spines['left'].set_visible(CLUSTER_SIZE_LINEPLOT_SHOW_LEFT)
            ax.spines['right'].set_visible(CLUSTER_SIZE_LINEPLOT_SHOW_RIGHT)

        n_groups = cluster_size_per_group[DATASET_NAME_FIELD_NAME_STR].nunique()
        self._remove_inner_plot_elements_grid(g,
                                              n_groups,
                                              n_cols,
                                              CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_X_AXIS_LABELS,
                                              CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_Y_AXIS_LABELS,
                                              CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_X_AXIS_TICKS,
                                              CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_Y_AXIS_TICKS,
                                              CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                              CLUSTER_SIZE_LINEPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS)
        
        plt.tight_layout()
        self._save_figure(CLUSTER_SIZE_LINEPLOT_NAME)
        plt.show(g);

    @aggregated_omnibus_test_result_per_dataset_stacked_barplot_decorator
    def plot_aggregated_omnibus_test_result_per_dataset_stacked_barplot(self) -> None:

        # plotting data 
        pct_df, count_df = self._return_aggregated_omnibus_test_result_per_dataset_plotting_pct_df_count_df()

        # use fields name appropriate for plotting
        moa_streng_categories = ([RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_GROUP_NON_SIGNIFICANT_NAME_STR] + 
                                 RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES)
        moa_streng_categories = self._format_moa_strength_labels(moa_streng_categories,
                                                                 OMNIBUS_TEST_RESULT_STACKED_BARPLOT_MOA_LABELS_ADD_EFFECT_SIZE_STR,
                                                                 OMNIBUS_TEST_RESULT_STACKED_BARPLOT_MOA_LABELS_CAPITALIZE)

        pct_df.columns = moa_streng_categories
        count_df.columns = moa_streng_categories

        pct_df = pct_df.sort_index(ascending=False)
        count_df = count_df.sort_index(ascending=False)
        pct_matrix = pct_df.to_numpy()
        count_matrix = count_df.to_numpy()

        x_axis_lim = return_axis_limits(None,
                                        True,
                                        False)
        x_axis_ticks = np.arange(0, 110, 10)

        n_moa_strength_categories = len(pct_df.columns) - 1
        moa_strength_colors = self._return_moa_strength_color_palette(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE,
                                                                      OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_NUMBER_COLORS,
                                                                      OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_COLOR_INDEX_MIN,
                                                                      OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_COLOR_INDEX_MAX,
                                                                      n_moa_strength_categories,
                                                                      OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_DESAT,
                                                                      OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PALETTE_ALPHA,
                                                                      OMNIBUS_TEST_RESULT_STACKED_BARPLOT_NON_SIG_COLOR)

        cmap = ListedColormap(moa_strength_colors)

        # plot
        ax = pct_df.plot.barh(stacked=True, 
                              colormap=cmap,
                              edgecolor=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_EDGECOLOR,
                              linewidth=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_LINEWIDTH,
                              width=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_BAR_WIDTH)

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

        self._set_axis_labels(ax,
                                OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_X_AXIS_LABEL,
                                OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_Y_AXIS_LABEL,
                                OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_AXIS_LABEL,
                                OMNIBUS_TEST_RESULT_STACKED_BARPLOT_Y_AXIS_LABEL)

        self._set_axis_ticks(ax,
                             OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_X_AXIS_TICKS,
                             OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_Y_AXIS_TICKS,
                             OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_X_AXIS_TICK_LABELS,
                             OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_Y_AXIS_TICK_LABELS,
                             OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_TICKS,
                             None,
                             x_axis_ticks_position=None,
                             y_axis_ticks_position=None,
                             x_rotation=OMNIBUS_TEST_RESULT_STACKED_BARPLOT_X_TICKS_ROTATION)

        # plot the legend with matching colors
        ax = plt.gca()
        ax.get_legend().remove()
        if OMNIBUS_TEST_RESULT_STACKED_BARPLOT_PLOT_LEGEND:
            
            self._add_moa_strength_legend(ax,
                                          OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_COLOR,
                                          OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_LINEWIDTH,
                                          OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_LENGTH,
                                          OMNIBUS_TEST_RESULT_STACKED_BARPLOT_LEGEND_BOX_WIDTH,
                                          pct_df.shape[1],
                                          None)

        plt.tight_layout()
        self._save_figure(OMNIBUS_TEST_RESULT_STACKED_BARPLOT_NAME)
        plt.show(ax);

    @aggregated_omnibus_test_result_per_dataset_grouped_barplot_decorator
    def plot_aggregated_omnibus_test_result_per_dataset_grouped_barplot(self) -> None:

        # plotting data 
        pct_df, count_df = self._return_aggregated_omnibus_test_result_per_dataset_plotting_pct_df_count_df()

        # use fields name appropriate for plotting
        moa_streng_categories = ([RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_GROUP_NON_SIGNIFICANT_NAME_STR] + 
                                 RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES)
        moa_streng_categories = self._format_moa_strength_labels(moa_streng_categories,
                                                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_MOA_LABELS_ADD_EFFECT_SIZE_STR,
                                                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_MOA_LABELS_CAPITALIZE)
        pct_df.columns = moa_streng_categories
        count_df.columns = moa_streng_categories

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

        n_moa_strength_categories = pct_df_long[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR].nunique() - 1
        moa_strength_colors = self._return_moa_strength_color_palette(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE,
                                                                      OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_NUMBER_COLORS,
                                                                      OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_COLOR_INDEX_MIN,
                                                                      OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_COLOR_INDEX_MAX,
                                                                      n_moa_strength_categories,
                                                                      OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_DESAT,
                                                                      OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PALETTE_ALPHA,
                                                                      OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_NON_SIG_COLOR)

        n_cols = set_facet_grid_column_number(pct_df_long[DATASET_NAME_FIELD_NAME_STR],
                                              RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)
        def plot_grouped_barplot(data, **kwargs):

            sns.barplot(data=data,
                        x=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR,
                        y=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GROUPS_PCT_FIELD_NAME_STR,
                        hue=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_FIELD_NAME_STR,
                        orient=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_ORIENTATION,
                        legend=True,
                        palette=moa_strength_colors,
                        saturation=1,
                        width=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_WIDTH,
                        linewidth=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_EDGEWIDTH)

            ax = plt.gca()
            for bar, color in zip(ax.patches, moa_strength_colors):
                bar.set_facecolor(color)
                bar.set_edgecolor(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_EDGECOLOR)

        g = sns.FacetGrid(pct_df_long,
                          col=DATASET_NAME_FIELD_NAME_STR,
                          col_wrap=n_cols,
                          height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                          aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                          sharex=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHAREX,
                          sharey=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_SHAREY)

        g.map_dataframe(plot_grouped_barplot)

        g.add_legend(
                    title=None,
                    frameon=True,
                    bbox_to_anchor=(0.98, 0.5), 
                    loc='center left')

        g.set_titles('{col_name}') 

        g.set(ylim=y_axis_lim)
        
        for ax, (df_name, df) in zip(g.axes.flat, pct_count_df_long.groupby(DATASET_NAME_FIELD_NAME_STR)):

            self._set_axis_labels(ax,
                                  OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_X_AXIS_LABEL,
                                  OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_Y_AXIS_LABEL,
                                  OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_X_AXIS_LABEL,
                                  OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_Y_AXIS_LABEL)

            self._set_axis_ticks(ax,
                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_X_AXIS_TICKS,
                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_Y_AXIS_TICKS,
                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_X_AXIS_TICK_LABELS,
                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_Y_AXIS_TICK_LABELS,
                                 None,
                                 OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_Y_TICKS,
                                 x_axis_ticks_position=None,
                                 y_axis_ticks_position=None,
                                 x_rotation=OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_X_TICKS_ROTATION)

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
        
        n_groups = pct_df_long[DATASET_NAME_FIELD_NAME_STR].nunique()
        self._remove_inner_plot_elements_grid(g,
                                              n_groups,
                                              n_cols,
                                              OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_X_AXIS_LABELS,
                                              OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_Y_AXIS_LABELS,
                                              OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_X_AXIS_TICKS,
                                              OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_Y_AXIS_TICKS,
                                              OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                              OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_REMOVE_INNER_Y_AXIS_TICK_LABELS)

        # plot the legend with matching colors
        g.legend.remove()
        if OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_PLOT_LEGEND:
            self._add_moa_strength_facet_grid_legend(g,
                                                     OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_COLOR,
                                                     OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_LINEWIDTH,
                                                     OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_LENGTH,
                                                     OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_LEGEND_BOX_WIDTH,
                                                     None)
        
        plt.tight_layout()
        self._save_figure(OMNIBUS_TEST_RESULT_GROUPED_BARPLOT_NAME)
        plt.show(g);

    @omnibus_test_result_moa_confidence_interval_per_group_per_dataset_decorator
    def plot_omnibus_test_result_moa_confidence_interval_per_group_per_dataset(self) -> None:

        data = self._return_omnibus_test_result_per_dataset_plotting_moa_conf_int_df()
        color_dict = self._return_color_per_group_per_dataset(data,
                                                              ColorPaletteAggregationLevel.GROUP,
                                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_COLOR_PALETTE,
                                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_COLOR_ALPHA,
                                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_COLOR_SATURATION)
        data = self._filter_data_by_significant_groups(data)

        field = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_EFFECT_SIZE_VALUE_FIELD
        field_lower_bound = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_EFFECT_SIZE_VALUE_LOWER_BOUND_FIELD
        field_upper_bound = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_EFFECT_SIZE_VALUE_UPPER_BOUND_FIELD

        x_axis_ticks = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_X_TICKS_PCT_RATIO
        statistic_is_pct = False 
        statistic_is_ratio = True
        share_x = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHAREX_PCT_RATIO
        share_y = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHAREY_PCT_RATIO

        n_cols = set_facet_grid_column_number(data[DATASET_NAME_FIELD_NAME_STR],
                                              RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

        def plot_moa_conf_int(data, **kwargs):

            dataset_name = data[DATASET_NAME_FIELD_NAME_STR].iloc[0]
            if OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_CONFIDENCE_INTERVALS:
                data = self._sort_groups_by_metric(data,
                                                   [field, OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_SECOND_SORT_FIELD],
                                                   OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_METRIC,
                                                   SortingEntity.GROUP,
                                                   OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_ASCENDING)

            colors = self._return_colors(data,
                                         ColorPaletteAggregationLevel.GROUP,
                                         dataset_name,
                                         color_dict)

            x_array = data[field.value].to_numpy()
            y_array = data[GROUP_FIELD_NAME_STR].to_numpy()
            y_array_numerical = self._return_y_array_numerical(y_array)

            distance_lower_bound = (x_array - 
                                    data[field_lower_bound.value])
            distance_upper_bound = (data[field_upper_bound.value] - 
                                    x_array)


            sns.scatterplot(data=data,
                            x=field.value,
                            y=y_array_numerical,
                            hue=GROUP_FIELD_NAME_STR,
                            palette=colors,
                            s=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SCATTER_MARKER_SIZE,
                            edgecolor=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SCATTER_MARKER_EDGECOLOR,
                            linewidth=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SCATTER_MARKER_EDGEWIDTH,
                            legend=False,
                            zorder=2)

            plt.errorbar(x=x_array, 
                         y=y_array_numerical, 
                         xerr=[distance_lower_bound, distance_upper_bound], 
                         fmt='none', 
                         capsize=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_CAP_SIZE,
                         capthick=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_CAP_THICKNESS,
                         ecolor=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_BAR_COLOR, 
                         elinewidth=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_ERRORBAR_BAR_LINEWIDTH,
                         zorder=1)
            
            plt.margins(y=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_Y_AXIS_MARGIN)

            moa_strength_cat_interval_mapping = data[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_BOUNDARIES_NAME_STR].iloc[0]
            n_moa_strength_categories = len(moa_strength_cat_interval_mapping)
            
            moa_strength_colors = self._return_moa_strength_color_palette(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE,
                                                                          OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_NUMBER_COLORS,
                                                                          OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_COLOR_INDEX_MIN,
                                                                          OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_COLOR_INDEX_MAX,
                                                                          n_moa_strength_categories,
                                                                          OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_DESAT,
                                                                          OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PALETTE_ALPHA,
                                                                          OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_NON_SIG_COLOR)
            
            moa_cat_interval_col = zip(moa_strength_cat_interval_mapping.keys(),
                                       moa_strength_cat_interval_mapping.values(),
                                       moa_strength_colors)

            for moa_strength_cat, moa_strength_interval, moa_strength_color in moa_cat_interval_col:

                lower_bound = moa_strength_interval[0]
                upper_bound = moa_strength_interval[1]

                if upper_bound == np.inf:
                    upper_bound = 1

                plt.axvspan(lower_bound,
                            upper_bound,
                            facecolor=moa_strength_color,
                            edgecolor='none',
                            label=moa_strength_cat.value,
                            zorder=-1)


        g = sns.FacetGrid(data,
                          col=DATASET_NAME_FIELD_NAME_STR,
                          col_wrap=n_cols,
                          height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                          aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                          sharex=share_x,
                          sharey=share_y)

        g.map_dataframe(plot_moa_conf_int)

        g.set_titles('{col_name}') 

        if share_x:
            x_axis_lim = return_axis_limits(data[field_upper_bound.value],
                                            statistic_is_pct,
                                            statistic_is_ratio)
            g.set(xlim=x_axis_lim)

        for ax, (dataset_name, facet_data) in zip(g.axes.flat, data.groupby(DATASET_NAME_FIELD_NAME_STR)):

            # x_label can be set statically or dynamically
            x_label = self._return_moa_conf_int_plot_x_label(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_X_AXIS_LABEL,
                                                             facet_data)

            if OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_CONFIDENCE_INTERVALS:
                facet_data = self._sort_groups_by_metric(facet_data,
                                                         [field, OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_SECOND_SORT_FIELD],
                                                         OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_METRIC,
                                                         SortingEntity.GROUP,
                                                         OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SORT_ASCENDING)

            y_array = facet_data[GROUP_FIELD_NAME_STR].to_numpy()
            y_array_numerical = self._return_y_array_numerical(y_array)

            self._set_axis_labels(ax,
                                  OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_X_AXIS_LABEL,
                                  OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_Y_AXIS_LABEL,
                                  x_label,
                                  OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_Y_AXIS_LABEL)

            self._set_axis_ticks(ax,
                                 OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_X_AXIS_TICKS,
                                 OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_Y_AXIS_TICKS,
                                 OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_X_AXIS_TICK_LABELS,
                                 OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_Y_AXIS_TICK_LABELS,
                                 x_axis_ticks,
                                 y_array,
                                 x_axis_ticks_position=None,
                                 y_axis_ticks_position=y_array_numerical)

            ax.grid(True,
                    axis=OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_GRID_LINE_AXIS.value,
                    which='both')

            if not share_x:
                x_axis_lim = return_axis_limits(facet_data[field_upper_bound.value],
                                                statistic_is_pct,
                                                statistic_is_ratio)
                ax.set_xlim(*x_axis_lim)

            ax.spines['top'].set_visible(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_TOP)
            ax.spines['bottom'].set_visible(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_BOTTOM)
            ax.spines['left'].set_visible(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_LEFT)
            ax.spines['right'].set_visible(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_SHOW_RIGHT)

        n_groups = data[DATASET_NAME_FIELD_NAME_STR].nunique()
        self._remove_inner_plot_elements_grid(g,
                                              n_groups,
                                              n_cols,
                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_X_AXIS_LABELS,
                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_Y_AXIS_LABELS,
                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_X_AXIS_TICKS,
                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_Y_AXIS_TICKS,
                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_X_AXIS_TICK_LABELS,
                                              OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_REMOVE_INNER_Y_AXIS_TICK_LABELS)
        
        # plot the legend with matching colors
        if OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_LEGEND:

            moa_strength_cat_interval_mapping = data[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_BOUNDARIES_NAME_STR].iloc[0]
            moa_streng_categories = [i.value for i in moa_strength_cat_interval_mapping.keys()]
            moa_streng_categories = self._format_moa_strength_labels(moa_streng_categories,
                                                                     OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_MOA_LABELS_ADD_EFFECT_SIZE_STR,
                                                                     OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_MOA_LABELS_CAPITALIZE)
            self._add_moa_strength_facet_grid_legend(g,
                                                     OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_COLOR,
                                                     OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_LINEWIDTH,
                                                     OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_LENGTH,
                                                     OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_LEGEND_BOX_WIDTH,
                                                     moa_streng_categories)

        plt.tight_layout()
        self._save_figure(OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_PLOT_NAME)
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

        learning_activity_sequence_stats_per_group_per_dataset = pd.concat(learning_activity_sequence_stats_per_group_df_list)
        learning_activity_sequence_stats_per_group_per_dataset.sort_values([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], inplace=True)

        return learning_activity_sequence_stats_per_group_per_dataset
    
    def _return_sequence_statistics_confidence_interval_per_group_per_dataset_df(self,
                                                                                 use_unique_sequences: bool) -> pd.DataFrame:

        sequence_statistics_conf_int_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                sequence_statistics_conf_int_df = pickle.load(f).seq_stat_conf_int_df
            
            sequence_statistics_conf_int_df_list.append(sequence_statistics_conf_int_df)

        sequence_statistics_conf_int_per_group_per_dataset = pd.concat(sequence_statistics_conf_int_df_list)

        if use_unique_sequences:
            sequence_type = SequenceType.UNIQUE_SEQUENCES.value
        else:
            sequence_type = SequenceType.ALL_SEQUENCES.value
        
        seq_type_filter = sequence_statistics_conf_int_per_group_per_dataset[ConfIntResultFields.SEQUENCE_TYPE.value] == sequence_type
        sequence_statistics_conf_int_per_group_per_dataset = sequence_statistics_conf_int_per_group_per_dataset.loc[seq_type_filter, :]


        sort_list = [ConfIntResultFields.DATASET_NAME.value,
                     ConfIntResultFields.SEQUENCE_TYPE.value,
                     ConfIntResultFields.SEQUENCE_STATISTIC.value,
                     ConfIntResultFields.ESTIMATOR.value,
                     ConfIntResultFields.GROUP.value]
        sequence_statistics_conf_int_per_group_per_dataset = (sequence_statistics_conf_int_per_group_per_dataset.sort_values(by=sort_list) 
                                                                                                                .reset_index(drop=True))

        return sequence_statistics_conf_int_per_group_per_dataset

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

    def _return_cluster_size_per_group_per_dataset_df(self) -> pd.DataFrame:

        cluster_size_per_group_per_dataset_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                sequence_cluster_per_group_df = pickle.load(f).sequence_cluster_per_group_df
                
            cluster_size_per_group_df = (sequence_cluster_per_group_df.groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR])
                                                                      .size()
                                                                      .reset_index(name=RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR))

            cluster_size_per_group_per_dataset_df_list.append(cluster_size_per_group_df)


        cluster_size_per_group_per_dataset_df = pd.concat(cluster_size_per_group_per_dataset_df_list, 
                                                          ignore_index=True)
        cluster_size_per_group_per_dataset_df.sort_values(by=[DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR], 
                                                          inplace=True, 
                                                          ignore_index=True)

        return cluster_size_per_group_per_dataset_df
    
    def _return_cluster_size_per_group_per_dataset_sorted_and_ranked_df(self,
                                                                        cluster_size_per_group: pd.DataFrame) -> pd.DataFrame:

        cluster_size_per_group = cluster_size_per_group.sort_values(by=[DATASET_NAME_FIELD_NAME_STR,
                                                                        GROUP_FIELD_NAME_STR,
                                                                        RESULT_AGGREGATION_CLUSTER_SIZE_SEQUENCE_COUNT_PER_CLUSTER_FIELD_NAME_STR])
        df_list = []
        for (_, _), df in cluster_size_per_group.groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR]):
            df = df.sort_values(by=[CLUSTER_FIELD_NAME_STR],
                                key=self._cluster_non_clustered_sorting_function)

            rank = [inflection.ordinalize(i) for i in range(1 , df.shape[0] + 1)]

            contains_non_clusterd = (df[CLUSTER_FIELD_NAME_STR] == -1).any()
            if contains_non_clusterd: 
                rank.insert(0, RESULT_AGGREGATION_CLUSTER_SIZE_NON_CLUSTERED_STR)
                _ = rank.pop(-1)

            df[RESULT_AGGREGATION_CLUSTER_SIZE_CLUSTER_RANK_FIELD_NAME_STR] = rank
            df_list.append(df)

        cluster_size_per_group_sorted = pd.concat(df_list, 
                                                  axis=0,
                                                  ignore_index=True)
        cluster_ranks = list(np.unique(cluster_size_per_group_sorted.loc[cluster_size_per_group_sorted[CLUSTER_FIELD_NAME_STR]!=-1, 
                                                                         RESULT_AGGREGATION_CLUSTER_SIZE_CLUSTER_RANK_FIELD_NAME_STR]))
        cluster_rank_sorted_levels = ([RESULT_AGGREGATION_CLUSTER_SIZE_NON_CLUSTERED_STR] + 
                                      cluster_ranks)
        cluster_size_per_group_sorted[RESULT_AGGREGATION_CLUSTER_SIZE_CLUSTER_RANK_FIELD_NAME_STR] = pd.Categorical(cluster_size_per_group_sorted[RESULT_AGGREGATION_CLUSTER_SIZE_CLUSTER_RANK_FIELD_NAME_STR], 
                                                                                                                    categories=cluster_rank_sorted_levels, 
                                                                                                                    ordered=True)
        return cluster_size_per_group_sorted
                                                
    def _cluster_non_clustered_sorting_function(self,
                                                series: pd.Series) -> pd.Series:
        """Used for sorting clusters such that non-clustered data will be ranked before clusters and clusters sorted by cluster size keep existing sort order."""
        if pd.api.types.is_numeric_dtype(series):
            series = (series >= 0).astype(int) 
            return series
        else:
            return series
    
    def _return_aggregated_omnibus_test_result_per_dataset_df(self) -> pd.DataFrame:

        aggregated_omnibus_test_result_per_dataset_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                omnibus_test_result_per_group = pickle.load(f).omnibus_test_result_df
            
            eval_metric_is_categorical = omnibus_test_result_per_group[OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR].iloc[0]

            p_val_is_significant_field_name, moa_strength_guidelines = self._return_result_aggregation_omnibus_test_pval_sig_moa_strength_guide_field_names(eval_metric_is_categorical)

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
    
    def _return_omnibus_test_result_per_dataset_plotting_moa_conf_int_df(self) -> pd.DataFrame:

        omnibus_test_result_per_group_per_dataset_moa_conf_int_df_list = []
        for file_path in self._path_to_result_tables:

            with open(file_path, 'rb') as f:
                omnibus_test_result_per_group = pickle.load(f).omnibus_test_result_df
            
            eval_metric_is_categorical = omnibus_test_result_per_group[OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR].iloc[0]

            (p_val_is_significant_field_name, 
             moa_value, 
             moa_conf_int) = self._return_result_aggregation_omnibus_test_pval_sig_moa_field_names(eval_metric_is_categorical)

            fields = [DATASET_NAME_FIELD_NAME_STR,
                      GROUP_FIELD_NAME_STR,
                      OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR,
                      OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR,
                      p_val_is_significant_field_name]

            moa_value_new = moa_value.replace(OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR, '')
            omnibus_test_result_per_group.rename({moa_value: moa_value_new}, inplace=True, axis=1)

            moa_conf_int_lower_values = omnibus_test_result_per_group[moa_conf_int].map(lambda x: x[0] if x[0] is not None else np.nan).to_numpy()
            moa_conf_int_upper_values = omnibus_test_result_per_group[moa_conf_int].map(lambda x: x[1] if x[1] is not None else np.nan).to_numpy()

            omnibus_test_result_per_group_moa_conf_int = pd.melt(omnibus_test_result_per_group, 
                                                                 id_vars=fields,
                                                                 value_vars=[moa_value_new],
                                                                 value_name=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_VALUE_NAME_STR,
                                                                 var_name=RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_KIND_NAME_STR)

            omnibus_test_result_per_group_moa_conf_int[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_VALUE_CONF_INT_LOWER_NAME_STR] = moa_conf_int_lower_values
            omnibus_test_result_per_group_moa_conf_int[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_VALUE_CONF_INT_UPPER_NAME_STR] = moa_conf_int_upper_values
    
            moa_strength_guide_boundaries = self._return_measure_association_strength_guideline_boundary_dict(eval_metric_is_categorical)
            omnibus_test_result_per_group_moa_conf_int[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_BOUNDARIES_NAME_STR] = [moa_strength_guide_boundaries] * omnibus_test_result_per_group_moa_conf_int.shape[0]

            omnibus_test_result_per_group_per_dataset_moa_conf_int_df_list.append(omnibus_test_result_per_group_moa_conf_int)
        

        omnibus_test_result_per_group_per_dataset_moa_conf_int_df = pd.concat(omnibus_test_result_per_group_per_dataset_moa_conf_int_df_list, 
                                                                              ignore_index=True)
        omnibus_test_result_per_group_per_dataset_moa_conf_int_df.sort_values(by=[DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR], 
                                                                              inplace=True, 
                                                                              ignore_index=True)

        return omnibus_test_result_per_group_per_dataset_moa_conf_int_df
    
    def _filter_data_by_significant_groups(self,
                                           data: pd.DataFrame) -> pd.DataFrame:

        group_inclusion = OMNIBUS_TEST_RESULT_MEASURE_ASSOCIATION_GROUP_INCLUSION

        match group_inclusion:
            case OmnibusTestResultMeasureAssociationConfIntGroupInclusion.ALL_GROUPS:
                pass

            case OmnibusTestResultMeasureAssociationConfIntGroupInclusion.SIGNIFICANT_GROUPS:
                p_val_is_significant_field_name = self._return_p_val_is_significant_field_name()
                data = data.loc[data[p_val_is_significant_field_name], :]

            case OmnibusTestResultMeasureAssociationConfIntGroupInclusion.NON_SIGNIFICANT_GROUPS:
                p_val_is_significant_field_name = self._return_p_val_is_significant_field_name()
                data = data.loc[~data[p_val_is_significant_field_name], :]

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{OmnibusTestResultMeasureAssociationConfIntGroupInclusion.__name__}')
            
        return data

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
        for moa_strength_calc_base in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_TABLE_VALUES:

            moa_calculation_base_suffix = self._return_measure_association_strength_calculation_base_suffix_str(moa_strength_calc_base)

            moa_strength_count_fields = [strength_count + moa_calculation_base_suffix for strength_count in self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names]
            moa_strength_pct_fields = [strength_pct + moa_calculation_base_suffix for strength_pct in self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names]

            agg_moa_strength_count = aggregated_omnibus_test_result_per_dataset[moa_strength_count_fields]
            agg_moa_strength_pct = aggregated_omnibus_test_result_per_dataset[moa_strength_pct_fields]

            agg_moa_strength_count_pct_combined = self._combine_count_and_pct(agg_moa_strength_count,
                                                                              agg_moa_strength_pct,
                                                                              self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names)

            agg_moa_strength_count_pct_combined_list.append(agg_moa_strength_count_pct_combined)

        def sum_data(array_first: np.ndarray,
                     array_second: np.ndarray) -> np.ndarray:

            return array_first + np.full_like(array_first, ' - ') + array_second

        index = agg_moa_strength_count_pct_combined_list[0].copy().index
        agg_moa_strength_count_pct_combined_list = [df.values for df in agg_moa_strength_count_pct_combined_list]

        agg_moa_strength_count_pct_combined = reduce(sum_data, agg_moa_strength_count_pct_combined_list)

        agg_moa_strength_count_pct_combined = pd.DataFrame(agg_moa_strength_count_pct_combined,
                                                            index=index)
        agg_moa_strength_count_pct_combined.columns = self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names

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
                               sort_by: List[SequenceStatisticsPlotFields 
                                             | UniqueSequenceFrequencyStatisticsPlotFields 
                                             | MeasureAssociationConfIntPlotFields
                                             | ClusterSizePlotFields],
                               sort_metric: SortMetric,
                               sorting_entity: SortingEntity,
                               ascending: bool) -> pd.DataFrame:
        """
        Sort entity by sort_by variable.by
        """

        data = data.copy()

        match sort_metric:
            case SortMetric.MEAN:
                sort_metric_func = np.mean

            case SortMetric.MEDIAN:
                sort_metric_func = np.median

            case SortMetric.MAX:
                sort_metric_func = np.max

            case SortMetric.MIN:
                sort_metric_func = np.min

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SortMetric.__name__}')
        
        match sorting_entity:
            case SortingEntity.GROUP:
                sorting_field = GROUP_FIELD_NAME_STR

            case SortingEntity.DATASET:
                sorting_field = DATASET_NAME_FIELD_NAME_STR

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SortingEntity.__name__}')

        sort_metric_values = data.groupby(sorting_field)[[field.value for field in sort_by]]\
                                 .agg(sort_metric_func)\
                                 .reset_index()

        # sort_metric_values.rename({sort_by.value: sort_metric.value}, 
        #                           axis=1, 
        #                           inplace=True)

        sort_metric_values.sort_values(by=[field.value for field in sort_by], inplace=True, ascending=ascending)

        data[sorting_field] = pd.Categorical(data[sorting_field], categories=sort_metric_values[sorting_field], ordered=True)
        data.sort_values(by=sorting_field, inplace=True)

        return data
    
    def _return_moa_strength_counter(self) -> defaultdict[str, int]:

        moa_strength_counter = defaultdict(int)
        for moa_str in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES:
            moa_strength_counter[moa_str]

        return moa_strength_counter
    
    def _return_p_val_is_significant_field_name(self) -> str:

        if RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_CORRECT_P_VALUES:
            p_val_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND.value + OMNIBUS_TESTS_PVAL_CORRECTED_FIELD_NAME_STR + RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_CORRECTION_METHOD.value
        else:
            p_val_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND.value

        p_val_is_significant_field_name = p_val_field_name + OMNIBUS_TESTS_PVAL_IS_SIGNIFICANT_FIELD_NAME_STR
    
        return p_val_is_significant_field_name
    
    def _return_result_aggregation_omnibus_test_pval_sig_moa_strength_guide_field_names(self,
                                                                                        eval_metric_is_categorical: bool) -> Tuple[str, Tuple[str, str, str]]:

        p_val_is_significant_field_name = self._return_p_val_is_significant_field_name()

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

    def _return_result_aggregation_omnibus_test_pval_sig_moa_field_names(self,
                                                                         eval_metric_is_categorical: bool) -> Tuple[str]:

        p_val_is_significant_field_name = self._return_p_val_is_significant_field_name()


        if eval_metric_is_categorical:

            moa_value = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY.value + 
                         OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR)
            moa_conf_int = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY.value + 
                            OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_VALUE_FIELD_NAME_STR)
        else:

            moa_value = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV.value + 
                         OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR)
            moa_conf_int = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV.value + 
                            OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_VALUE_FIELD_NAME_STR)

        return (p_val_is_significant_field_name, moa_value, moa_conf_int)
    
    def _return_measure_association_strength_guideline_boundary_dict(self,
                                                                     eval_metric_is_categorical: bool) -> Dict[MeasureAssociationStrengthValuesEnum, Tuple[float, float]]:

        if eval_metric_is_categorical:

            moa_guideline = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_CONTINGENCY

            match moa_guideline:
                case ContingencyMeasureAssociationStrengthGuidelineEnum.COHEN_1988:
                    boundary_dict = Cohen1988MeasureAssociationStrengthContingency.association_strength_values

                case ContingencyMeasureAssociationStrengthGuidelineEnum.GIGNAC_SZODORAI_2016:
                    boundary_dict = GignacSzodorai2016MeasureAssociationStrengthContingency.association_strength_values

                case ContingencyMeasureAssociationStrengthGuidelineEnum.FUNDER_OZER_2019:
                    boundary_dict = FunderOzer2019MeasureAssociationStrengthContingency.association_strength_values

                case ContingencyMeasureAssociationStrengthGuidelineEnum.LOVAKOV_AGADULLINA_2021:
                    boundary_dict = LovakovAgadullina2021MeasureAssociationStrengthContingency.association_strength_values

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ContingencyMeasureAssociationStrengthGuidelineEnum.__name__}')

        else:

            moa_guideline = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_AOV

            match moa_guideline:
                case AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988:
                    boundary_dict = Cohen1988MeasureAssociationStrengthAOV.association_strength_values

                case AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988_F:
                    boundary_dict = Cohen1988FMeasureAssociationStrengthAOV.association_strength_values

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{AOVMeasureAssociationStrengthGuidelineEnum.__name__}')
        
        return boundary_dict
    
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
                        x_axis_ticks_labels: NDArray[np.number] | None,
                        y_axis_ticks_labels: NDArray[np.number] | None,
                        x_axis_ticks_position: NDArray[np.number] | None  = None,
                        y_axis_ticks_position: NDArray[np.number] | None  = None,
                        x_rotation: int = 0,
                        y_rotation: int = 0) -> None:

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

        if x_axis_ticks_labels is not None:
            if x_axis_ticks_position is None:
                ax.set_xticks(x_axis_ticks_labels)
            else:
                ax.set_xticks(x_axis_ticks_position)
            ax.set_xticklabels([f'{tick}' for tick in x_axis_ticks_labels])

        if y_axis_ticks_labels is not None:
            if y_axis_ticks_position is None:
                ax.set_yticks(y_axis_ticks_labels)
            else:
                ax.set_yticks(y_axis_ticks_position)
            ax.set_yticklabels([f'{tick}' for tick in y_axis_ticks_labels])
        
        plt.setp(ax.get_xticklabels(), rotation=x_rotation)
        plt.setp(ax.get_yticklabels(), rotation=y_rotation)

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
    
    def _plot_kde(self,
                  field_data: np.ndarray,
                  field: SequenceStatisticsPlotFields,
                  data_range_limits: Tuple[float],
                  outer_color: str,
                  outer_linewidth: int | float,
                  outer_alpha: float,
                  inner_color: str,
                  inner_linewidth: int | float,
                  inner_alpha: float,
                  bottom_color: str,
                  bottom_linewidth: int | float,
                  bottom_alpha: float,
                  fill_color: Tuple[float],
                  fill_alpha: int | float | None,
                  include_bottom_line: bool,
                  apply_boundary_reflection: bool, 
                  bw_method: str | int | float,
                  bw_cut: int | float,
                  shift_value: int | float,
                  zorder: int | float) -> None:

        x_min = min(field_data)
        x_max = max(field_data)

        # kde
        if apply_boundary_reflection:
            field_data = self._kde_reflect_data(field_data,
                                                field)
        kde = sp.stats.gaussian_kde(field_data, 
                                    bw_method=bw_method)

        bandwidth = kde.factor * np.std(field_data)
        x_lower_bound = max(x_min - bandwidth * bw_cut, data_range_limits[0])
        x_upper_bound = min(x_max + bandwidth * bw_cut, data_range_limits[1])

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
                     color=outer_color,
                     linewidth=outer_linewidth,
                     alpha=outer_alpha,
                     zorder=zorder)
        sns.lineplot(x=x_vals, 
                     y=y_vals_shifted,
                     color=inner_color,
                     linewidth=inner_linewidth,
                     alpha=inner_alpha,
                     zorder=zorder+0.5)
        if include_bottom_line:
            sns.lineplot(x=x_vals, 
                         y=np.zeros_like(x_vals) + shift_value,
                         color=bottom_color,
                         linewidth=bottom_linewidth,
                         alpha=bottom_alpha,
                         zorder=zorder+0.5)
        plt.fill_between(x_vals, 
                         y_vals_shifted, 
                         shift_value, 
                         color=fill_color, 
                         zorder=zorder,
                         alpha=fill_alpha)

    def _plot_kde_mockup(self,
                         x_vals: np.ndarray,
                         y_vals: np.ndarray,
                         outer_color: str,
                         outer_linewidth: int | float,
                         outer_alpha: float,
                         inner_color: str,
                         inner_linewidth: int | float,
                         inner_alpha: float,
                         bottom_color: str,
                         bottom_linewidth: int | float,
                         bottom_alpha: float,
                         fill_color: Tuple[float],
                         fill_alpha: int | float | None,
                         include_bottom_line: bool,
                         shift_value: int | float,
                         zorder: int | float) -> None:


        # kde
        sns.lineplot(x=x_vals, 
                     y=y_vals,
                     color=outer_color,
                     linewidth=outer_linewidth,
                     alpha=outer_alpha,
                     zorder=zorder)
        sns.lineplot(x=x_vals, 
                     y=y_vals,
                     color=inner_color,
                     linewidth=inner_linewidth,
                     alpha=inner_alpha,
                     zorder=zorder+0.5)
        if include_bottom_line:
            sns.lineplot(x=x_vals, 
                         y=np.zeros_like(x_vals),
                         color=bottom_color,
                         linewidth=bottom_linewidth,
                         alpha=bottom_alpha,
                         zorder=zorder+0.5)
        plt.fill_between(x_vals, 
                         y_vals, 
                         shift_value, 
                         color=fill_color, 
                         zorder=zorder,
                         alpha=fill_alpha)
    
    def _kde_reflect_data(self,
                          data: NDArray,
                          field: SequenceStatisticsPlotFields) -> NDArray[np.number]:
        match field:
            case SequenceStatisticsPlotFields.SEQUENCE_LENGTH:

                data_lower_reflection = -data

                data = np.concatenate([data_lower_reflection, 
                                       data])

            case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                    SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                data_lower_reflection = -data
                data_upper_reflection = 200 - data

                data = np.concatenate([data_lower_reflection, 
                                       data, 
                                       data_upper_reflection])

            case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES |\
                    SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES:

                data_lower_reflection = -data
                data_upper_reflection = 2 - data

                data = np.concatenate([data_lower_reflection, 
                                       data, 
                                       data_upper_reflection])

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SequenceStatisticsPlotFields.__name__}')
            
        return data
    
    def _kde_get_line_width_in_data_coordinates(self,
                                                ax: matplotlib.axes.Axes, 
                                                dpi: int, 
                                                linewidth_in_points: int) -> float:
        linewidth_in_inches = linewidth_in_points / 72
        linewidth_in_pixels = linewidth_in_inches * dpi
        y_0 = 0
        y_1_pix = ax.transData.transform((0, y_0))[1] + linewidth_in_pixels
        y_1 = ax.transData.inverted().transform((0, y_1_pix))[1]
        line_width =  y_1 - y_0

        return line_width

    def _kde_draw_iqr_range_conf_int_box(self,
                                         ax: matplotlib.axes.Axes,
                                         dataset_name: str,
                                         group: int,
                                         sequence_statistics_per_group_per_dataset: pd.DataFrame,
                                         field: SequenceStatisticsPlotFields,
                                         shift_value: int | float,
                                         zorder: int | float) -> None:
        
        field_data = sequence_statistics_per_group_per_dataset[field.value].values

        # iqr and range plot data 
        box_height_iqr = self._kde_get_line_width_in_data_coordinates(ax,
                                                                      RESULT_AGGREGATION_FIG_SIZE_DPI,
                                                                      SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_HEIGHT_IN_LINEWIDTH)
        box_height_range = self._kde_get_line_width_in_data_coordinates(ax,
                                                                        RESULT_AGGREGATION_FIG_SIZE_DPI,
                                                                        SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_HEIGHT_IN_LINEWIDTH)

        # iqr/range box
        match SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_BOX_KIND:
            case BoxKind.IQR:
                iqr_box_data = self._return_iqr_box_data(field_data,
                                                         box_height_iqr,
                                                         shift_value)
                self._plot_iqr_range(ax,
                                     iqr_box_data,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGE_LINEWIDTH,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGECOLOR,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_FACECOLOR,
                                     zorder+0.6)

            case BoxKind.RANGE:
                range_box_data = self._return_range_box_data(field_data,
                                                             box_height_range,
                                                             shift_value)
                self._plot_iqr_range(ax,
                                     range_box_data,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_EDGE_LINEWIDTH,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_EDGECOLOR,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_FACECOLOR,
                                     zorder+0.5)

            case BoxKind.BOTH:
                iqr_box_data = self._return_iqr_box_data(field_data,
                                                         box_height_iqr,
                                                         shift_value)
                range_box_data = self._return_range_box_data(field_data,
                                                             box_height_range,
                                                             shift_value)
                self._plot_iqr_range(ax,
                                     iqr_box_data,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGE_LINEWIDTH,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_EDGECOLOR,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_IQR_BOX_FACECOLOR,
                                     zorder+0.6)

                self._plot_iqr_range(ax,
                                     range_box_data,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_EDGE_LINEWIDTH,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_EDGECOLOR,
                                     SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_RANGE_BOX_FACECOLOR,
                                     zorder+0.5)
        
            case BoxKind.NONE:
                pass

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{BoxKind.__name__}')

        # confidence interval
        match SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_KIND:
            case ConfidenceIntervalKind.MEAN:
                seq_stat_conf_int_res = self._calculate_conf_int_per_sequence_statistic(dataset_name,
                                                                                        SequenceType.ALL_SEQUENCES,
                                                                                        field,
                                                                                        ConfIntEstimator.MEAN,
                                                                                        group)

                single_conf_int_box_data = self._return_single_conf_int_box_data(seq_stat_conf_int_res,
                                                                                 box_height_iqr,
                                                                                 shift_value)
                self._plot_single_conf_int(ax,
                                           single_conf_int_box_data,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_EDGE_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_UPPER_BOX_FACECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_ALPHA,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_COLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SINGLE_BOX_SCATTER_SIZE,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SINGLE_BOX_SCATTER_MARKER,
                                           zorder)

            case ConfidenceIntervalKind.MEDIAN:
                seq_stat_conf_int_res = self._calculate_conf_int_per_sequence_statistic(dataset_name,
                                                                                        SequenceType.ALL_SEQUENCES,
                                                                                        field,
                                                                                        ConfIntEstimator.MEDIAN,
                                                                                        group)

                single_conf_int_box_data = self._return_single_conf_int_box_data(seq_stat_conf_int_res,
                                                                                 box_height_iqr,
                                                                                 shift_value)
                self._plot_single_conf_int(ax,
                                           single_conf_int_box_data,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_EDGE_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_LOWER_BOX_FACECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_ALPHA,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_COLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SINGLE_BOX_SCATTER_SIZE,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_LINEWIDTH,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_EDGECOLOR,
                                           SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_SINGLE_BOX_SCATTER_MARKER,
                                           zorder)

            case ConfidenceIntervalKind.BOTH:
                upper_seq_stat_conf_int_res = self._calculate_conf_int_per_sequence_statistic(dataset_name,
                                                                                              SequenceType.ALL_SEQUENCES,
                                                                                              field,
                                                                                              ConfIntEstimator.MEAN,
                                                                                              group)

                lower_seq_stat_conf_int_res = self._calculate_conf_int_per_sequence_statistic(dataset_name,
                                                                                              SequenceType.ALL_SEQUENCES,
                                                                                              field,
                                                                                              ConfIntEstimator.MEDIAN,
                                                                                              group)

                dual_conf_int_box_data = self._return_dual_conf_int_box_data(upper_seq_stat_conf_int_res,
                                                                             lower_seq_stat_conf_int_res,
                                                                             box_height_iqr,
                                                                             shift_value)
                self._plot_dual_conf_int(ax,
                                         dual_conf_int_box_data,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_EDGE_LINEWIDTH,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_EDGECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_UPPER_BOX_FACECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_LOWER_BOX_FACECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_ALPHA,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_COLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_DUAL_BOX_SCATTER_SIZE,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_LINEWIDTH,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_BOX_SCATTER_EDGECOLOR,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_DUAL_UPPER_BOX_SCATTER_MARKER,
                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_CONF_INT_DUAL_LOWER_BOX_SCATTER_MARKER,
                                         zorder)

            case ConfidenceIntervalKind.NONE:
                pass

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ConfidenceIntervalKind.__name__}')

    def _return_iqr_box_data(self,
                             field_data: np.ndarray,
                             box_height_iqr: float,
                             shift_value: int | float) -> IQRRangeBoxData:

        first_quartile = np.quantile(field_data, 0.25)
        third_quartile = np.quantile(field_data, 0.75)
        y_start_iqr = shift_value - box_height_iqr / 2
        y_end_iqr = shift_value + box_height_iqr / 2
        iqr_val = third_quartile - first_quartile

        return IQRRangeBoxData(first_quartile,
                               third_quartile,
                               y_start_iqr,
                               y_end_iqr,
                               box_height_iqr,
                               iqr_val)

    def _return_range_box_data(self,
                               field_data: np.ndarray,
                               box_height_range: float,
                               shift_value: int | float) -> IQRRangeBoxData:

        x_min = min(field_data)
        x_max = max(field_data)
        y_start_range = shift_value - box_height_range / 2
        y_end_range = shift_value + box_height_range / 2
        range_val = x_max - x_min

        return IQRRangeBoxData(x_min,
                               x_max,
                               y_start_range,
                               y_end_range,
                               box_height_range,
                               range_val)

    def _return_iqr_box_data_mockup(self,
                                    mu: int | float,
                                    sigma: int | float,
                                    box_height_iqr: float,
                                    shift_value: int | float) -> IQRRangeBoxData:

        first_quartile = sp.stats.norm.ppf(0.25, 
                                           loc=mu, 
                                           scale=sigma)
        third_quartile = sp.stats.norm.ppf(0.75, 
                                           loc=mu, 
                                           scale=sigma)
        y_start_iqr = shift_value - box_height_iqr / 2
        y_end_iqr = shift_value + box_height_iqr / 2
        iqr_val = third_quartile - first_quartile

        return IQRRangeBoxData(first_quartile,
                               third_quartile,
                               y_start_iqr,
                               y_end_iqr,
                               box_height_iqr,
                               iqr_val)

    def _return_range_box_data_mockup(self,
                                      mu: int | float,
                                      sigma: int | float,
                                      lower_quantile: float,
                                      upper_quantile: float,
                                      box_height_range: float,
                                      shift_value: int | float) -> IQRRangeBoxData:

        x_min = sp.stats.norm.ppf(lower_quantile, 
                                  loc=mu, 
                                  scale=sigma)
        x_max = sp.stats.norm.ppf(upper_quantile, 
                                  loc=mu, 
                                  scale=sigma)
        y_start_range = shift_value - box_height_range / 2
        y_end_range = shift_value + box_height_range / 2
        range_val = x_max - x_min

        return IQRRangeBoxData(x_min,
                               x_max,
                               y_start_range,
                               y_end_range,
                               box_height_range,
                               range_val)

    def _return_single_conf_int_box_data(self,
                                         seq_stat_conf_int_res: SequenceStatisticConfIntResult,
                                         box_height_iqr: float,
                                         shift_value: int | float) -> SingleConfIntBoxData:
        ci_x_start = seq_stat_conf_int_res.conf_int_lower_bound
        ci_x_end = seq_stat_conf_int_res.conf_int_upper_bound
        ci_y_start = shift_value - box_height_iqr / 2
        ci_y_end = shift_value + box_height_iqr / 2
        ci_box_height = box_height_iqr
        ci_width = seq_stat_conf_int_res.conf_int_upper_bound - seq_stat_conf_int_res.conf_int_lower_bound
        statistic_x_value = seq_stat_conf_int_res.statistic_value
        statistic_y_value = ci_y_start + ci_box_height / 2

        return SingleConfIntBoxData(ci_x_start,
                                    ci_x_end,
                                    ci_y_start,
                                    ci_y_end,
                                    ci_box_height,
                                    ci_width,
                                    statistic_x_value,
                                    statistic_y_value)

    def _return_dual_conf_int_box_data(self,
                                       upper_seq_stat_conf_int_res: SequenceStatisticConfIntResult,
                                       lower_seq_stat_conf_int_res: SequenceStatisticConfIntResult,
                                       box_height_iqr: float,
                                       shift_value: int | float) -> DualConfIntBoxData:

        ci_upper_x_start = upper_seq_stat_conf_int_res.conf_int_lower_bound
        ci_upper_y_start = shift_value 
        ci_lower_x_start = lower_seq_stat_conf_int_res.conf_int_lower_bound
        ci_lower_y_start = shift_value - box_height_iqr / 2
        ci_box_height = box_height_iqr / 2
        ci_upper_width = upper_seq_stat_conf_int_res.conf_int_upper_bound - upper_seq_stat_conf_int_res.conf_int_lower_bound
        ci_lower_width = lower_seq_stat_conf_int_res.conf_int_upper_bound - lower_seq_stat_conf_int_res.conf_int_lower_bound
        statistic_upper_x_value = upper_seq_stat_conf_int_res.statistic_value
        statistic_upper_y_value = ci_upper_y_start + ci_box_height / 2
        statistic_lower_x_value = lower_seq_stat_conf_int_res.statistic_value
        statistic_lower_y_value = ci_lower_y_start + ci_box_height / 2


        return DualConfIntBoxData(ci_upper_x_start,
                                  ci_upper_y_start,
                                  ci_lower_x_start,
                                  ci_lower_y_start,
                                  ci_box_height,
                                  ci_upper_width,
                                  ci_lower_width,
                                  statistic_upper_x_value,
                                  statistic_upper_y_value,
                                  statistic_lower_x_value,
                                  statistic_lower_y_value)

    def _plot_iqr_range(self,
                        ax: matplotlib.axes.Axes,
                        box_data: IQRRangeBoxData,
                        linewidth: int | float,
                        edgecolor: str,
                        facecolor: str,
                        zorder: int | float) -> None:

        rectangle = Rectangle((box_data.x_start, box_data.y_start),
                              box_data.box_width,
                              box_data.box_height,
                              linewidth=linewidth,
                              edgecolor=edgecolor,
                              facecolor=facecolor,
                              zorder=zorder)
        ax.add_patch(rectangle)

    def _plot_single_conf_int(self,
                              ax: matplotlib.axes.Axes,
                              single_conf_int_box_data: SingleConfIntBoxData,
                              box_linewidth: int | float,
                              box_edgecolor: 'str',
                              box_facecolor: str,
                              box_alpha: float,
                              marker_color: str,
                              marker_size: int | float,
                              marker_linewidth: int | float,
                              marker_edgecolor: str,
                              marker_kind: str,
                              zorder: int | float) -> None:

        rectangle_single_conf_int = Rectangle((single_conf_int_box_data.x_start, single_conf_int_box_data.y_start),
                                              single_conf_int_box_data.box_width,
                                              single_conf_int_box_data.box_height,
                                              linewidth=box_linewidth,
                                              edgecolor=box_edgecolor,
                                              facecolor=box_facecolor,
                                              alpha=box_alpha,
                                              zorder=zorder+0.7)
        ax.add_patch(rectangle_single_conf_int)
        sns.scatterplot(x=[single_conf_int_box_data.statistic_x_value],
                        y=[single_conf_int_box_data.statistic_y_value],
                        legend=False,
                        color=marker_color,
                        s=marker_size,
                        linewidth=marker_linewidth,
                        edgecolor=marker_edgecolor,
                        marker=marker_kind,
                        zorder=zorder+0.9)

    def _plot_dual_conf_int(self,
                            ax: matplotlib.axes.Axes,
                            dual_conf_int_box_data: DualConfIntBoxData,
                            box_linewidth: int | float,
                            box_edgecolor: 'str',
                            box_facecolor_upper: str,
                            box_facecolor_lower: str,
                            box_alpha: float,
                            marker_color: str,
                            marker_size: int | float,
                            marker_linewidth: int | float,
                            marker_edgecolor: str,
                            marker_kind_upper: str,
                            marker_kind_lower: str,
                            zorder: int | float) -> None:

        # upper
        rectangle_upper_conf_int = Rectangle((dual_conf_int_box_data.ci_upper_x_start, dual_conf_int_box_data.ci_upper_y_start),
                                             dual_conf_int_box_data.ci_upper_width,
                                             dual_conf_int_box_data.ci_box_height,
                                             linewidth=box_linewidth,
                                             edgecolor=box_edgecolor,
                                             facecolor=box_facecolor_upper,
                                             alpha=box_alpha,
                                             zorder=zorder+0.7)
        ax.add_patch(rectangle_upper_conf_int)
        sns.scatterplot(x=[dual_conf_int_box_data.statistic_upper_x_value],
                        y=[dual_conf_int_box_data.statistic_upper_y_value],
                        legend=False,
                        color=marker_color,
                        s=marker_size,
                        linewidth=marker_linewidth,
                        edgecolor=marker_edgecolor,
                        marker=marker_kind_upper,
                        zorder=zorder+0.9)
        # lower
        rectangle_lower_conf_int = Rectangle((dual_conf_int_box_data.ci_lower_x_start, dual_conf_int_box_data.ci_lower_y_start),
                                              dual_conf_int_box_data.ci_lower_width,
                                              dual_conf_int_box_data.ci_box_height,
                                              linewidth=box_linewidth,
                                              edgecolor=box_edgecolor,
                                              facecolor=box_facecolor_lower,
                                              alpha=box_alpha,
                                              zorder=zorder+0.7)
        ax.add_patch(rectangle_lower_conf_int)
        sns.scatterplot(x=[dual_conf_int_box_data.statistic_lower_x_value],
                        y=[dual_conf_int_box_data.statistic_lower_y_value],
                        legend=False,
                        color=marker_color,
                        s=marker_size,
                        linewidth=marker_linewidth,
                        edgecolor=marker_edgecolor,
                        marker=marker_kind_lower,
                        zorder=zorder+0.9)

    def _calculate_conf_int_per_sequence_statistic(self,
                                                   dataset_name: str,
                                                   sequence_type: SequenceType,
                                                   field: SequenceStatisticsPlotFields,
                                                   estimator: ConfIntEstimator,
                                                   group: int) -> SequenceStatisticConfIntResult:

        match sequence_type:
            case SequenceType.UNIQUE_SEQUENCES:
                use_unique_sequences = True

            case SequenceType.ALL_SEQUENCES:
                use_unique_sequences = False

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SequenceType.__name__}')

        seq_stat_conf_int_df = self._return_sequence_statistics_confidence_interval_per_group_per_dataset_df(use_unique_sequences)

        dataset_filter = seq_stat_conf_int_df[ConfIntResultFields.DATASET_NAME.value] == dataset_name
        field_filter = seq_stat_conf_int_df[ConfIntResultFields.SEQUENCE_STATISTIC.value] == field.value
        estimator_filter = seq_stat_conf_int_df[ConfIntResultFields.ESTIMATOR.value] == estimator.value
        group_filter = seq_stat_conf_int_df[ConfIntResultFields.GROUP.value] == group

        seq_stat_conf_int_df_filter = dataset_filter & field_filter & estimator_filter & group_filter

        seq_stat_conf_int_result = seq_stat_conf_int_df.loc[seq_stat_conf_int_df_filter, ConfIntResultFields.RESULT.value].iloc[0]

        return seq_stat_conf_int_result

    def _return_conf_int_mock_up(self,
                                 mu: int | float,
                                 iqr_box_width: float,
                                 conf_int_estimator: ConfIntEstimator) -> SequenceStatisticConfIntResult:

        match conf_int_estimator:
            case ConfIntEstimator.MEAN:
                conf_int_multiplier = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_CI_MEAN_RATIO_OF_BOX_WIDTH

            case ConfIntEstimator.MEDIAN:
                conf_int_multiplier = SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_DATA_CI_MEDIAN_RATIO_OF_BOX_WIDTH

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ConfIntEstimator.__name__}')

        conf_int_width = iqr_box_width * conf_int_multiplier

        ci_lower = mu - conf_int_width / 2
        ci_upper = mu + conf_int_width / 2

        return SequenceStatisticConfIntResult(mu,
                                              ci_lower,
                                              ci_upper,
                                              None,
                                              None,
                                              None,
                                              None)
    
    def _return_mock_up_dist_data(self,
                                  mu: int | float,
                                  sigma: int | float,
                                  data_range_limits: Tuple[int | float],
                                  n_data_points: int) -> MockUpDistData:

        x_vals = np.linspace(data_range_limits[0], 
                             data_range_limits[1], 
                             n_data_points)
        y_vals = sp.stats.norm.pdf(x_vals, 
                                   loc=mu, 
                                   scale=sigma)

        return MockUpDistData(mu,
                              sigma,
                              x_vals,
                              y_vals)
    
    def _add_mock_up_annotation_box_bracket(self,
                                            ax: matplotlib.axes.Axes,
                                            box_data: IQRRangeBoxData | SingleConfIntBoxData,
                                            y_offset_from_box: int | float,
                                            annotation_is_below_element: bool,
                                            bracket_height: int | float,
                                            text_y_offset_from_arrow: int | float,
                                            annotation_str: str,
                                            str_fontsize: int,
                                            str_box_linewidth: int | float,
                                            box_style: str,
                                            arrow_linewidth: int | float,
                                            zorder: int | float) -> None:

        x_coordinates = [box_data.x_start,
                         box_data.x_start,
                         box_data.x_end,
                         box_data.x_end]
        
        if annotation_is_below_element:
            y_offset_from_box *= -1
            bracket_height *= -1
            text_y_offset_from_arrow *= -1
        else:
            y_offset_from_box += box_data.box_height


        y_start = box_data.y_start + y_offset_from_box
        y_end = box_data.y_start + y_offset_from_box + bracket_height
        
        y_coordinates = [y_start,
                         y_end,
                         y_end,
                         y_start]

        ax.plot(x_coordinates,
                y_coordinates, 
                color='k', 
                lw=arrow_linewidth,
                zorder=zorder)

        ax.annotate(annotation_str,
                    xy=((box_data.x_start + box_data.x_end)/2, y_end),
                    xytext=((box_data.x_start + box_data.x_end)/2, y_end + text_y_offset_from_arrow),
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle=f'{box_style},pad=0.3', fc='white', ec='black', lw=str_box_linewidth),
                    fontsize=str_fontsize,
                    arrowprops=dict(arrowstyle='-', color='black', lw=arrow_linewidth),
                    annotation_clip=False,
                    zorder=zorder)

    def _add_mock_up_annotation_arrow_single_central_tendency_marker(self,
                                                                     ax: matplotlib.axes.Axes,
                                                                     box_data: SingleConfIntBoxData,
                                                                     figure_proportion_text: Tuple[float, float],
                                                                     annotation_str: str,
                                                                     str_fontsize: int,
                                                                     str_box_linewidth: int | float,
                                                                     box_style: str,
                                                                     arrow_linewidth: int | float,
                                                                     zorder: int | float) -> None:
        
        ax.annotate(annotation_str,
                    xy=(box_data.statistic_x_value, box_data.statistic_y_value),
                    xytext=figure_proportion_text,
                    xycoords='data',
                    textcoords='figure fraction',
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle=f'{box_style},pad=0.3', fc='white', ec='black', lw=str_box_linewidth),
                    fontsize=str_fontsize,
                    arrowprops=dict(arrowstyle='->', color='black', lw=arrow_linewidth),
                    annotation_clip=False,
                    zorder=zorder)

    def _add_mock_up_annotation_arrow_dual_central_tendency_marker(self,
                                                                   ax: matplotlib.axes.Axes,
                                                                   box_data: DualConfIntBoxData,
                                                                   figure_proportion_upper_text: Tuple[float, float],
                                                                   figure_proportion_lower_text: Tuple[float, float],
                                                                   annotation_str_upper: str,
                                                                   annotation_str_lower: str,
                                                                   str_fontsize: int,
                                                                   str_box_linewidth: int | float,
                                                                   box_style: str,
                                                                   arrow_linewidth: int | float,
                                                                   zorder: int | float) -> None:
        
        ax.annotate(annotation_str_upper,
                    xy=(box_data.statistic_upper_x_value, box_data.statistic_upper_y_value),
                    xytext=figure_proportion_upper_text,
                    xycoords='data',
                    textcoords='figure fraction',
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle=f'{box_style},pad=0.3', fc='white', ec='black', lw=str_box_linewidth),
                    fontsize=str_fontsize,
                    arrowprops=dict(arrowstyle='->', color='black', lw=arrow_linewidth),
                    annotation_clip=False,
                    zorder=zorder)

        ax.annotate(annotation_str_lower,
                    xy=(box_data.statistic_lower_x_value, box_data.statistic_lower_y_value),
                    xytext=figure_proportion_lower_text,
                    xycoords='data',
                    textcoords='figure fraction',
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle=f'{box_style},pad=0.3', fc='white', ec='black', lw=str_box_linewidth),
                    fontsize=str_fontsize,
                    arrowprops=dict(arrowstyle='->', color='black', lw=arrow_linewidth),
                    annotation_clip=False,
                    zorder=zorder)

    def _add_mock_up_annotation_arrow_kde(self,
                                          ax: matplotlib.axes.Axes,
                                          mock_up_dist_data: MockUpDistData,
                                          quantile_of_data: float,
                                          figure_proportion_text: Tuple[float, float],
                                          annotation_str: str,
                                          str_fontsize: int,
                                          str_box_linewidth: int | float,
                                          box_style: str,
                                          arrow_linewidth: int | float,
                                          zorder: int | float) -> None:
        
        x_value = np.quantile(mock_up_dist_data.x_values,
                              quantile_of_data,
                              method='closest_observation')
        index_quantile = np.where(mock_up_dist_data.x_values == x_value)[0][0]
        y_value = mock_up_dist_data.y_values[index_quantile]
        
        ax.annotate(annotation_str,
                    xy=(x_value, y_value),
                    xytext=figure_proportion_text,
                    xycoords='data',
                    textcoords='figure fraction',
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle=f'{box_style},pad=0.3', fc='white', ec='black', lw=str_box_linewidth),
                    fontsize=str_fontsize,
                    arrowprops=dict(arrowstyle='->', color='black', lw=arrow_linewidth),
                    annotation_clip=False,
                    zorder=zorder)

    def _add_mock_up_annotations(self,
                                 ax: matplotlib.axes.Axes,
                                 single_conf_int_box_data_mean: SingleConfIntBoxData,
                                 single_conf_int_box_data_median: SingleConfIntBoxData,
                                 dual_conf_int_box_data: DualConfIntBoxData,
                                 mock_up_data: MockUpDistData,
                                 iqr_box_data: IQRRangeBoxData,
                                 range_box_data: IQRRangeBoxData,
                                 zorder) -> None:

        match SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_CONF_INT_KIND:
            case ConfidenceIntervalKind.MEAN:

                self._add_mock_up_annotation_arrow_single_central_tendency_marker(ax,
                                                                                  single_conf_int_box_data_mean,
                                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_SINGLE_CONF_INT_MEAN_MARKER_TEXT_POSITION_PROPORTION,
                                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_SINGLE_CONF_INT_MEAN_MARKER_TEXT_STR,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                                                  zorder+1)

                self._add_mock_up_annotation_box_bracket(ax,
                                                         single_conf_int_box_data_mean,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_Y_OFFSET_FROM_BOX,
                                                         True,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_BRACKET_HEIGTH,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_TEXT_Y_OFFSET_FROM_BRACKET ,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_TEXT_STR,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                         zorder+1)

            case ConfidenceIntervalKind.MEDIAN:

                self._add_mock_up_annotation_arrow_single_central_tendency_marker(ax,
                                                                                  single_conf_int_box_data_median,
                                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_SINGLE_CONF_INT_MEDIAN_MARKER_TEXT_POSITION_PROPORTION,
                                                                                  SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_SINGLE_CONF_INT_MEDIAN_MARKER_TEXT_STR,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                                                  EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                                                  zorder+1)

                self._add_mock_up_annotation_box_bracket(ax,
                                                         single_conf_int_box_data_median,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_Y_OFFSET_FROM_BOX,
                                                         True,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_BRACKET_HEIGTH,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_TEXT_Y_OFFSET_FROM_BRACKET,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_TEXT_STR,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                         zorder+1)

            case ConfidenceIntervalKind.BOTH:

                self._add_mock_up_annotation_arrow_dual_central_tendency_marker(ax,
                                                                                dual_conf_int_box_data,
                                                                                SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_DUAL_CONF_INT_UPPER_MARKER_TEXT_POSITION_PROPORTION,
                                                                                SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_DUAL_CONF_INT_LOWER_MARKER_TEXT_POSITION_PROPORTION,
                                                                                SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_DUAL_CONF_INT_UPPER_MARKER_TEXT_STR,
                                                                                SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_DUAL_CONF_INT_LOWER_MARKER_TEXT_STR,
                                                                                EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                                                EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                                                EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                                                EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                                                zorder+1)

                self._add_mock_up_annotation_box_bracket(ax,
                                                         single_conf_int_box_data_mean,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_Y_OFFSET_FROM_BOX,
                                                         False,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_BRACKET_HEIGTH,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_TEXT_Y_OFFSET_FROM_BRACKET ,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEAN_TEXT_STR,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                         zorder+1)

                self._add_mock_up_annotation_box_bracket(ax,
                                                         single_conf_int_box_data_median,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_Y_OFFSET_FROM_BOX,
                                                         True,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_BRACKET_HEIGTH,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_TEXT_Y_OFFSET_FROM_BRACKET,
                                                         SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_CONF_INT_MEDIAN_TEXT_STR,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                         EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                         zorder+1)

            case ConfidenceIntervalKind.NONE:
                pass

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{ConfidenceIntervalKind.__name__}')

        self._add_mock_up_annotation_arrow_kde(ax,
                                               mock_up_data,
                                               SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_KDE_ARROW_POSITION_QUANTILE,
                                               SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_KDE_ARROW_TEXT_POSITION_PROPORTION,
                                               SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_KDE_ARROW_TEXT_STR,
                                               EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                               EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                               EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                               EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                               zorder+1)

        self._add_mock_up_annotation_box_bracket(ax,
                                                 iqr_box_data,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_IQR_Y_OFFSET_FROM_BOX,
                                                 True,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_IQR_BRACKET_HEIGTH,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_IQR_TEXT_Y_OFFSET_FROM_BRACKET,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_IQR_TEXT_STR,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                 zorder+1)

        self._add_mock_up_annotation_box_bracket(ax,
                                                 range_box_data,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_RANGE_Y_OFFSET_FROM_BOX,
                                                 True,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_RANGE_BRACKET_HEIGTH,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_RANGE_TEXT_Y_OFFSET_FROM_BRACKET,
                                                 SEQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_RANGE_TEXT_STR ,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXT_FONTSIZE,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_LINEWIDTH,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_TEXTBOX_STYLE,
                                                 EQUENCE_STATISTICS_DISTRIBUTION_RIDGEPLOT_MOCKUP_ANNOTATION_ARROW_LINEWIDTH,
                                                 zorder+1)

    def _return_moa_conf_int_plot_x_label(self,
                                          x_label_kind: OmnibusTestResultMeasureAssociationConfIntXLabelKind,
                                          facet_data: pd.DataFrame) -> str:

        match x_label_kind:
            case OmnibusTestResultMeasureAssociationConfIntXLabelKind.DYNAMIC:
                x_label = facet_data[RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_KIND_NAME_STR].iloc[0]
                x_label =  ' '.join([i.capitalize() for i in x_label.split('_')])

            case OmnibusTestResultMeasureAssociationConfIntXLabelKind.STATIC:
                x_label = x_label_kind.value
                x_label = x_label.replace('_', ' ').title()

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{OmnibusTestResultMeasureAssociationConfIntXLabelKind.__name__}')

        return x_label
    
    def _return_y_array_numerical(self,
                                  y_array: np.ndarray) -> np.ndarray:

        y_array_numerical = np.array(range(len(y_array))) * -1

        return y_array_numerical
    
    def _return_moa_strength_color_palette(self,
                                           color_palette: str, 
                                           col_palette_size: int,
                                           col_palette_index_min: int,
                                           col_palette_index_max: int,
                                           n_moa_strength_categories: int,
                                           desaturation: int | float,
                                           alpha: int | float | None,
                                           non_sig_color: str | None) -> List[Tuple[float]]:
        
        col_palette_index = np.linspace(col_palette_index_min, 
                                        col_palette_index_max, 
                                        n_moa_strength_categories).astype(int)
        col_palette_index = list(col_palette_index)

        color_palette = sns.color_palette(color_palette,
                                          n_colors=col_palette_size,
                                          desat=desaturation)
        color_palette = [color_palette[i] for i in col_palette_index]

        if non_sig_color is not None:
            rgb_col = mcolors.to_rgb(non_sig_color)
            color_palette.insert(0, rgb_col)

        if alpha is not None:
            color_palette = [i + (alpha,) for i in color_palette]

        return color_palette
    
    def _add_moa_strength_facet_grid_legend(self,
                                            g: FacetGrid,
                                            box_edgecolor: str,
                                            box_linewidth: int | float,
                                            box_length: int | float,
                                            box_height: int | float,
                                            custom_label: List[str] | None) -> None:

        handles = {}
        if custom_label is not None:
            for ax in g.axes.flat:
                for p, label in zip(ax.patches, custom_label):
                        facecolor = p.get_facecolor()
                        patch = Patch(facecolor=facecolor,
                                      edgecolor=box_edgecolor,
                                      linewidth=box_linewidth,
                                      label=label)
                        handles[label] = patch
        else:
            for ax in g.axes.flat:
                for p in ax.patches:
                    label = p.get_label()
                    if label != '_nolegend_':
                        facecolor = p.get_facecolor()
                        patch = Patch(facecolor=facecolor,
                                      edgecolor=box_edgecolor,
                                      linewidth=box_linewidth,
                                      label=label)
                        handles[label] = patch

        g.figure.legend(handles=handles.values(),
                        loc='lower center',
                        bbox_to_anchor=(.5, 1), 
                        title=None,
                        ncol=len(handles),
                        handlelength=box_length, 
                        handleheight=box_height,
                        frameon=False)
    
    def _add_moa_strength_legend(self,
                                 ax: matplotlib.axes.Axes,
                                 box_edgecolor: str,
                                 box_linewidth: int | float,
                                 box_length: int | float,
                                 box_height: int | float,
                                 n_cols: int,
                                 custom_labels: List[str] | None) -> None:

        if custom_labels is not None:
            legend = ax.legend(loc='lower center',
                               labels=custom_labels,
                               bbox_to_anchor=(.5, 1), 
                               title=None,
                               ncol=n_cols,
                               handlelength=box_length, 
                               handleheight=box_height,
                               frameon=False)
        else:
            legend = ax.legend(loc='lower center',
                               bbox_to_anchor=(.5, 1), 
                               title=None,
                               ncol=n_cols,
                               handlelength=box_length, 
                               handleheight=box_height,
                               frameon=False)

        for handle in legend.legend_handles:
            handle.set_linewidth(box_linewidth)
            handle.set_edgecolor(box_edgecolor)

    def _format_moa_strength_labels(self,
                                    moa_labels_raw: List[str],
                                    add_effect_size_str: bool,
                                    capitalize: bool) -> List[str]:
        if capitalize:
            effect_size_str = ' '.join([i.capitalize() for i in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_NAME_STR.split(' ')])
            moa_labels_raw = ['_'.join([j.capitalize() for j in i.split('_')]) for i in moa_labels_raw]
        else:
            effect_size_str = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_NAME_STR
        
        if not add_effect_size_str:
            effect_size_str = ''

        moa_streng_categories = ['-'.join(i.split('_')) + ' ' + effect_size_str for i in moa_labels_raw]

        return moa_streng_categories
    
    def _plot_cluster_size_scatter_legend(self,
                                          g: FacetGrid,
                                          marker_clustered: str,
                                          marker_non_clustered: str,
                                          label_clustered: str,
                                          label_non_clustered: str,
                                          edgecolor_clustered: str,
                                          edgecolor_non_clustered: str,
                                          markersize_clustered: int | float,
                                          markersize_non_clustered: int | float) -> None:

        label_clustered = label_clustered.lower()
        label_non_clustered = label_non_clustered.lower()       

        marker_clustered = Line2D([0], 
                                    [0], 
                                    linestyle='',
                                    marker=marker_clustered, 
                                    label=label_clustered,
                                    markerfacecolor='none', 
                                    markeredgecolor=edgecolor_clustered, 
                                    markersize=markersize_clustered)
        marker_non_clustered = Line2D([0], 
                                        [0], 
                                        linestyle='',
                                        marker=marker_non_clustered, 
                                        label=label_non_clustered,
                                        markerfacecolor='none', 
                                        markeredgecolor=edgecolor_non_clustered, 
                                        markersize=markersize_non_clustered)
        markers = [marker_clustered,
                   marker_non_clustered]

        g.figure.legend(handles=markers,
                        loc='lower center',
                        bbox_to_anchor=(.5, 1), 
                        title=None,
                        ncol=len(markers),
                        handletextpad=0,
                        frameon=False)