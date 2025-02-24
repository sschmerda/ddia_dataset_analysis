from .standard_import import *
from .constants.constants import *
from .configs.result_aggregation_config import *
from .plotting_functions import *
from .html_style_functions import *

class AggregatedResults():
    """docstring for ClassName."""
    def __init__(self):

        self._path_to_result_tables = self._return_result_tables_paths()

        self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names = [strength_value + '_count' for strength_value in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES]
        self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names = [strength_value + '_pct' for strength_value in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES]
        self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names = [strength_value + '_count_and_pct' for strength_value in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES]

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
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    False,
                                                    False)
                    x_axis_ticks = None

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                     SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = avg_sequence_statistics_per_group_per_dataset
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    True,
                                                    False)
                    x_axis_ticks = np.arange(0, 110, 10)

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = avg_sequence_statistics_per_group_per_dataset
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    True,
                                                    True)
                    x_axis_ticks = np.arange(0, 1.1, 0.1)

                case UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY:

                    data = avg_unique_sequence_frequency_statistics_per_group_per_dataset
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    False,
                                                    False)
                    x_axis_ticks = None

                case UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY:

                    data = avg_unique_sequence_frequency_statistics_per_group_per_dataset
                    x_axis_lim = return_axis_limits(data[field.value],
                                                    True,
                                                    False)
                    x_axis_ticks = np.arange(0, 110, 10)

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{field}')

            # boxplot
            g = sns.boxplot(
                            data, 
                            x=field.value, 
                            y=DATASET_NAME_FIELD_NAME_STR, 
                            hue=DATASET_NAME_FIELD_NAME_STR,
                            palette=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_PALETTE,
                            showfliers=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_OUTLIERS,
                            linewidth=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_LINE_WIDTH,
                            width=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WIDTH,
                            whis=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_BOX_WHISKERS,
                            showmeans=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_SHOW_MEANS,
                            meanprops=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_MARKER,
                            saturation=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_BOXPLOT_COLOR_SATURATION,
                           )
            # strip or swarmplot
            g = sns.swarmplot(
                              data, 
                              x=field.value, 
                              y=DATASET_NAME_FIELD_NAME_STR, 
                              size=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_SIZE, 
                              color=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_COLOR,
                              alpha=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_ALPHA,
                              edgecolor=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_EDGECOLOR,
                              linewidth=AVG_SEQUENCE_STATISTICS_PER_GROUP_PER_DATASET_SWARMPLOT_POINT_LINEWIDTH,
                             )
            g.set(
                xlabel=x_label,
                ylabel='',
                xlim=x_axis_lim,
                )
            plt.xticks(x_axis_ticks)
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
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = sequence_summary_stats_per_group_per_dataset
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = sequence_summary_stats_per_group_per_dataset
                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT_RATIO
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT_RATIO

                case UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_summary_stats_per_group_per_dataset
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_RAW
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_RAW

                case UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_summary_stats_per_group_per_dataset
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SUMMARY_SEQUENCE_STATISTICS_SHAREX_PCT
                    share_y = SUMMARY_SEQUENCE_STATISTICS_SHAREY_PCT

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{field}')

            n_cols = set_facet_grid_column_number(data[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

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
                                           sharey=share_y)
                        )

            if share_y:
                y_axis_lim = return_axis_limits(data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio)
                g.set(ylim=y_axis_lim)

            for ax, (facet_val, facet_data) in zip(g.axes.flat, data.groupby(DATASET_NAME_FIELD_NAME_STR)):

                if not share_y:
                    y_axis_lim = return_axis_limits(facet_data[field.value],
                                                    statistic_is_pct,
                                                    statistic_is_ratio)
                    ax.set_ylim(*y_axis_lim)
                
                color_palette = self._return_color_palette(facet_data,
                                                           RESULT_AGGREGATION_COLOR_PALETTE,
                                                           RESULT_AGGREGATION_COLOR_SATURATION)
                
                for line, color in zip(ax.lines, color_palette):
                    line.set_color(color)
                    line.set_alpha(SUMMARY_SEQUENCE_STATISTICS_LINE_ALPHA)

                ax.spines['top'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_TOP)
                ax.spines['bottom'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_LEFT)
                ax.spines['right'].set_visible(SUMMARY_SEQUENCE_STATISTICS_SHOW_RIGHT)

            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            plt.tight_layout()
            title = SUMMARY_SEQUENCE_STATISTICS_PLOT_NAME + field.value
            self._save_figure(title)
            plt.show(g);

    @sequence_statistics_distribution_per_group_per_dataset_decorator
    def plot_sequence_statistics_distribution_per_group_per_dataset(self,
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
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_RAW

                case SequenceStatisticsPlotFields.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ |\
                        SequenceStatisticsPlotFields.PCT_REPEATED_LEARNING_ACTIVITIES:

                    data = sequence_statistics_per_group_per_dataset
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT

                case SequenceStatisticsPlotFields.MEAN_NORMALIZED_SEQUENCE_DISTANCE |\
                     SequenceStatisticsPlotFields.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE:

                    data = sequence_statistics_per_group_per_dataset
                    statistic_is_pct = True 
                    statistic_is_ratio = True
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT_RATIO
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT_RATIO
                
                case UniqueSequenceFrequencyStatisticsPlotFields.SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_statistics_per_group_per_dataset
                    statistic_is_pct = False 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_RAW
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_RAW

                case UniqueSequenceFrequencyStatisticsPlotFields.RELATIVE_SEQUENCE_FREQUENCY:

                    data = unique_sequence_frequency_statistics_per_group_per_dataset
                    statistic_is_pct = True 
                    statistic_is_ratio = False
                    share_x = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREX_PCT
                    share_y = SEQUENCE_STATISTICS_DISTRIBUTION_SHAREY_PCT

                case _:
                    raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{field}')

            n_cols = set_facet_grid_column_number(data[DATASET_NAME_FIELD_NAME_STR],
                                                  RESULT_AGGREGATION_FACET_GRID_N_COLUMNS)

            def plot_boxplot(data, **kwargs):

                if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                    data = self._sort_groups_by_metric(data,
                                                       field.value)

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
                            saturation=RESULT_AGGREGATION_COLOR_SATURATION,
                            **kwargs)

            g = sns.FacetGrid(data,
                              col=DATASET_NAME_FIELD_NAME_STR,
                              col_wrap=n_cols,
                              height=RESULT_AGGREGATION_FACET_GRID_HEIGHT,
                              aspect=RESULT_AGGREGATION_FACET_GRID_ASPECT,
                              sharex=share_x,
                              sharey=share_y,
            )
            g.map_dataframe(plot_boxplot)

            if share_x:
                x_axis_lim = return_axis_limits(data[field.value],
                                                statistic_is_pct,
                                                statistic_is_ratio)
                g.set(xlim=x_axis_lim)

            if include_unique_sequences:
                g.add_legend(
                            title=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                            frameon=True,
                            bbox_to_anchor=(0.98, 0.5), 
                            loc='center left')

            for ax, (facet_val, facet_data) in zip(g.axes.flat, data.groupby(DATASET_NAME_FIELD_NAME_STR)):

                if not share_x:
                    x_axis_lim = return_axis_limits(facet_data[field.value],
                                                    statistic_is_pct,
                                                    statistic_is_ratio)
                    ax.set_xlim(*x_axis_lim)

                if include_unique_sequences:
                    pass
                else:
                    color_palette = self._return_color_palette(facet_data,
                                                               RESULT_AGGREGATION_COLOR_PALETTE,
                                                               RESULT_AGGREGATION_COLOR_SATURATION)

                    if SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_BOXES:
                        groups = np.unique(facet_data[GROUP_FIELD_NAME_STR])
                        color_dict = dict(zip(groups, color_palette))

                        labels = [int(tick.get_text()) for tick in ax.get_yticklabels()]

                        for box, group in zip(ax.patches, labels):
                            box.set_facecolor(color_dict[group])
                    else:
                        for box, color in zip(ax.patches, color_palette):
                            box.set_facecolor(color)

                ax.spines['top'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_TOP)
                ax.spines['bottom'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_BOTTOM)
                ax.spines['left'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_LEFT)
                ax.spines['right'].set_visible(SEQUENCE_STATISTICS_DISTRIBUTION_SHOW_RIGHT)

            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            plt.tight_layout()
            if include_unique_sequences:
                title = SEQUENCE_STATISTICS_DISTRIBUTION_NON_UNIQUE_UNIQUE_SPLIT_BOXPLOT_PLOT_NAME
            else:
                title = SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_PLOT_NAME
            title += field.value
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

        g = sns.relplot(sequence_count_per_group_per_dataset,
                        x=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
                        y=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
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
                        alpha=SEQUENCE_COUNT_MARKER_ALPHA,
                        zorder=101,
                        facet_kws=dict(sharex=SEQUENCE_COUNT_SHAREX,
                                       sharey=SEQUENCE_COUNT_SHAREX)
                    )

        if SEQUENCE_COUNT_SHAREX:
            x_axis_lim = return_axis_limits(sequence_count_per_group_per_dataset[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR],
                                            False,
                                            False)
            g.set(xlim=x_axis_lim,
                  ylim=x_axis_lim)


        for ax, (facet_val, facet_data) in zip(g.axes.flat, sequence_count_per_group_per_dataset.groupby(DATASET_NAME_FIELD_NAME_STR)):

            if not SEQUENCE_COUNT_SHAREX:
                x_axis_lim = return_axis_limits(facet_data[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR],
                                                False,
                                                False)
                ax.set_xlim(*x_axis_lim)
                ax.set_ylim(*x_axis_lim)

            ax.axline(xy1=(0,0), 
                      slope=1, 
                      color=SEQUENCE_COUNT_45_DEGREE_LINE_COLOR, 
                      linewidth=SEQUENCE_COUNT_45_DEGREE_LINE_WIDTH, 
                      zorder=100)

            color_palette = self._return_color_palette(facet_data,
                                                       RESULT_AGGREGATION_COLOR_PALETTE,
                                                       RESULT_AGGREGATION_COLOR_SATURATION)

            collection = ax.collections[0]
            collection.set_facecolor(color_palette)

            ax.spines['top'].set_visible(SEQUENCE_COUNT_SHOW_TOP)
            ax.spines['bottom'].set_visible(SEQUENCE_COUNT_SHOW_BOTTOM)
            ax.spines['left'].set_visible(SEQUENCE_COUNT_SHOW_LEFT)
            ax.spines['right'].set_visible(SEQUENCE_COUNT_SHOW_RIGHT)

        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.tight_layout()
        self._save_figure(SEQUENCE_COUNT_PLOT_NAME)
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

            p_val_is_significant_field_name, moa_strength_guideline = self._return_result_aggregation_omnibus_test_fields(eval_metric_is_categorical)

            fields = [DATASET_NAME_FIELD_NAME_STR,
                      GROUP_FIELD_NAME_STR,
                      OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR,
                      OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR,
                      p_val_is_significant_field_name,
                      moa_strength_guideline]

            omnibus_test_result_per_group = omnibus_test_result_per_group[fields]
            aggregation_dict = self._return_result_aggregation_omnibus_test_aggregation_dict(p_val_is_significant_field_name)

            agg_results = omnibus_test_result_per_group.groupby(DATASET_NAME_FIELD_NAME_STR)\
                                                       .agg(**aggregation_dict)\
                                                       .reset_index()

            moa_strength_counter = self._return_moa_strength_counter()
            for _, value in omnibus_test_result_per_group.query(f'{p_val_is_significant_field_name}==True')[moa_strength_guideline].iteritems():
                moa_strength_counter[value] += 1

            agg_moa_strength = pd.DataFrame(moa_strength_counter, 
                                            index=pd.Index([0]))
            agg_moa_strength.columns = self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names

            n_significant_groups = omnibus_test_result_per_group[p_val_is_significant_field_name].sum()
            agg_moa_strength_pct = agg_moa_strength.copy() / n_significant_groups * 100
            agg_moa_strength_pct.columns = self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names

            agg_moa_strength = pd.concat([agg_results, agg_moa_strength, agg_moa_strength_pct], 
                                         axis=1, 
                                         ignore_index=False)

            aggregated_omnibus_test_result_per_dataset_df_list.append(agg_moa_strength)


        aggregated_omnibus_test_result_per_dataset_df = pd.concat(aggregated_omnibus_test_result_per_dataset_df_list, 
                                                                  ignore_index=True)
        aggregated_omnibus_test_result_per_dataset_df.sort_values(by=DATASET_NAME_FIELD_NAME_STR, 
                                                                  inplace=True, 
                                                                  ignore_index=True)

        return aggregated_omnibus_test_result_per_dataset_df

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
        agg_moa_strength_count = aggregated_omnibus_test_result_per_dataset[self.result_aggregation_omnibus_test_result_moa_strength_counts_field_names]
        agg_moa_strength_pct = aggregated_omnibus_test_result_per_dataset[self.result_aggregation_omnibus_test_result_moa_strength_pct_field_names]

        agg_moa_strength_count_pct_combined = self._combine_count_and_pct(agg_moa_strength_count,
                                                                          agg_moa_strength_pct,
                                                                          self.result_aggregation_omnibus_test_result_moa_strength_counts_pct_combined_field_names)

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
                               sequence_statistic: str) -> pd.DataFrame:
        """
        Sorts the data by sequence statistic. Used for sorting boxplots. 
        """

        data = data.copy()

        match SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC:
            case BoxplotSortMetric.MEAN:
                sort_metric = np.mean

            case BoxplotSortMetric.MEDIAN:
                sort_metric = np.median

            case BoxplotSortMetric.MAX:
                sort_metric = np.max

            case BoxplotSortMetric.MIN:
                sort_metric = np.min

            case _:
                raise ValueError(RESULT_AGGREGATION_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC}')

        sort_metric_values = data.groupby([DATASET_NAME_FIELD_NAME_STR, GROUP_FIELD_NAME_STR])[sequence_statistic]\
                                 .agg(sort_metric)\
                                 .reset_index()

        sort_metric_values.rename({sequence_statistic: SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC.name}, 
                                  axis=1, 
                                  inplace=True)

        sort_metric_values.sort_values(by=[DATASET_NAME_FIELD_NAME_STR, SEQUENCE_STATISTICS_DISTRIBUTION_BOXPLOT_SORT_METRIC.name], inplace=True, ascending=False)

        data[GROUP_FIELD_NAME_STR] = pd.Categorical(data[GROUP_FIELD_NAME_STR], categories=sort_metric_values[GROUP_FIELD_NAME_STR], ordered=True)
        data.sort_values(by=GROUP_FIELD_NAME_STR, inplace=True)

        return data

    def _return_moa_strength_counter(self) -> defaultdict[str, int]:

        moa_strength_counter = defaultdict(int)
        for moa_str in RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_VALUES:
            moa_strength_counter[moa_str]

        return moa_strength_counter
    
    def _return_result_aggregation_omnibus_test_fields(self,
                                                       eval_metric_is_categorical: bool) -> tuple[str, str]:

        if RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_CORRECT_P_VALUES:
            p_val_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND.value + OMNIBUS_TESTS_PVAL_CORRECTED_FIELD_NAME_STR + RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_CORRECTION_METHOD.value
        else:
            p_val_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_P_VALUE_KIND.value

        p_val_is_significant_field_name = p_val_field_name + OMNIBUS_TESTS_PVAL_IS_SIGNIFICANT_FIELD_NAME_STR

        moa_calculation_base_suffix = self._return_measure_association_strength_calculation_base_suffix_str(RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_CALCULATION_BASE)

        if eval_metric_is_categorical:

            # moa_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY.value + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR
            # moa_conf_int_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY.value + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_VALUE_FIELD_NAME_STR
            moa_strength_guideline = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_CONTINGENCY.value + 
                                      '_' + 
                                      RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_CONTINGENCY.value + 
                                      OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_FIELD_NAME_STR + 
                                      moa_calculation_base_suffix)

        else:
            # moa_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV.value + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR
            # moa_conf_int_field_name = RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV.value + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_VALUE_FIELD_NAME_STR
            moa_strength_guideline = (RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_AOV.value + 
                                      '_' + 
                                      RESULT_AGGREGATION_OMNIBUS_TEST_RESULT_MOA_STRENGTH_GUIDELINE_AOV.value + 
                                      OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_FIELD_NAME_STR + 
                                      moa_calculation_base_suffix)

        return p_val_is_significant_field_name, moa_strength_guideline
    
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
    
    def _return_color_palette(self,
                              data: pd.DataFrame,
                              palette: str,
                              saturation: float) -> Iterable[Tuple[float]]:

        n_groups = data[GROUP_FIELD_NAME_STR].nunique()

        color_palette = sns.color_palette(palette,
                                          desat=saturation, 
                                          n_colors=n_groups)
        return color_palette