from .standard_import import *
from .constants import *
from .config import *
from .miscellaneous import *

class SequenceStatistics():
    """A class used to calculate and plot sequence statistics

    Attributes
    ----------
    unique_learning_activity_sequence_stats_per_group : pd.DataFrame
        A dataframe containing statistics for unique sequences per group 
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A dataframe containing statistics for all sequences per group 
    sequence_count_per_group : pd.DataFrame
        A dataframe containing sequence counts and unique sequence counts per group

    Methods
    -------
    return_unique_learning_activity_sequence_stats_per_group(interactions, dataset_name, group_field)
        Return a dataframe which contains statistics (frequencies and lengths) of unique learning_activity sequences over user entities grouped by group entities
    return_learning_activity_sequence_stats_per_group(unique_learning_activity_seq_stats_per_group)
        Return a dataframe which contains statistics (frequencies and lengths) of learning_activity sequences over user entities grouped by group entities
    plot_sequence_stats()
        Plot unique sequence statistics
    plot_sequence_stats_per_group()
        Plot unique sequence statistics per grouping variable
    print_seq_count_per_group()
        Prints sequence count and unique sequence count per group
    """

    def __init__(self, 
                 interactions: pd.DataFrame,
                 dataset_name: str,
                 group_field: str):
        
        self.interactions = interactions
        self.dataset_name = dataset_name
        self.group_field = group_field
        self._n_groups = self.interactions[GROUP_FIELD_NAME_STR].nunique()

        self.unique_learning_activity_sequence_stats_per_group = self.return_unique_learning_activity_sequence_stats_per_group(self.interactions,
                                                                                                                               self.dataset_name,
                                                                                                                               self.group_field)

        self.learning_activity_sequence_stats_per_group = self.return_learning_activity_sequence_stats_per_group(self.unique_learning_activity_sequence_stats_per_group)

        self.sequence_count_per_group = self._return_seq_count_per_group()

    @classmethod
    def return_unique_learning_activity_sequence_stats_per_group(cls,
                                                                 interactions: pd.DataFrame,
                                                                 dataset_name: str,
                                                                 group_field: str) -> pd.DataFrame:
        """Return a dataframe which contains statistics (frequencies and lengths) of unique learning_activity sequences over user entities grouped by group entities

        Parameters
        ----------
        interactions : pd.DataFrame
            The interactions dataframe
        dataset_name : str
            The name of the dataset.
        group_field : str
            The group field column
            This argument should be set to None if the interactions dataframe does not have a group_field

        Returns
        -------
        pd.DataFrame
            A dataframe containing statistics (frequencies and lengths) of unique learning_activity sequences over user entities grouped by group entities
        """
        interactions = interactions.copy()

        # helper functions
        def calc_n_repeated(seq_tuple):
            length = len(seq_tuple) 
            n_unique_elements = len(set(seq_tuple))
            number_repeated_elements = length - n_unique_elements

            return number_repeated_elements

        def calc_pct_repeated(seq_tuple):
            length = len(seq_tuple) 
            n_unique_elements = len(set(seq_tuple))
            number_repeated_elements = length - n_unique_elements
            percentage_repeated_elements = number_repeated_elements / length * 100

            return percentage_repeated_elements

        if not group_field:
            interactions[GROUP_FIELD_NAME_STR] = '0'
        group_field = GROUP_FIELD_NAME_STR

        
        group_data = []
        seq_count_per_group_data = []
        unique_seq_count_per_group_data = []
        unique_seq_data = []
        unique_seq_id_data = []
        seq_freq_data = []
        seq_freq_pct_data = []
        seq_len_data = []
        n_repeated_learning_activity_in_seq_data = []
        pct_repeated_learning_activity_in_seq_data = []
        n_unique_learning_activities_per_sequence_data = []
        pct_unique_learning_activities_per_group_in_sequence_data = []

        for n, (group, df_1) in enumerate(interactions.groupby(group_field)):
            learning_activity_seq_frequency_over_user_dict = defaultdict(int) 
            learning_activity_seq_id_remapping_dict = defaultdict(int)
            df_1 = df_1.sort_values(by=[USER_FIELD_NAME_STR, TIMESTAMP_FIELD_NAME_STR])
            for user, df_2 in df_1.groupby(USER_FIELD_NAME_STR):
                seq_col_3 = df_2[LEARNING_ACTIVITY_FIELD_NAME_STR].to_list()
                seq_col_3 = tuple(seq_col_3)
                sequence_id = df_2[SEQUENCE_ID_FIELD_NAME_STR].iloc[0]

                learning_activity_seq_frequency_over_user_dict[seq_col_3] += 1
                learning_activity_seq_id_remapping_dict[seq_col_3] = sequence_id
            
            seq_count = sum(list(learning_activity_seq_frequency_over_user_dict.values()))
            unique_seq_count = len(list(learning_activity_seq_frequency_over_user_dict.keys()))

            unique_seq_list = list(learning_activity_seq_frequency_over_user_dict.keys())
            unique_seq_id_list = [learning_activity_seq_id_remapping_dict[i] for i in unique_seq_list]
            seq_freq_list = list(learning_activity_seq_frequency_over_user_dict.values())
            seq_freq_pct_list = [freq/seq_count*100 for freq in learning_activity_seq_frequency_over_user_dict.values()]
            seq_len_list = list(map(len, learning_activity_seq_frequency_over_user_dict.keys()))
            n_repeated_learning_activity_in_seq_list = list(map(calc_n_repeated, unique_seq_list)) 
            pct_repeated_learning_activity_in_seq_list = list(map(calc_pct_repeated, unique_seq_list)) 
            n_unique_learning_activities_per_sequence_list = [len(set(tup)) for tup in unique_seq_list]
            n_unique_learning_activities_in_group = len({value for tup in unique_seq_list for value in tup})
            pct_unique_learning_activities_per_group_in_sequence_list = [i/n_unique_learning_activities_in_group*100 for i in n_unique_learning_activities_per_sequence_list]

            group_data.extend([int(group)] * len(unique_seq_list))
            seq_count_per_group_data.extend([seq_count] * len(unique_seq_list))
            unique_seq_count_per_group_data.extend([unique_seq_count] * len(unique_seq_list))
            unique_seq_data.extend(unique_seq_list)
            unique_seq_id_data.extend(unique_seq_id_list)
            seq_freq_data.extend(seq_freq_list)
            seq_freq_pct_data.extend(seq_freq_pct_list)
            seq_len_data.extend(seq_len_list)
            n_repeated_learning_activity_in_seq_data.extend(n_repeated_learning_activity_in_seq_list)
            pct_repeated_learning_activity_in_seq_data.extend(pct_repeated_learning_activity_in_seq_list)
            n_unique_learning_activities_per_sequence_data.extend(n_unique_learning_activities_per_sequence_list)
            pct_unique_learning_activities_per_group_in_sequence_data.extend(pct_unique_learning_activities_per_group_in_sequence_list)



        seq_stats_dict = {DATASET_NAME_FIELD_NAME_STR: dataset_name,
                          GROUP_FIELD_NAME_STR: group_data,
                          LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR: seq_count_per_group_data,
                          LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR: unique_seq_count_per_group_data,
                          LEARNING_ACTIVITY_SEQUENCE_NAME_STR: unique_seq_data,
                          LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: unique_seq_id_data,
                          LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR: seq_freq_data,
                          LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR: seq_freq_pct_data,
                          LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: seq_len_data,
                          LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_NAME_STR: n_repeated_learning_activity_in_seq_data,
                          LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR: pct_repeated_learning_activity_in_seq_data,
                          LEARNING_ACTIVITY_SEQUENCE_NUMBER_UNIQUE_LEARNING_ACTIVITIES_NAME_STR: n_unique_learning_activities_per_sequence_data,
                          LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR: pct_unique_learning_activities_per_group_in_sequence_data} 

        unique_learning_activity_seq_stats_per_group = pd.DataFrame(seq_stats_dict)
        unique_learning_activity_seq_stats_per_group = (unique_learning_activity_seq_stats_per_group
                                                        .sort_values(by=[GROUP_FIELD_NAME_STR, 
                                                                        LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR], 
                                                                    ascending=[True, False])
                                                        .reset_index(drop=True))

        unique_learning_activity_seq_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR] = LEARNING_ACTIVITY_SEQUENCE_TYPE_UNIQUE_SEQ_VALUE_STR

        return unique_learning_activity_seq_stats_per_group

    @classmethod
    def return_learning_activity_sequence_stats_per_group(cls,
                                                          unique_learning_activity_seq_stats_per_group: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe which contains statistics (frequencies and lengths) of learning_activity sequences over user entities grouped by group entities

        Parameters
        ----------
        unique_learning_activity_seq_stats_per_group : pd.DataFrame
            The unique_learning_activity_seq_stats_per_group dataframe created by function return_unique_learning_activity_sequence_stats_per_group

        Returns
        -------
        pd.DataFrame
            A dataframe containing statistics (frequencies and lengths) of learning_activity sequences over user entities grouped by group entities
        """

        column_names = unique_learning_activity_seq_stats_per_group.columns
        concat_list = []
        for _, row in unique_learning_activity_seq_stats_per_group.iterrows():
            freq_within_topic = row[LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR]
            concat_list.append(np.repeat(np.array([row.values]), freq_within_topic, axis=0))
        
        if concat_list:
            learning_activity_seq_stats_per_group = pd.DataFrame(np.concatenate(concat_list, axis=0), columns=column_names)
        else:
            learning_activity_seq_stats_per_group = pd.DataFrame([], columns=column_names)

        learning_activity_seq_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR] = LEARNING_ACTIVITY_SEQUENCE_TYPE_ALL_SEQ_VALUE_STR

        return learning_activity_seq_stats_per_group

    def _return_aggregated_statistic_per_group(self,
                                               sequence_stats_per_group: pd.DataFrame,
                                               statistic: str,
                                               sequence_type: str) -> pd.DataFrame:
        """Aggregates a field, specified by the statistic parameter, in the input sequence stats dataframe and returns the\
        results as a dataframe in the long format

        Parameters
        ----------
        sequence_stats_per_group : pd.DataFrame
            A dataframe containing statistics of learning_activity sequences over user entities grouped by group entities

        Returns
        -------
        pd.DataFrame
            A dataframe in long format containing an aggregated statistic of learning_activity sequences
        """

        # helper functions for quartiles 
        def first_quartile(array):
            return np.quantile(array, 0.25)
        def third_quartile(array):
            return np.quantile(array, 0.75)

        aggregated_stats_per_group = (sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR)[statistic]
                                                              .agg([min, max, np.median, first_quartile, third_quartile])
                                                              .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                              'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                              'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                                              'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                              'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})
                                                              .reset_index())
        
        aggregated_stats_per_group_long = pd.melt(aggregated_stats_per_group[[GROUP_FIELD_NAME_STR, 
                                                                            LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                            LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                            LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                            LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                                                            LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                  id_vars=GROUP_FIELD_NAME_STR,
                                                  var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                  value_name=statistic)

        aggregated_stats_per_group_long[LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR] = sequence_type

        return aggregated_stats_per_group_long

    def _return_seq_count_per_group(self) -> pd.DataFrame:
    
        seq_count_df = (self.unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR]
                                                                              .agg([sum, len])
                                                                              .reset_index()
                                                                              .rename({'sum': LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR,
                                                                                      'len': LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_NAME_STR}))

        seq_count_df.insert(0, 
                            DATASET_NAME_FIELD_NAME_STR,
                            self.dataset_name)

        return seq_count_df

    def _plot_stat_plot(self,
                        sequence_stats_per_group_df: pd.DataFrame,
                        aggregated_statistics_df_long: pd.DataFrame,
                        statistic: str,
                        stat_is_pct: bool,
                        boxplot_title: str,
                        pointplot_title: str,
                        ecdfplot_title: str) -> None:
        """Plot a pointplot and ecdfplot of a sequence statistic over sequences per group. The plots are split by the
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
        boxplot_title : str
            The title for the respective boxplot
        pointplot_title : str
            The title for the respective pointplot
        ecdfplot_title : str
            The title for the respective ecdfplot
        """
        if stat_is_pct:
            axis_lim = (-5, 105)
        else:
            axis_lim = return_axis_limits(sequence_stats_per_group_df[statistic])

        # boxplot
        g = sns.FacetGrid(data=sequence_stats_per_group_df,
                          col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                          col_wrap=2, 
                          sharex=False, 
                          sharey=True,
                          height=9,
                          xlim=axis_lim)
        g.map_dataframe(sns.boxplot,
                        x=statistic,
                        y=GROUP_FIELD_NAME_STR,
                        orient='h',
                        showmeans=True, 
                        meanprops=marker_config,
                        palette=sns.color_palette("husl", n_colors=self._n_groups))
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(boxplot_title, 
                          fontsize=20)
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
        g.add_legend()
        plt.show(g);

        # pointplot
        g = sns.FacetGrid(data=aggregated_statistics_df_long,
                          col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                          col_wrap=2, 
                          sharex=False, 
                          sharey=True,
                          height=9,
                          ylim=axis_lim)
        g.map_dataframe(sns.pointplot,
                        x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                        y=statistic, 
                        hue=GROUP_FIELD_NAME_STR,
                        palette=sns.color_palette("husl", n_colors=self._n_groups))
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(pointplot_title, 
                          fontsize=20)
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
        g.add_legend()
        plt.show(g);

        # ecdf plot
        g = sns.FacetGrid(data=sequence_stats_per_group_df,
                          col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                          col_wrap=2, 
                          sharex=False, 
                          sharey=True,
                          height=9,
                          xlim=axis_lim)
        g.map_dataframe(sns.ecdfplot,
                        x=statistic,
                        hue=GROUP_FIELD_NAME_STR,
                        palette=sns.color_palette("husl", n_colors=self._n_groups))
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(ecdfplot_title, 
                          fontsize=20)
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
        g.add_legend()
        plt.show(g);

    def _plot_stat_vs_freq_plot_per_group(self,
                                          statistic: str,
                                          statistic_label: str,
                                          stat_is_pct: bool,
                                          title_list: list[str],
                                          share_x: bool,
                                          share_y: bool) -> None:
        """Plot a scatterplot of a sequence statistic against the respective sequence frequency of a unique sequence
        per group.

        Parameters
        ----------
        statistic : str
            The sequence statistic to be plotted
        statistic_label : str
            The label used for the respective statistic
        stat_is_pct : bool
            A flag indicating whether the sequence statistic is a percentage
        title_list : list[str]
            A list containing the titles of sequence frequency % and sequence frequency plots
        share_x : bool
            A flag indicating whether the x axis should be shared across subplot
        share_y : bool
            A flag indicating whether the y axis should be shared across subplot
        """
        xlim_pct_plot = (-5, 105)  
        xlim_plot = return_axis_limits(self.unique_learning_activity_sequence_stats_per_group[LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR])

        ylim_pct = (-5, 105)
        ylim = return_axis_limits(self.unique_learning_activity_sequence_stats_per_group[statistic])

        if stat_is_pct:
            ylim_plot = ylim_pct
        else:
            ylim_plot = ylim

        # sequence frequency %
        g = sns.FacetGrid(self.unique_learning_activity_sequence_stats_per_group, 
                          col=GROUP_FIELD_NAME_STR, 
                          col_wrap=6, 
                          sharex=share_x, 
                          sharey=share_y)
        g.map_dataframe(sns.scatterplot, 
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
                        y=statistic, 
                        s=100, 
                        alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
              ylabel=statistic_label, 
              xlim=xlim_pct_plot, 
              ylim=ylim_plot)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(title_list[0], 
                    fontsize=20)
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
        plt.show(g)

        # absolute sequence frequency
        g = sns.FacetGrid(self.unique_learning_activity_sequence_stats_per_group, 
                          col=GROUP_FIELD_NAME_STR, 
                          col_wrap=6, 
                          sharex=True, 
                          sharey=True)
        g.map_dataframe(sns.scatterplot, 
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
                        y=statistic, 
                        s=100, 
                        alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
              ylabel=statistic_label, 
              xlim=xlim_plot, 
              ylim=ylim_plot)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(title_list[1], 
                          fontsize=20)
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
        plt.show(g)
    
    def _plot_stat_hist_plot_per_group(self,
                                       sequence_stats_per_group_df: pd.DataFrame,
                                       statistic: str,
                                       stat_is_pct: bool,
                                       statistic_label: str,
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
        statistic_label : str
            The label used for the respective statistic
        title_list : list[str]
            A list containing the titles of sequence frequency % and sequence frequency plots
        share_x : bool
            A flag indicating whether the x axis should be shared across subplot
        share_y : bool
            A flag indicating whether the y axis should be shared across subplot
        """
        if stat_is_pct:
            xlim_plot = (-5, 105)
        else:
            xlim_plot = return_axis_limits(sequence_stats_per_group_df[statistic])

        title = LEARNING_ACTIVITY_SEQUENCE_HISTOGRAM_TITLE_NAME_STR + statistic
        g = sns.FacetGrid(sequence_stats_per_group_df,
                          col=LEARNING_ACTIVITY_SEQUENCE_TYPE_NAME_STR,
                          row=GROUP_FIELD_NAME_STR, 
                          sharex=share_x, 
                          sharey=share_y,
                          height=6)
        g.map_dataframe(sns.histplot, 
                        x=statistic, 
                        stat='count')
        g.set(xlabel=statistic_label,
              xlim=xlim_plot)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(title, 
                          fontsize=20)
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
        plt.show(g)

    def plot_sequence_stats(self) -> None:
        """Plot unique sequence statistics
        """
        xlim_pct_plot = (-1, 105)  
        ylim_pct_plot = (-1, None)
        xlim_plot = (None, None)  
        ylim_plot = (-1, None)
        #
        g = sns.scatterplot(data=self.unique_learning_activity_sequence_stats_per_group,
                            x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
                            y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                            s=100, 
                            alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
              xlim=xlim_pct_plot, 
              ylim=ylim_pct_plot)
        g.set_title(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                    fontsize=20)
        plt.show(g)

        #
        g = sns.scatterplot(data=self.unique_learning_activity_sequence_stats_per_group,
                            x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
                            y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                            s=100, 
                            alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
              xlim=xlim_plot, 
              ylim=ylim_plot)
        g.set_title(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_TITLE_NAME_STR, 
                    fontsize=20)
        plt.show(g)

        #
        g = sns.scatterplot(data=self.unique_learning_activity_sequence_stats_per_group,
                            x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
                            y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                            s=100, 
                            alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
              xlim=xlim_pct_plot, 
              ylim=ylim_pct_plot)
        g.set_title(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                    fontsize=20)
        plt.show(g)

        #
        g = sns.scatterplot(data=self.unique_learning_activity_sequence_stats_per_group,
                            x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
                            y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                            s=100, 
                            alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
              xlim=xlim_plot, 
              ylim=ylim_plot)
        g.set_title(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_TITLE_NAME_STR, 
                    fontsize=20)
        plt.show(g)

        #
        g = sns.scatterplot(data=self.unique_learning_activity_sequence_stats_per_group,
                            x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
                            y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                            s=100, 
                            alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
              xlim=xlim_pct_plot, 
              ylim=ylim_pct_plot)
        g.set_title(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                    fontsize=20)
        plt.show(g)

        #
        g = sns.scatterplot(data=self.unique_learning_activity_sequence_stats_per_group,
                            x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
                            y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                            s=100, 
                            alpha=0.4)
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
              xlim=xlim_plot, 
              ylim=ylim_plot)
        g.set_title(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_TITLE_NAME_STR, 
                    fontsize=20)
        plt.show(g)

    def plot_sequence_stats_per_group(self) -> None:
        """Plot unique sequence statistics per grouping variable
        """

        # create aggregated dataframes for plotting    
        learning_activity_sequence_stats_merged = pd.concat([self.learning_activity_sequence_stats_per_group,
                                                             self.unique_learning_activity_sequence_stats_per_group],
                                                             axis=0)

        # seq frequency
        unique_sequence_frequency_stats_per_group_long = self._return_aggregated_statistic_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                                                     LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR,
                                                                                                     LEARNING_ACTIVITY_SEQUENCE_TYPE_UNIQUE_SEQ_VALUE_STR)

        # seq length
        sequence_length_stats_per_group_long = self._return_aggregated_statistic_per_group(self.learning_activity_sequence_stats_per_group,
                                                                                           LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                                                                           LEARNING_ACTIVITY_SEQUENCE_TYPE_ALL_SEQ_VALUE_STR)
        unique_sequence_length_stats_per_group_long = self._return_aggregated_statistic_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                                                  LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                                                                                  LEARNING_ACTIVITY_SEQUENCE_TYPE_UNIQUE_SEQ_VALUE_STR)
        sequence_length_stats_per_group_long_merged = pd.concat([sequence_length_stats_per_group_long,
                                                                 unique_sequence_length_stats_per_group_long], 
                                                                 axis=0)


        # seq % repeated learning activities
        sequences_repeated_learning_activities_stats_per_group_long = self._return_aggregated_statistic_per_group(self.learning_activity_sequence_stats_per_group,
                                                                                                                  LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                                                                                                                  LEARNING_ACTIVITY_SEQUENCE_TYPE_ALL_SEQ_VALUE_STR)
        unique_sequences_repeated_learning_activities_stats_per_group_long = self._return_aggregated_statistic_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                                                                         LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                                                                                                                         LEARNING_ACTIVITY_SEQUENCE_TYPE_UNIQUE_SEQ_VALUE_STR)
        sequences_repeated_learning_activities_stats_per_group_long_merged = pd.concat([sequences_repeated_learning_activities_stats_per_group_long,
                                                                                        unique_sequences_repeated_learning_activities_stats_per_group_long], 
                                                                                        axis=0)

        # seq % unique learning activities
        sequence_pct_learning_activities_per_group_stats_per_group_long = self._return_aggregated_statistic_per_group(self.learning_activity_sequence_stats_per_group,
                                                                                                                      LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                                                                                                                      LEARNING_ACTIVITY_SEQUENCE_TYPE_ALL_SEQ_VALUE_STR)
        unique_sequence_pct_learning_activities_per_group_stats_per_group_long = self._return_aggregated_statistic_per_group(self.unique_learning_activity_sequence_stats_per_group,
                                                                                                                             LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                                                                                                                             LEARNING_ACTIVITY_SEQUENCE_TYPE_UNIQUE_SEQ_VALUE_STR)
        sequence_pct_learning_activities_per_group_stats_per_group_long_merged = pd.concat([sequence_pct_learning_activities_per_group_stats_per_group_long,
                                                                                            unique_sequence_pct_learning_activities_per_group_stats_per_group_long], 
                                                                                            axis=0)

        ####################################################################################################################

        # all groups in one figure - unique seq count vs seq count
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_UNIQUE_VS_TOTAL_NUMBER_OF_SEQUENCES_PER_GROUP_TITLE_NAME_STR}')
        print(' ')
        count_df = self.unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR).head(1)
        limits = return_axis_limits(count_df[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR])
        g = sns.scatterplot(data=count_df,
                            x=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
                            y=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
                            hue=GROUP_FIELD_NAME_STR, 
                            s=100,
                            alpha=0.9,
                            palette=sns.color_palette("husl", n_colors=self._n_groups))
        g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
              ylabel=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
              xlim=(limits[0], limits[1]),
              ylim=(limits[0], limits[1]))
        g.set_title(LEARNING_ACTIVITY_UNIQUE_VS_TOTAL_NUMBER_OF_SEQUENCES_PER_GROUP_TITLE_NAME_STR, 
                    fontsize=20)
        g.axline(xy1=(0,0), slope=1, color='r', linewidth=3);
        g.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show(g)

        # all groups in one figure - seq freq stats
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_TITLE_NAME_STR}')
        print(' ')
        print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
        print(' ')

        self._plot_stat_plot(self.unique_learning_activity_sequence_stats_per_group,
                             unique_sequence_frequency_stats_per_group_long,
                             LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_WITHIN_GROUP_PCT_NAME_STR,
                             True,
                             LEARNING_ACTIVITY_UNIQUE_SEQUENCE_BOXPLOT_FREQUENCY_PCT_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_UNIQUE_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_FREQUENCY_PCT_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_UNIQUE_SEQUENCE_ECDF_FREQUENCY_PCT_TITLE_NAME_STR)

        # all groups in one figure - seq len stats
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_LENGTH_TITLE_NAME_STR}')
        print(' ')
        print(LEARNING_ACTIVITY_STATS_SEQUENCE_NAME_STR)
        print('VS')
        print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
        print(' ')

        self._plot_stat_plot(learning_activity_sequence_stats_merged,
                             sequence_length_stats_per_group_long_merged,
                             LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                             False,
                             LEARNING_ACTIVITY_SEQUENCE_BOXPLOT_LENGTH_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_LENGTH_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_SEQUENCE_ECDF_LENGTH_TITLE_NAME_STR)

        # all groups in one figure - repeated learning activities stats
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_TITLE_NAME_STR}')
        print(' ')
        print(LEARNING_ACTIVITY_STATS_SEQUENCE_NAME_STR)
        print('VS')
        print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
        print(' ')

        self._plot_stat_plot(learning_activity_sequence_stats_merged,
                             sequences_repeated_learning_activities_stats_per_group_long_merged,
                             LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                             True,
                             LEARNING_ACTIVITY_SEQUENCE_BOXPLOT_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_SEQUENCE_ECDF_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR)


        # all groups in one figure - pct of unique learning activities per group in seq stats
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR} in {SEQUENCE_STR}')
        print(' ')
        print(LEARNING_ACTIVITY_STATS_SEQUENCE_NAME_STR)
        print('VS')
        print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
        print(' ')

        self._plot_stat_plot(learning_activity_sequence_stats_merged,
                             sequence_pct_learning_activities_per_group_stats_per_group_long_merged,
                             LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                             True,
                             LEARNING_ACTIVITY_SEQUENCE_BOXPLOT_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR,
                             LEARNING_ACTIVITY_SEQUENCE_ECDF_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR)


        ####################################################################################################################

        # unique sequences per group figures - statistic vs frequency
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_UNIQUE_SEQUENCES_STAT_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR}')
        print(' ')
        print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
        print(' ')

        self._plot_stat_vs_freq_plot_per_group(LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                               LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                               False,
                                               [LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                                LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                               True,
                                               True)

        self._plot_stat_vs_freq_plot_per_group(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                                               LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
                                               True,
                                               [LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                                   LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                               True,
                                               True)

        self._plot_stat_vs_freq_plot_per_group(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                                               LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
                                               True,
                                               [LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR,
                                                   LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR],
                                               True,
                                               True)

        ####################################################################################################################

        # all sequence vs unique sequences - statistic histogram
        print(' ')
        print(STAR_STRING)
        print(STAR_STRING)
        print(' ')
        print(f'{LEARNING_ACTIVITY_SEQUENCE_HISTOGRAM_TITLE_NAME_STR}')
        print(' ')
        print(LEARNING_ACTIVITY_STATS_SEQUENCE_NAME_STR)
        print('VS')
        print(LEARNING_ACTIVITY_STATS_UNIQUE_SEQUENCE_NAME_STR)
        print(' ')

        self._plot_stat_hist_plot_per_group(learning_activity_sequence_stats_merged,
                                            LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                            False,
                                            LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
                                            True,
                                            True)

        self._plot_stat_hist_plot_per_group(learning_activity_sequence_stats_merged,
                                            LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
                                            True,
                                            LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
                                            True,
                                            True)

        self._plot_stat_hist_plot_per_group(learning_activity_sequence_stats_merged,
                                            LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
                                            True,
                                            LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
                                            True,
                                            True)

    def print_seq_count_per_group(self) -> None:
        """Prints sequence count and unique sequence count per group
        """
        print('')
        print(f'== {LEARNING_ACTIVITY_SEQUENCE_FILTER_SEQUENCE_COUNT_DATAFRAME_NAME_STR} ==')
        print(DASH_STRING)
        print(self.sequence_count_per_group)
        print(DASH_STRING)
        print('')