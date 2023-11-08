from .standard_import import *
from .constants import *
from .config import *

def return_learning_activity_sequence_stats_over_user_per_group(interactions: pd.DataFrame,
                                                                dataset_name: str,
                                                                group_field: str):
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


        group_data.extend([str(group)] * len(unique_seq_list))
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
                      LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR: seq_count_per_group_data,
                      UNIQUE_LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR: unique_seq_count_per_group_data,
                      LEARNING_ACTIVITY_SEQUENCE_NAME_STR: unique_seq_data,
                      LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR: unique_seq_id_data,
                      LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR: seq_freq_data,
                      LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR: seq_freq_pct_data,
                      LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR: seq_len_data,
                      LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_NAME_STR: n_repeated_learning_activity_in_seq_data,
                      LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR: pct_repeated_learning_activity_in_seq_data,
                      LEARNING_ACTIVITY_SEQUENCE_NUMBER_UNIQUE_LEARNING_ACTIVITIES_NAME_STR: n_unique_learning_activities_per_sequence_data,
                      LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR: pct_unique_learning_activities_per_group_in_sequence_data} 

    learning_activity_seq_stats_over_user_per_group = pd.DataFrame(seq_stats_dict)
    learning_activity_seq_stats_over_user_per_group = (learning_activity_seq_stats_over_user_per_group
                                                       .sort_values(by=[GROUP_FIELD_NAME_STR, 
                                                                        LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR], 
                                                                    ascending=[True, False])
                                                        .reset_index(drop=True))

    return learning_activity_seq_stats_over_user_per_group

def plot_sequence_stats(learning_activity_sequence_stats_per_group: pd.DataFrame):
    """Plot unique sequence statistics

    Parameters
    ----------
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A learning activity sequence statistics per group dataframe created by return_col3_sequence_stats_over_col2_per_col1 
    """
    xlim_pct_plot = (-1, 105)  
    ylim_pct_plot = (-1, None)
    xlim_plot = (None, None)  
    ylim_plot = (-1, None)
    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=xlim_pct_plot, 
          ylim=ylim_pct_plot)
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=xlim_plot, 
          ylim=ylim_plot)
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
          xlim=xlim_pct_plot, 
          ylim=ylim_pct_plot)
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
          xlim=xlim_plot, 
          ylim=ylim_plot)
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
          xlim=xlim_pct_plot, 
          ylim=ylim_pct_plot)
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    #
    g = sns.scatterplot(data=learning_activity_sequence_stats_per_group,
                        x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                        s=100, 
                        alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
          xlim=xlim_plot, 
          ylim=ylim_plot)
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

def plot_sequence_stats_per_group(learning_activity_sequence_stats_per_group: pd.DataFrame):
    """Plot unique sequence statistics per grouping variable

    Parameters
    ----------
    learning_activity_sequence_stats_per_group : pd.DataFrame
        A learning activity sequence statistics per group dataframe created by return_col3_sequence_stats_over_col2_per_col1 
    """
    # helper functions for quartiles 
    def first_quartile(array):
        return np.quantile(array, 0.25)
    def third_quartile(array):
        return np.quantile(array, 0.75)

    sequence_frequency_stats_per_group = learning_activity_sequence_stats_per_group\
                                         .groupby(GROUP_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR]\
                                         .agg([min, max, np.median, first_quartile, third_quartile])\
                                         .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                          'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                          'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                          'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                          'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})\
                                         .reset_index()
    
    sequence_frequency_stats_per_group_long = pd.melt(sequence_frequency_stats_per_group[[GROUP_FIELD_NAME_STR, 
                                                                                          LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                          LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                                          LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                          LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                                                                          LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                      id_vars=GROUP_FIELD_NAME_STR,
                                                      var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                      value_name=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR)

    sequence_length_stats_per_group = learning_activity_sequence_stats_per_group\
                                         .groupby(GROUP_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]\
                                         .agg([min, max, np.median, first_quartile, third_quartile])\
                                         .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                          'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                          'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                          'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                          'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})\
                                         .reset_index()
    
    sequence_length_stats_per_group_long = pd.melt(sequence_length_stats_per_group[[GROUP_FIELD_NAME_STR, 
                                                                                    LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                    LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                                    LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                    LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                                                                    LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                      id_vars=GROUP_FIELD_NAME_STR,
                                                      var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                      value_name=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR)

    repeated_learning_activities_stats_per_group = learning_activity_sequence_stats_per_group\
                                                   .groupby(GROUP_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR]\
                                                   .agg([min, max, np.median, first_quartile, third_quartile])\
                                                   .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                    'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                    'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                                    'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                    'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})\
                                                   .reset_index()
    
    repeated_learning_activities_stats_per_group_long = pd.melt(repeated_learning_activities_stats_per_group[[GROUP_FIELD_NAME_STR, 
                                                                                                              LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                                              LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                                                              LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                                              LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                                                                                              LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                                id_vars=GROUP_FIELD_NAME_STR,
                                                                var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                                value_name=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR)

    pct_learning_activities_per_group_stats_per_group = learning_activity_sequence_stats_per_group\
                                                        .groupby(GROUP_FIELD_NAME_STR)[LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR]\
                                                        .agg([min, max, np.median, first_quartile, third_quartile])\
                                                        .rename(columns={'min': LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                         'median': LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                         'max': LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR,
                                                                         'first_quartile': LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                         'third_quartile': LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR})\
                                                        .reset_index()
    
    pct_learning_activities_per_group_stats_per_group_long = pd.melt(pct_learning_activities_per_group_stats_per_group[[GROUP_FIELD_NAME_STR, 
                                                                                                                        LEARNING_ACTIVITY_SEQUENCE_MIN_NAME_STR, 
                                                                                                                        LEARNING_ACTIVITY_SEQUENCE_FIRST_QUARTILE_NAME_STR,
                                                                                                                        LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR, 
                                                                                                                        LEARNING_ACTIVITY_SEQUENCE_THIRD_QUARTILE_NAME_STR,
                                                                                                                        LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR]], 
                                                                     id_vars=GROUP_FIELD_NAME_STR,
                                                                     var_name=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR,
                                                                     value_name=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR)

    # all groups in one figure - unique seq count vs seq count
    count_df = learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR).head(1)
    ylim = count_df[LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR].max()
    g = sns.scatterplot(data=count_df,
                        x=LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR, 
                        y=UNIQUE_LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR, 
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR, 
          ylabel=UNIQUE_LEARNING_ACTIVITY_SEQUENCE_COUNT_NAME_STR,
          ylim=(-5,ylim))
    g.set_title(LEARNING_ACTIVITY_UNIQUE_VS_TOTAL_NUMBER_OF_SEQUENCES_PER_GROUP_TITLE_NAME_STR, 
                fontsize=20)
    g.axline(xy1=(0,0), slope=1, color='r', linewidth=3);
    plt.show(g)

    # all groups in one figure - seq freq stats
    g = sns.scatterplot(data=sequence_frequency_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_FREQUENCY_PCT_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_FREQUENCY_PCT_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=sequence_frequency_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                      hue=GROUP_FIELD_NAME_STR)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_FREQUENCY_PCT_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # all groups in one figure - seq len stats
    g = sns.scatterplot(data=sequence_length_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_LENGTH_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_LENGTH_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=sequence_length_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                      hue=GROUP_FIELD_NAME_STR)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_LENGTH_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # all groups in one figure - repeated learning activities stats
    g = sns.scatterplot(data=repeated_learning_activities_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=repeated_learning_activities_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                      hue=GROUP_FIELD_NAME_STR)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_REPEATED_LEARNING_ACTIVITIES_PCT_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # all groups in one figure - pct of unique learning activities per group in seq stats
    g = sns.scatterplot(data=pct_learning_activities_per_group_stats_per_group, 
                        x=LEARNING_ACTIVITY_SEQUENCE_MAX_NAME_STR, 
                        y=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_NAME_STR,
                        s=100, 
                        alpha=0.7)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_MEDIAN_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR,
          xlim=(-5,105), 
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MEDIAN_VS_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR, 
                fontsize=20)
    plt.show(g)

    g = sns.pointplot(data=pct_learning_activities_per_group_stats_per_group_long, 
                      x=LEARNING_ACTIVITY_SEQUENCE_DESCRIPTIVE_STATISTIC_NAME_STR, 
                      y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                      hue=GROUP_FIELD_NAME_STR)
    g.set(xlabel=None,
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR,
          ylim=(-5,105))
    g.set_title(LEARNING_ACTIVITY_SEQUENCE_MIN_VS_MEDIAN_VS_MAX_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_TITLE_NAME_STR, 
                fontsize=20)
    g.legend_ = None
    plt.show(g)

    # per group figures
    xlim_pct_plot = (-10, 110)  
    ylim_pct_plot = (-1, None)
    xlim_plot = (-1, None)  
    ylim_plot = (-1, None)
    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=GROUP_FIELD_NAME_STR, 
                      col_wrap=6, 
                      sharex=False, 
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=xlim_pct_plot, 
          ylim=ylim_pct_plot)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=GROUP_FIELD_NAME_STR, 
                      col_wrap=6, 
                      sharex=False, 
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR, 
          xlim=xlim_plot, 
          ylim=ylim_plot)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_LENGTH_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=GROUP_FIELD_NAME_STR, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR, 
          xlim=xlim_pct_plot, 
          ylim=ylim_pct_plot)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=GROUP_FIELD_NAME_STR, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_LABEL_NAME_STR, 
          xlim=xlim_plot, 
          ylim=ylim_plot)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_REPEATED_LEARNING_ACTIVITIES_PCT_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=GROUP_FIELD_NAME_STR, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_PCT_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR, 
          xlim=xlim_pct_plot, 
          ylim=ylim_pct_plot)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PCT_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)

    g = sns.FacetGrid(learning_activity_sequence_stats_per_group, 
                      col=GROUP_FIELD_NAME_STR, 
                      col_wrap=6, 
                      sharex=False,
                      sharey=False)
    g.map_dataframe(sns.scatterplot, 
                    x=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
                    y=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_NAME_STR, 
                    s=100, 
                    alpha=0.4)
    g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_FREQUENCY_NAME_STR, 
          ylabel=LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_LABEL_NAME_STR, 
          xlim=xlim_plot, 
          ylim=ylim_plot)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(LEARNING_ACTIVITY_SEQUENCE_PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_VS_FREQUENCY_PER_GROUP_TITLE_NAME_STR, 
                   fontsize=20)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
    plt.show(g)