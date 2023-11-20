from .standard_import import *
from .constants import *
from .config import *
from .functions import *
from .sequence_distance import *
from .sequence_distance_no_group import *

def calculate_sequence_distances(interactions: pd.DataFrame,
                                 group_field: str) -> tuple:
    """For each group calculates the (learning activity-) sequence distances between each possible user\
    combination(seq_distances) and sequence combination(unique_seq_distances) pair.\
    If a interactions dataframe does not contain a grouping field, the sequence distance results will be treated as if
    they belong to a single group(group '0') ranging over the entire length of the interactions dataframe.


    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field

    Returns
    -------
    dict
        A dictionary consisting of:\

        A dictionary containing for every group(group '0' only if there is no grouping field in the interactions dataframe)\
        a ndarray of sequence distances between user combinations, a ndarray of lengths of the longer\
        of two compared sequences, a ndarray of users id combinations used for sequence distance calculation,\
        a ndarray of sequence id combinations, a ndarray of user ids per group, a ndarray of sequence lengths\
        for every user per group, a ndarray of sequence ids and a ndarray of tuples containing the sequence of\
        learning activities the sequence ids map to.\

        A dictionary containing for every group(group '0' only if there is no grouping field in the interactions dataframe)\
        a ndarray of sequence distances between sequence combinations, a ndarray of lengths of the longer\
        of two compared sequence and a ndarray of sequence id combinations.
    """
    if group_field:
        print('-'*20)
        print(f'{GROUP_FIELD_NAME_STR}-Field Available:')
        print(f'Calculate {SEQUENCE_STR} Distances for each {GROUP_FIELD_NAME_STR}')
        print('-'*20)
        seq_sim = SeqDist(interactions, 
                          USER_FIELD_NAME_STR, 
                          GROUP_FIELD_NAME_STR, 
                          LEARNING_ACTIVITY_FIELD_NAME_STR, 
                          SEQUENCE_ID_FIELD_NAME_STR)
        seq_distances = seq_sim.get_user_sequence_distances_per_group(distance)
    else:
        print('-'*20)
        print(f'{GROUP_FIELD_NAME_STR}-Field NOT Available:')
        print(f'Calculate {SEQUENCE_STR} Distances')
        print('-'*20)
        seq_sim = SeqDistNoGroup(interactions,
                                 USER_FIELD_NAME_STR,
                                 LEARNING_ACTIVITY_FIELD_NAME_STR,
                                 SEQUENCE_ID_FIELD_NAME_STR)
        seq_distances = seq_sim.get_user_sequence_distances(distance)

    return seq_distances

class SequenceDistanceAnalytics:
    """docstring for ClassName."""

    # class variables
    learning_activity_sequence_distance_based_on_user_combinations_name_str = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_USER_COMBINATIONS_NAME_STR
    learning_activity_sequence_distance_based_on_sequence_combinations_name_str = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_BASED_ON_SEQUENCE_COMBINATIONS_NAME_STR

    learning_activity_sequence_distance_name_str = LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_normalized_sequence_distance_name_str = LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_sequence_sequence_id_combination_name_str = LEARNING_ACTIVITY_SEQUENCE_SEQUENCE_ID_COMBINATION_NAME_STR
    learning_activity_sequence_user_combination_name_str = LEARNING_ACTIVITY_SEQUENCE_USER_COMBINATION_NAME_STR
    learning_activity_sequence_max_length_name_str = LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR
    learning_activity_sequence_user_name_str = LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR
    learning_activity_sequence_id_name_str = LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR

    learning_activity_mean_sequence_distance_name_str = LEARNING_ACTIVITY_MEAN_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_median_sequence_distance_name_str = LEARNING_ACTIVITY_MEDIAN_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_mean_normalized_sequence_distance_name_str = LEARNING_ACTIVITY_MEAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR
    learning_activity_median_normalized_sequence_distance_name_str = LEARNING_ACTIVITY_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR   

    learning_activity_sequence_sequence_id_name_str = LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR
    learning_activity_sequence_length_name_str = LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR
    learning_activity_mean_sequence_length_name_str = LEARNING_ACTIVITY_MEAN_SEQUENCE_LENGTH_NAME_STR
    learning_activity_median_sequence_length_name_str = LEARNING_ACTIVITY_MEDIAN_SEQUENCE_LENGTH_NAME_STR

    dataset_name_str = DATASET_NAME_STR
    group_field_name_str = GROUP_FIELD_NAME_STR
    user_field_name_str = USER_FIELD_NAME_STR
    sequence_str = SEQUENCE_STR

    distance_fields = [learning_activity_sequence_distance_name_str, 
                       learning_activity_normalized_sequence_distance_name_str]
    
    avg_seq_dist_fields_list = [learning_activity_mean_sequence_distance_name_str,
                                learning_activity_median_sequence_distance_name_str,
                                learning_activity_mean_normalized_sequence_distance_name_str,
                                learning_activity_median_normalized_sequence_distance_name_str]

    def __init__(self, 
                 sequence_distances_dict: dict,
                 dataset_name: str):
        self.sequence_distances_dict = sequence_distances_dict
        self.dataset_name = dataset_name

        self.sequence_distance_per_group_df = self._generate_seq_dist_per_group_df()
        self.unique_sequence_distance_per_group_df = self._generate_unique_seq_dist_per_group()
        self.avg_sequence_distance_per_group_df = self._generate_avg_seq_dist_per_group()
        self.avg_unique_sequence_distance_per_group_df = self._generate_avg_unique_seq_dist_per_group()
        self.sequence_length_per_group_df = self._generate_seq_len_per_group_df()
        self.unique_sequence_length_per_group_df = self._generate_unique_seq_len_per_group()
        self.avg_sequence_length_per_group_df = self._generate_avg_seq_len_per_group()
        self.avg_unique_sequence_length_per_group_df = self._generate_avg_unique_seq_len_per_group()
        self.avg_sequence_distance_sequence_length_per_group_df = self._generate_avg_seq_dist_seq_len_per_group_df()
        self.avg_unique_sequence_distance_unique_sequence_length_per_group_df = self._generate_avg_unique_seq_dist_seq_len_per_group_df()

        self.sequence_distance_square_matrix_per_group_dict = self._generate_seq_dist_square_matrix_per_group_dict(False, False)
        self.normalized_sequence_distance_square_matrix_per_group_dict = self._generate_seq_dist_square_matrix_per_group_dict(True, False)
        self.unique_sequence_distance_square_matrix_per_group_dict = self._generate_seq_dist_square_matrix_per_group_dict(False, True)
        self.normalized_unique_sequence_distance_square_matrix_per_group_dict = self._generate_seq_dist_square_matrix_per_group_dict(True, True)

    def _generate_seq_dist_per_group_df(self):

        seq_dist_per_group_fields_list = [SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str,
                                          SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str,
                                          SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_combination_name_str,
                                          SequenceDistanceAnalytics.learning_activity_sequence_user_combination_name_str,
                                          SequenceDistanceAnalytics.learning_activity_sequence_max_length_name_str]
        to_string_field_list = [SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_combination_name_str,
                                SequenceDistanceAnalytics.learning_activity_sequence_user_combination_name_str,]

        seq_dist_per_group_df = pd.DataFrame()

        for group, subdict in self.sequence_distances_dict[SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_user_combinations_name_str].items():
            subdict = {k:v for k,v in subdict.items() if k in seq_dist_per_group_fields_list}
            subdict = {k: (map(tuple, v) if k in to_string_field_list else v) for k,v in subdict.items()}
            df = pd.DataFrame(subdict)
            df[SequenceDistanceAnalytics.dataset_name_str] = self.dataset_name
            df[SequenceDistanceAnalytics.group_field_name_str] = group
            # df = df.sort_values(by=SequenceDistanceAnalytics.learning_activity_sequence_user_combination_name_str)

            seq_dist_per_group_df = pd.concat([seq_dist_per_group_df, df])

        seq_dist_per_group_df = seq_dist_per_group_df.reset_index(drop=True)

        seq_dist_per_group_df[SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str] = seq_dist_per_group_df[SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str] / seq_dist_per_group_df[SequenceDistanceAnalytics.learning_activity_sequence_max_length_name_str] 
        seq_dist_per_group_df = seq_dist_per_group_df[[SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str] + seq_dist_per_group_fields_list]

        return seq_dist_per_group_df

    def _generate_unique_seq_dist_per_group(self):

        unique_seq_dist_per_group_fields_list = [SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str,
                                                 SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str,
                                                 SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_combination_name_str,
                                                 SequenceDistanceAnalytics.learning_activity_sequence_max_length_name_str]
        to_string_field_list = [SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_combination_name_str]
                                
        unique_seq_dist_per_group_df = pd.DataFrame()

        for group, subdict in self.sequence_distances_dict[SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_sequence_combinations_name_str].items():
            subdict = {k:v for k,v in subdict.items() if k in unique_seq_dist_per_group_fields_list}
            subdict = {k: (map(tuple, v) if k in to_string_field_list else v) for k,v in subdict.items()}
            df = pd.DataFrame(subdict)
            df[SequenceDistanceAnalytics.dataset_name_str] = self.dataset_name
            df[SequenceDistanceAnalytics.group_field_name_str] = group
            # df = df.sort_values(by=SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_combination_name_str)

            unique_seq_dist_per_group_df = pd.concat([unique_seq_dist_per_group_df, df])
        
        unique_seq_dist_per_group_df = unique_seq_dist_per_group_df.reset_index(drop=True)

        unique_seq_dist_per_group_df[SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str] = unique_seq_dist_per_group_df[SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str] / unique_seq_dist_per_group_df[SequenceDistanceAnalytics.learning_activity_sequence_max_length_name_str] 
        unique_seq_dist_per_group_df = unique_seq_dist_per_group_df[[SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str] + unique_seq_dist_per_group_fields_list]

        return unique_seq_dist_per_group_df
        
    def _generate_avg_seq_dist_per_group(self):

        grouping_list = [SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str]

        avg_seq_dist_per_group_df = (self.sequence_distance_per_group_df.groupby(grouping_list)[SequenceDistanceAnalytics.distance_fields]
                                                                        .agg([np.mean, np.median]))
        avg_seq_dist_per_group_df = avg_seq_dist_per_group_df.droplevel(0, axis=1)
        avg_seq_dist_per_group_df = avg_seq_dist_per_group_df.reset_index()
        avg_seq_dist_per_group_df.columns = grouping_list + SequenceDistanceAnalytics.avg_seq_dist_fields_list

        return avg_seq_dist_per_group_df

    def _generate_avg_unique_seq_dist_per_group(self):
        
        grouping_list = [SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str]

        avg_unique_seq_dist_per_group_df = (self.unique_sequence_distance_per_group_df.groupby(grouping_list)[SequenceDistanceAnalytics.distance_fields]
                                                                                      .agg([np.mean, np.median]))
        avg_unique_seq_dist_per_group_df = avg_unique_seq_dist_per_group_df.droplevel(0, axis=1)
        avg_unique_seq_dist_per_group_df = avg_unique_seq_dist_per_group_df.reset_index()
        avg_unique_seq_dist_per_group_df.columns = grouping_list + SequenceDistanceAnalytics.avg_seq_dist_fields_list

        return avg_unique_seq_dist_per_group_df

    def _generate_seq_len_per_group_df(self):

        seq_len_per_group_fields_list = [SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_name_str,
                                         SequenceDistanceAnalytics.learning_activity_sequence_length_name_str]

        seq_len_per_group_df = pd.DataFrame()

        for group, subdict in self.sequence_distances_dict[SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_user_combinations_name_str].items():
            subdict = {k:v for k,v in subdict.items() if k in seq_len_per_group_fields_list}
            df = pd.DataFrame(subdict)
            df[SequenceDistanceAnalytics.dataset_name_str] = self.dataset_name
            df[SequenceDistanceAnalytics.group_field_name_str] = group
            df = df.sort_values(by=SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_name_str)

            seq_len_per_group_df = pd.concat([seq_len_per_group_df, df])

        seq_len_per_group_df = seq_len_per_group_df[[SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str] + seq_len_per_group_fields_list]

        return seq_len_per_group_df
    
    def _generate_unique_seq_len_per_group(self):
        
        grouping_list = [SequenceDistanceAnalytics.dataset_name_str,
                         SequenceDistanceAnalytics.group_field_name_str, 
                         SequenceDistanceAnalytics.learning_activity_sequence_sequence_id_name_str]       

        unique_seq_length_per_group_df = (self.sequence_length_per_group_df.groupby(grouping_list)[SequenceDistanceAnalytics.learning_activity_sequence_length_name_str]
                                                                           .first()
                                                                           .reset_index())
        
        return unique_seq_length_per_group_df

    def _generate_avg_seq_len_per_group(self):

        grouping_list = [SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str]

        avg_seq_length_per_group_df = (self.sequence_length_per_group_df.groupby(grouping_list)[SequenceDistanceAnalytics.learning_activity_sequence_length_name_str]
                                                                        .agg([np.mean, np.median]))
        avg_seq_length_per_group_df = avg_seq_length_per_group_df.reset_index()
        avg_seq_length_per_group_df.columns = grouping_list + [SequenceDistanceAnalytics.learning_activity_mean_sequence_length_name_str, SequenceDistanceAnalytics.learning_activity_median_sequence_length_name_str]

        return avg_seq_length_per_group_df

    def _generate_avg_unique_seq_len_per_group(self):

        grouping_list = [SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str]

        avg_unique_seq_length_per_group_df = (self.unique_sequence_length_per_group_df.groupby(grouping_list)[SequenceDistanceAnalytics.learning_activity_sequence_length_name_str]
                                                                               .agg([np.mean, np.median]))
        avg_unique_seq_length_per_group_df = avg_unique_seq_length_per_group_df.reset_index()
        avg_unique_seq_length_per_group_df.columns = grouping_list + [SequenceDistanceAnalytics.learning_activity_mean_sequence_length_name_str, SequenceDistanceAnalytics.learning_activity_median_sequence_length_name_str]

        return avg_unique_seq_length_per_group_df

    def _generate_avg_seq_dist_seq_len_per_group_df(self):

        avg_seq_dist_seq_len_per_group_df = pd.merge(self.avg_sequence_distance_per_group_df, 
                                                     self.avg_sequence_length_per_group_df, 
                                                     how='inner', 
                                                     on=[SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str])
        return avg_seq_dist_seq_len_per_group_df

    def _generate_avg_unique_seq_dist_seq_len_per_group_df(self):

        avg_unique_seq_dist_seq_len_per_group_df = pd.merge(self.avg_unique_sequence_distance_per_group_df, 
                                                            self.avg_unique_sequence_length_per_group_df, 
                                                            how='inner', 
                                                            on=[SequenceDistanceAnalytics.dataset_name_str, SequenceDistanceAnalytics.group_field_name_str])
        return avg_unique_seq_dist_seq_len_per_group_df

    def _generate_seq_dist_square_matrix_per_group_dict(self,
                                                        normalize_distance: bool,
                                                        use_unique_sequence_distances: bool) -> dict:

        if use_unique_sequence_distances:
            seq_dist_df = self.unique_sequence_distance_per_group_df
            sequence_distance_base = SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_sequence_combinations_name_str
            label_type = SequenceDistanceAnalytics.learning_activity_sequence_id_name_str
        else:
            seq_dist_df = self.sequence_distance_per_group_df
            sequence_distance_base = SequenceDistanceAnalytics.learning_activity_sequence_distance_based_on_user_combinations_name_str
            label_type = SequenceDistanceAnalytics.learning_activity_sequence_user_name_str
        

        square_matrices_per_group_dict = {}
        for group, df in seq_dist_df.groupby(SequenceDistanceAnalytics.group_field_name_str):

            if normalize_distance:
                distances = df[SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str]
            else:
                distances = df[SequenceDistanceAnalytics.learning_activity_sequence_distance_name_str]


            labels = (self.sequence_distances_dict[sequence_distance_base]
                                                  [group]
                                                  [label_type])

            square_matrix = squareform(distances)
        
            square_matrix_df = pd.DataFrame(square_matrix,
                                            index=labels, 
                                            columns=labels) 
            
            labels = sorted(labels, key=int)
            square_matrix_df = square_matrix_df.loc[labels, labels]
            
            square_matrices_per_group_dict[group] = square_matrix_df

        return square_matrices_per_group_dict
    
    def plot_sequence_distance_per_group(self):

        # plot sequence distance per group
        data_list = [self.sequence_distance_per_group_df, 
                     self.unique_sequence_distance_per_group_df]
        caption_list = [f'Base: All {SequenceDistanceAnalytics.user_field_name_str}-{SequenceDistanceAnalytics.sequence_str} Combinations', 
                        f'Base: All Unique-{SequenceDistanceAnalytics.sequence_str} Combinations',]
        data_caption_list = list(zip(data_list, caption_list))

        print('*'*100)
        print('*'*100)
        print(' ')
        for x_var in SequenceDistanceAnalytics.distance_fields:
            for data, caption in data_caption_list:
                print()
                print('-'*100)
                print(f'{x_var} per {SequenceDistanceAnalytics.group_field_name_str}:')
                print(caption)
                print('-'*100)
                print('\n')
                print('Plots:')
                g = sns.boxplot(data=data, 
                                x=x_var, 
                                y=SequenceDistanceAnalytics.group_field_name_str,
                                showmeans=True, 
                                meanprops=marker_config);

                if x_var == SequenceDistanceAnalytics.learning_activity_normalized_sequence_distance_name_str:
                    g.set(xlim=(-0.01, 1.01))

                for patch in g.patches:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0.5))

                g = sns.stripplot(data=data, 
                                x=x_var, 
                                y=SequenceDistanceAnalytics.group_field_name_str,
                                size=2, 
                                color="red",
                                alpha=0.1)
                g.set(xlabel=x_var);
                plt.show()
                print('*'*100)
                print('*'*100)
                print(' ')

        # plot avg sequence distance per group
        avg_per_group_data_list = [self.avg_sequence_distance_sequence_length_per_group_df, 
                                   self.avg_unique_sequence_distance_unique_sequence_length_per_group_df] 
        avg_data_caption_list = list(zip(avg_per_group_data_list, caption_list))
        for var in SequenceDistanceAnalytics.avg_seq_dist_fields_list:
            for data, caption in avg_data_caption_list:
                print('-'*100)
                print(f'{var} per {SequenceDistanceAnalytics.group_field_name_str}:')
                print(caption)
                print('-'*100)
                print('\n')
                print('Plots:')
                # sequence distance distribution
                plot_distribution(data, 
                                var,
                                var,
                                False)

                # avg sequence length vs avg sequence distance
                g = sns.regplot(data=data, 
                                x=var, 
                                y=SequenceDistanceAnalytics.learning_activity_mean_sequence_length_name_str)
                g.set(xlabel=f'{var} per {SequenceDistanceAnalytics.group_field_name_str}', 
                    ylabel=f'{SequenceDistanceAnalytics.learning_activity_mean_sequence_length_name_str} per {SequenceDistanceAnalytics.group_field_name_str}');
                plt.show()
                g = sns.regplot(data=data, 
                                x=var, 
                                y=SequenceDistanceAnalytics.learning_activity_median_sequence_length_name_str)
                g.set(xlabel=f'{var} per {SequenceDistanceAnalytics.group_field_name_str}', 
                      ylabel=f'{SequenceDistanceAnalytics.learning_activity_median_sequence_length_name_str} per {SequenceDistanceAnalytics.group_field_name_str}');
                plt.show()
            print('*'*100)
            print('*'*100)
            print(' ')
    
    def plot_sequence_distance_matrix_per_group(self,
                                                normalize_distance: bool,
                                                use_unique_sequence_distances: bool,
                                                height: int,
                                                group_str=None) -> None:
        """Plot the sequence distance matrix group.

        Parameters
        ----------
        normalize_distance : bool
            A boolean indicating whether the sequence distances are being normalized between 0 and 1
        use_unique_sequence_distances: bool
            A boolean indicating whether only unique sequences are being used as the basis for distance calculations
        height : int
            The height of the subplots
        group_str : list of str, optional
            A list of groups for which the sequence distance matrices will be plotted. If None all groups will de displayed , by default None
        """

        if (normalize_distance) and (use_unique_sequence_distances):
            square_matrix_dict = self.normalized_unique_sequence_distance_square_matrix_per_group_dict
        elif  (normalize_distance) and (not use_unique_sequence_distances):
            square_matrix_dict = self.normalized_sequence_distance_square_matrix_per_group_dict
        elif (not normalize_distance) and (use_unique_sequence_distances):
            square_matrix_dict = self.unique_sequence_distance_square_matrix_per_group_dict
        else:
            square_matrix_dict = self.sequence_distance_square_matrix_per_group_dict
        
        if group_str:
            square_matrix_dict = {k: v for k,v in square_matrix_dict.items() if (k in group_str)}
        
        square_matrices_per_group_df = pd.DataFrame()
        for group, df in square_matrix_dict.items():
            square_matrix_df = (pd.melt(df, 
                                       ignore_index=False, 
                                       var_name='row', 
                                       value_name='value')
                                  .reset_index()
                                  .rename(columns={'index': 'column'}))
            square_matrix_df[SequenceDistanceAnalytics.group_field_name_str] = group
            square_matrices_per_group_df = pd.concat([square_matrices_per_group_df, square_matrix_df])
        

        # helper function
        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            labels = d.columns
            labels_sorted = sorted(labels, key=int)
            d = d.loc[labels_sorted,labels_sorted]
            sns.heatmap(d, **kwargs)

        if normalize_distance:
            distance_str = f'Normalized {SequenceDistanceAnalytics.sequence_str} Distance Matrix per {SequenceDistanceAnalytics.group_field_name_str}:'
        else:
            distance_str = f'{SequenceDistanceAnalytics.sequence_str} Distance Matrix per {SequenceDistanceAnalytics.group_field_name_str}:'

        if use_unique_sequence_distances:
            base_str = f'Base: All Unique-{SequenceDistanceAnalytics.sequence_str} Combinations'
        else:
            base_str = f'Base: All {SequenceDistanceAnalytics.user_field_name_str}-{SequenceDistanceAnalytics.sequence_str} Combinations' 

        print('*'*100)
        print('*'*100)
        print(' ')
        print('-'*100)
        print(distance_str)
        print(' ')
        print(base_str)
        print('-'*100)
        g = sns.FacetGrid(square_matrices_per_group_df, 
                        col=SequenceDistanceAnalytics.group_field_name_str,
                        col_wrap=6, 
                        sharex=False,
                        sharey=False,
                        height=height, 
                        aspect= 1)
        if normalize_distance:
            g.map_dataframe(draw_heatmap, 
                            'column', 
                            'row', 
                            'value',
                            vmin=0,
                            vmax=1,
                            cbar=True, 
                            square = True)
        else:
            g.map_dataframe(draw_heatmap, 
                            'column', 
                            'row', 
                            'value', 
                            cbar=True, 
                            square = True)
        g.set(xlabel=SequenceDistanceAnalytics.sequence_str, 
              ylabel=SequenceDistanceAnalytics.sequence_str)
        # get figure background color
        facecolor=plt.gcf().get_facecolor()
        for ax in g.axes.flat:
            # set aspect of all axis
            ax.set_aspect('equal','box')
            # set background color of axis instance
            ax.set_facecolor(facecolor)
        plt.show()