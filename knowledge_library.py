import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import seaborn as sns

sns.set()

from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D


complement_indicator_field = 'is_complement'

def complete_missing_x_y_pairs(data, X_field, Y_field, event_field):
    x_values = data[X_field].unique()
    y_values = data[Y_field].unique()
    # Having all possible values of x and y, if i found that some of the pairs dont actually happen
    # in the data append those whit zero counts.
    values_to_append = []
    for x in x_values:
        for y in y_values:
            if len(data[(data[X_field]==x) & (data[Y_field]==y)]) == 0:
                values_to_append.append({X_field:x, Y_field:y, event_field:0})
    if len(values_to_append) > 0:
        return pd.concat([data, pd.DataFrame(values_to_append)]).fillna(0.)
    else:
        return data

def calculate_event_count(data, field, event_field):
    return data.merge(data.groupby(field)[event_field].sum().reset_index().rename(columns={event_field:'{}_total'.format(field)}),
                      on=field)


def joint_count_calculation(data, X_field, Y_field, event_field):
    data = data.groupby([X_field, Y_field])[event_field].sum().reset_index()
    data = calculate_event_count(data, X_field, event_field)
    data = calculate_event_count(data, Y_field, event_field)
    return data


def joint_prob_calculation(data, X_field, Y_field, event_field):
    data = joint_count_calculation(data, X_field, Y_field, event_field)
    data['{}_{}_count'.format(X_field, Y_field)] = data[event_field]
    data['{}_{}_prob'.format(X_field, Y_field)] = data[event_field] / data[event_field].sum()
    return data


def marginal_prob_calculation(data, X_field, Y_field, event_field):
    data = joint_prob_calculation(data, X_field, Y_field, event_field)
    data['{}_prob_given_{}'.format(X_field, Y_field)] = data[event_field] / data['{}_total'.format(Y_field)]
    data['{}_prob_given_{}'.format(Y_field, X_field)] = data[event_field] / data['{}_total'.format(X_field)]
    return data


def sample_sufficient_statistics(data, X_field, Y_field):
    N = data['{}_total'.format(X_field)]
    x_ = data['{}_prob_given_{}'.format(X_field, Y_field)].fillna(0.)
    var = ((1 / N) * x_ * (1 - x_)).fillna(1.)
    return x_, var**0.5


def godness_of_fit_to_bernoulli_sample_test(sample_one, sample_two, X_field, Y_field):
    sample_one_mean, sample_one_variance = sample_sufficient_statistics(sample_one, X_field, Y_field)
    sample_two_mean, sample_two_variance = sample_sufficient_statistics(sample_two, X_field, Y_field)
    z_statistic = (sample_one_mean - sample_two_mean) / np.sqrt(sample_one_variance + sample_two_variance)
    return z_statistic, sps.norm.sf(z_statistic), sps.norm.cdf(z_statistic)

def welch_statistic(sample_one, sample_two, X_field, Y_field):
    m1, s1 = sample_sufficient_statistics(sample_one, X_field, Y_field)
    N1 = sample_one['{}_total'.format(Y_field)]
    m2, s2 = sample_sufficient_statistics(sample_two, X_field, Y_field)
    N2 = sample_two['{}_total'.format(Y_field)]
    return (m1 - m2) / np.sqrt((s1 ** 2 / N1) + (s2 ** 2 / N2))

def welch_degrees_of_freedom(sample_one, sample_two, X_field, Y_field):
    _, s1 = sample_sufficient_statistics(sample_one, X_field, Y_field)
    N1 = sample_one['{}_total'.format(Y_field)]
    n1 = N1 - 1
    _, s2 = sample_sufficient_statistics(sample_two, X_field, Y_field)
    N2 = sample_two['{}_total'.format(Y_field)]
    n2 = N2 - 1

    n = ( ((s1 ** 2) / N1) + ((s2 **2 ) / N2)) ** 2
    d = ( (s1 ** 4) / ((N1 ** 2) * n1) ) + ( (s2 ** 4) / ((N2 ** 2) * n2) )

    return n / d


# Welch test of independent samples with the null hypothesis that the mean of the samples are the same.
def welch_test(sample_one, sample_two, X_field, Y_field):
    mean1, std1 = sample_sufficient_statistics(sample_one, X_field, Y_field)
    nobs1 = sample_one['{}_total'.format(Y_field)].iloc[0]
    mean2, std2 = sample_sufficient_statistics(sample_two, X_field, Y_field)
    nobs2 = sample_two['{}_total'.format(Y_field)].iloc[0]
    nobs_aprox = min(nobs1, nobs2)
    mine_statistic = welch_statistic(sample_one, sample_two, X_field, Y_field)
    statistic, p_value = sps.ttest_ind_from_stats(mean1, std1, nobs_aprox, mean2, std2, nobs_aprox, equal_var=False)

    #assert (statistic == mine_statistic).all()
    p_value = np.nan_to_num(p_value, .5)
    # Symmetric distribution backups a one tailed test with half of the p-value.
    # https://stackoverflow.com/questions/15984221/how-to-perform-two-sample-one-tailed-t-test-with-numpy-scipy
    return statistic, \
           (p_value / 2.) if statistic.iloc[0] > 0 else 1 - (p_value / 2.), \
           (p_value / 2.) if statistic.iloc[0] < 0 else 1 - (p_value / 2.)


def y_value_partition(data, X_field, Y_field, event_field, y_value):
    y_value_data = data[data[Y_field]==y_value].copy()

    if complement_indicator_field in list(data):
        data = data[data[complement_indicator_field]==False]

    not_y_value_data = data[data[Y_field]!=y_value].copy()
    not_y_value_data[Y_field] = '{} COMPLEMENT'.format(y_value)

    return pd.concat([y_value_data, not_y_value_data])


def y_value_complement_calculation(data, X_field, Y_field, event_field, y_value):
    y_value_divided_data = y_value_partition(data, X_field, Y_field, event_field, y_value)
    assert y_value_divided_data[event_field].sum() == data[event_field].sum()
    not_y_value_marginal_distribution = marginal_prob_calculation(y_value_divided_data, X_field, Y_field, event_field)

    return not_y_value_marginal_distribution[not_y_value_marginal_distribution[Y_field]!=y_value]


# One sample in this case is the events with an specific value. The other sample is the rest of the set.
def y_value_bernoulli_x_sample_test(data, X_field, Y_field, event_field, x_value, y_value):

    test_statistic_label = '{}_z_statistic'.format(Y_field)
    test_greater_hyp_label = '{}_isgreater_p_value'.format(Y_field)
    test_lower_hyp_label = '{}_islower_p_value'.format(Y_field)

    y_value_data = data[(data[X_field]==x_value) & (data[Y_field]==y_value)].copy().reset_index()

    y_value_x_, y_value_var = sample_sufficient_statistics(y_value_data, X_field, Y_field)
    # Since the x_ is a predictor of p (probability of success(probability of X_field being x when Y_field is y_value))
    # it should be always between 0 and 1, and since the variance is being calculated from a Bernoulli distribution it
    # also should be between 0 and 1.
    assert (0 <= y_value_x_).all() and (y_value_x_ <= 1.).all()
    assert (0 <= y_value_var).all() and (y_value_var <= 1.).all()

    # If complement set is explicitly annotated i have to remove it from the calculation.
    not_y_value_data = y_value_complement_calculation(data, X_field, Y_field, event_field, y_value).reset_index()
    not_y_value_data = not_y_value_data[not_y_value_data[X_field]==x_value].reset_index()
    y_value_is_the_entire_sample = (y_value_data[event_field].sum() == 0)
    if not y_value_is_the_entire_sample:

        not_y_value_x_, not_y_value_var = sample_sufficient_statistics(not_y_value_data, X_field, Y_field)

        # One sample in this case is the events with an specific value. The other sample is the rest of the set.
        z_statistic, \
        y_value_has_greater_mean_p_value, \
        y_value_has_lower_mean_p_value = welch_test(y_value_data, not_y_value_data, X_field, Y_field)
    else:
        # There is no knowledge(hence 0.5 prob) about whether this y values is lower than other
        # because there is no data for other values.
        z_statistic, y_value_has_greater_mean_p_value, y_value_has_lower_mean_p_value = 0, 0.5, 0.5

    y_value_data[test_statistic_label] = z_statistic
    y_value_data[test_greater_hyp_label] = y_value_has_greater_mean_p_value
    y_value_data[test_lower_hyp_label] = y_value_has_lower_mean_p_value
    y_value_data[complement_indicator_field] = False

    # Complement set recieves opposite treatment.
    not_y_value_data[test_statistic_label] = z_statistic
    not_y_value_data[test_lower_hyp_label] = y_value_has_greater_mean_p_value
    not_y_value_data[test_greater_hyp_label] = y_value_has_lower_mean_p_value
    not_y_value_data[complement_indicator_field] = True

    return pd.concat([y_value_data[y_value_data[X_field]==x_value], not_y_value_data])


def annotate_favored_hypothesis(data, Y_field, significance_level):
    data['favored hypothesis'] = 'null'
    data.loc[data['{}_islower_p_value'.format(Y_field)] <= significance_level,
               'favored hypothesis'] = 'significantly_lower'
    data.loc[data['{}_isgreater_p_value'.format(Y_field)] <= significance_level,
               'favored hypothesis'] = 'significantly_greater'
    data['significance_level'] = significance_level
    return data


def joint_distribution(data, X_field, Y_field, event_field='count', significance_level=.01):
    """ Calculation of empirical joint distribution over the entire set for variables described by X_field and Y_field.
    Events are being consider to be the rows of the data set.
    The frequentist's histogram method is being used to define an empirical distribution. Hence the probabilities of
    an event are being calculated by:
    x-value_y-value_prob = x-value_y-value_count / total_event_count

    Similarly the probabilities are being calculated using the product rule:
    x_prob_given_y = x_y_prob / y_prob

    Hence by:
    x-value_prob_given_y-value = (x-value_y-value_count / total_event_count) / (y-value_count / total_event_count)
    x-value_prob_given_y-value = (x-value_y-value_count / y-value_count)

    and the same is done for Y variable.

    Finally in order to inform significant variation of x prob for the y values a Bernoulli sample independence test
    is performed to prove that for the events taking the value of the Y variable behaves differently from the events
    taking other values.
    """
    joint_df = complete_missing_x_y_pairs(marginal_prob_calculation(data.copy(), X_field, Y_field, event_field),
                                          X_field, Y_field, event_field)

    marginal_X_ber_test_dfs = [y_value_bernoulli_x_sample_test(joint_df.copy(), X_field, Y_field, event_field, x, y)
                               for x in joint_df[X_field].unique()
                               for y in joint_df[Y_field].unique()]
    return annotate_favored_hypothesis(pd.concat(marginal_X_ber_test_dfs),
                                       Y_field, significance_level).drop_duplicates()


def plot_Y_distribution_marginalized_by_X(data, X_field, Y_field, X_values_of_interest=None,
                                          Y_values_of_interest=None, significance_level=.01):
    data = data.sort_values(Y_field)
    for Y_value in Y_values_of_interest or data[Y_field].unique():
        resolve_and_plot_joint_distribution(data[data[Y_field]==Y_value], X_field=X_field, Y_field=Y_field,
                                  X_values_of_interest=X_values_of_interest,
                                  Y_values_of_interest=Y_values_of_interest, significance_level=.01)
        plt.show()

def plot_joint_distribution_bar_chart(joint_data, X_field, Y_field, chart_field,
                                      significance_level=.01, lower_color='b', greater_color='r'):
    plt.interactive(False)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(chart_field)
    sns.barplot(x=X_field, y=chart_field, hue=Y_field, data=joint_data, ax=ax,
                   palette=sns.color_palette("Paired", 24))

    plt.xticks(rotation=90)

    patch_index = 0
    if significance_level is not None:
        # Some distinguish draw for the statistically different values
        for y_value_index, y_value in enumerate(joint_data[Y_field].unique()):
            for x_value_index, x_value in enumerate(joint_data[X_field].unique()):
                row = joint_data[(joint_data[X_field]==x_value) & (joint_data[Y_field]==y_value)]
                if not row.empty:
                    if row['favored hypothesis'].iloc[0]=='significantly_lower':
                        ax.patches[patch_index].set_edgecolor(lower_color)
                        ax.patches[patch_index].set_linewidth(1.)
                    if row['favored hypothesis'].iloc[0]=='significantly_greater':
                        ax.patches[patch_index].set_edgecolor(greater_color)
                        ax.patches[patch_index].set_linewidth(1.)

                patch_index += 1
    plt.tight_layout()
    plt.show()


def plot_joint_distribution_heatmap_chart(heatmap_data, joint_data, X_field, Y_field, chart_field, significance_level,
                                          lower_color='b', greater_color='r'):
    fig2, ax2 = plt.subplots(figsize=(10,10))
    sns.heatmap(heatmap_data, annot=True, ax=ax2, fmt='.2f')

    if significance_level is not None:
        for x_value_index, x_value in enumerate(joint_data[X_field].unique()):
            for y_value_index, y_value in enumerate(joint_data[Y_field].unique()):
                row = joint_data[(joint_data[X_field]==x_value) & (joint_data[Y_field]==y_value)]
                if not row.empty:
                    if row['favored hypothesis'].iloc[0]=='significantly_lower':
                        ax2.add_patch(Rectangle((x_value_index, joint_data[Y_field].nunique() - y_value_index - 1),
                                               1, 1, fill=False, edgecolor=lower_color, lw=1))
                    if row['favored hypothesis'].iloc[0]=='significantly_greater':
                        ax2.add_patch(Rectangle((x_value_index, joint_data[Y_field].nunique() - y_value_index - 1),
                                               1, 1, fill=False, edgecolor=greater_color, lw=1))

    x_values = list(joint_data[X_field].unique())
    plt.xticks(range(len(x_values)), x_values, rotation=90)
    plt.tight_layout()
    plt.show()


def plot_joint_distribution_3dbar_chart(joint_data, X_field, Y_field, chart_field,
                                        lower_color=(0., 0., .85, 1.),
                                        greater_color=(.85, 0., 0., 1.),
                                        neutral_color=(0., .25, .0, 0.25),
                                        add_x_label=False, add_y_label=False):
    plt.interactive(True)
    fig_3d = plt.figure(figsize=(10,10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    joint_data.sort_values([Y_field, chart_field, X_field], ascending=[True, False, True], inplace=True)

    x_values = list(joint_data[X_field].drop_duplicates())
    joint_data[X_field + '_idx'] = joint_data[X_field].apply(lambda x: x_values.index(x))

    y_values = list(joint_data[Y_field].drop_duplicates())
    joint_data[Y_field + '_idx'] = joint_data[Y_field].apply(lambda y: y_values.index(y))

    x_data = joint_data[X_field + '_idx'].as_matrix()
    y_data = joint_data[Y_field + '_idx'].as_matrix()
    z_data = joint_data[chart_field].as_matrix()
    colors = joint_data['favored hypothesis'].apply(lambda x: greater_color if x == 'significantly_greater'
                                                              else lower_color if x == 'significantly_lower'
                                                              else neutral_color)
    joint_data['color'] = colors

    ax_3d.bar3d(x_data, y_data, np.zeros(len(joint_data)), 1, 1, z_data, color=colors, linewidth=1.)

    _, x_labels_texts = plt.xticks(range(len(x_values)), x_values, rotation=90, fontsize=8)
    _, y_labels_texts = plt.yticks(range(len(y_values)), y_values, rotation=-90, fontsize=8)

    for x_index, x_label_text in enumerate(x_labels_texts):
        avg_color_of_value = tuple(map(np.mean,
                                       zip(*joint_data[joint_data[X_field].astype(str)==x_label_text._text]['color'])))
        x_label_text.set_color(avg_color_of_value)

    for y_index, y_label_text in enumerate(y_labels_texts):
        avg_color_of_value = tuple(map(np.mean,
                                       zip(*joint_data[joint_data[Y_field].astype(str)==y_label_text._text]['color'])))
        y_label_text.set_color(avg_color_of_value)

    if add_x_label:
        ax_3d.set_xlabel(X_field)
    if add_y_label:
        ax_3d.set_ylabel(Y_field)

    plt.show()


def plot_joint_distribution(joint_data, X_field, Y_field, chart_field,
                            X_values_of_interest=None, Y_values_of_interest=None, significance_level=None,
                            do_bar_chart=False, do_heatmap_chart=False, do_3dbar_chart=True,
                            add_x_label=False, add_y_label=False):
    if X_values_of_interest is not None:
        joint_data = joint_data[joint_data[X_field].isin(X_values_of_interest)]
        if joint_data.empty:
            raise Exception("You dropped all the data with X filters!")

    if Y_values_of_interest is not None:
        joint_data = joint_data[joint_data[Y_field].isin(Y_values_of_interest)]
        if joint_data.empty:
            raise Exception("You dropped all the data with Y filters!")

    joint_data.set_index(X_field)

    if do_bar_chart:
        plot_joint_distribution_bar_chart(joint_data, X_field, Y_field, chart_field, significance_level)

    if do_heatmap_chart:
        heatmap_data = joint_data[[X_field, Y_field, chart_field]].pivot_table(index=Y_field, columns=X_field,
                                                                               values=chart_field).fillna(0)
        plot_joint_distribution_heatmap_chart(heatmap_data, joint_data, X_field, Y_field, chart_field,
                                              significance_level)

    if do_3dbar_chart:
        plot_joint_distribution_3dbar_chart(joint_data, X_field, Y_field, chart_field,
                                            add_x_label=add_x_label, add_y_label=add_y_label)




def resolve_and_plot_joint_distribution(data, X_field, Y_field, event_field='count', chart_field=None,
                                        X_values_of_interest=None, Y_values_of_interest=None,
                                        significance_level=.01, qx=None, qy=None,
                                        do_bar_chart=False, do_heatmap_chart=False, do_3dbar_chart=True,
                                        add_x_label=True, add_y_label=True, sort_fields=None):

    chart_field = chart_field if chart_field is not None else '{}_prob_given_{}'.format(X_field, Y_field)

    df = data.copy()
    if (str(data[X_field].dtype) not in ['object', 'category']):
        df[X_field] = pd.cut(df[X_field], qx if qx is not None else df[X_field].nunique(), precision=10)

    if (str(data[Y_field].dtype) not in ['object', 'category']):
        df[Y_field] = pd.cut(df[Y_field], qy if qy is not None else df[Y_field].nunique(), precision=10)

    joint_data = joint_distribution(df, X_field, Y_field, event_field, significance_level)

    plot_joint_distribution(joint_data, X_field, Y_field, chart_field,
                            X_values_of_interest, Y_values_of_interest, significance_level,
                            do_bar_chart=do_bar_chart, do_heatmap_chart=do_heatmap_chart,
                            do_3dbar_chart=do_3dbar_chart, add_x_label=add_x_label, add_y_label=add_y_label)
    return joint_data


def aggregate_samples(data_df, sample_features_fields, aggregated_field='sample'):
    '''Separate samples for combination of the sample fields'''
    data_df = data_df.copy()
    data_df[aggregated_field]= data_df.apply(lambda row: '( ' +
                                                         ', '.join(['{value}'.format(value=row[feature_field])
                                                         for feature_field
                                                         in sample_features_fields])+
                                                         ' )',
                                             axis=1)
    return data_df

def discriminate_statistically(raw_data_df, sample_field='sample', discriminant_field='event',
                               occurrences_field='count',
                               samples_of_interest=None, discriminant_values_of_interest=None,
                               significance_level=0, qx=None, qy=None,
                               add_sample_label=True, add_discriminant_label=True,
                               show_complements=False, only_statistically_significant_samples=False):
    logging.info('Started discriminate_statistically')
    if 'auto:' in sample_field:
        from sklearn import tree, preprocessing
        categorical_columns = list(raw_data_df.select_dtypes(include=['object']))
        tmp_df = raw_data_df.copy()
        for column in categorical_columns:
            labelEncoder = preprocessing.LabelEncoder().fit(tmp_df[column].dropna().unique())
            tmp_df[column] = labelEncoder.transform(tmp_df[column].dropna())

        decision_tree = tree.DecisionTreeClassifier()
        decision_tree.fit(X=tmp_df[[column
                                    for column
                                    in list(tmp_df)
                                    if column != discriminant_field and column != occurrences_field]],
                          y=tmp_df[discriminant_field], sample_weight=tmp_df[occurrences_field])
        feature_importances = decision_tree.feature_importances_

        feature_importances_df = pd.DataFrame([{'feature': list(raw_data_df)[feature_index],
                                                'importance': feature_importance}
                                               for feature_index, feature_importance
                                               in enumerate(feature_importances)]).sort_values('importance',
                                                                                               ascending=False)
        sample_field = feature_importances_df['feature'].tolist()[:int(sample_field[-1])]


    # If user requested to build samples from more than one column a new field is built with the values of all those
    # columns.
    if isinstance(sample_field, list):
        final_sample_field = '(' + ','.join(sample_field) + ')'
        raw_data_df = aggregate_samples(raw_data_df, sample_field, final_sample_field)
        samples_of_interest_values_df = raw_data_df[sample_field].drop_duplicates()

        # If the user defined some values of particular interest this are expected as a list of lists.
        # The first dimension ranges among the columns that constitute every sample and the second one ranges along the
        # requested values.
        if samples_of_interest is not None:
            if isinstance(samples_of_interest, list):
                # First we agregate all the data to have only the fields that the user is interested in.
                samples_of_interest_values_df = aggregate_samples(samples_of_interest_values_df, sample_field, final_sample_field)
                # Then filter by filter the rows of such aggregated dataframe are reduced.
                for field_index, requested_field_values in enumerate(samples_of_interest):
                    filter = samples_of_interest_values_df[sample_field[field_index]].isin(requested_field_values)
                    samples_of_interest_values_df = samples_of_interest_values_df[filter]
                # Finally for the rows that survived the values of the field builded to identify the sample are taken.
                samples_of_interest = samples_of_interest_values_df[final_sample_field].unique()
        else:
            samples_of_interest = aggregate_samples(samples_of_interest_values_df, sample_field, final_sample_field)[final_sample_field].unique()
        sample_field = final_sample_field

    if not show_complements:
        if samples_of_interest is None:
            samples_of_interest = raw_data_df[sample_field].unique()
        if discriminant_values_of_interest is None:
            discriminant_values_of_interest = raw_data_df[discriminant_field].unique()

    logging.info('Samples segregated.')
    # Count successes and failures and add an artificial sample in order to obtain the average
    # of all the samples as its complement. This complement shall be obtained through the library's functionality.
    # This reshaping is needed in order to use the library since it counts for samples that are expressed
    # likeways(as rows).
    sample_data_df = pd.concat([raw_data_df,
                                pd.DataFrame([{sample_field: 'NONE',
                                               discriminant_field: 'NONE',
                                               occurrences_field: 1}],
                                             index=None)
                                ]).groupby([sample_field, discriminant_field])[occurrences_field].sum().reset_index()

    # Add total occurrences data by aggregating withouth the discriminant field.
    sample_data_df = sample_data_df.merge(sample_data_df.groupby(sample_field)[occurrences_field].sum().reset_index()\
                                                       .rename(columns={occurrences_field: 'total'}))

    conditional_probability_label = '{}_prob_given_{}'.format(discriminant_field, sample_field)

    logging.info('Data prepeared.')

    joint_distribution_df = resolve_and_plot_joint_distribution(sample_data_df,
                                                                X_field=sample_field,
                                                                Y_field=discriminant_field,
                                                                X_values_of_interest=samples_of_interest,
                                                                Y_values_of_interest=discriminant_values_of_interest,
                                                                chart_field=conditional_probability_label,
                                                                qx=qx, qy=qy,
                                                                significance_level=significance_level,
                                                                add_x_label=add_sample_label,
                                                                add_y_label=add_discriminant_label)

    logging.info('Distribution calculated.')

    # Once we calculated the results of the hypothesis test and we know which hypothesis were favored
    # then we show the metrics that lead to such results. These are again filtered by the values of interest.
    df = joint_distribution_df[joint_distribution_df['favored hypothesis'] != 'null']
    df = df[[conditional_probability_label, sample_field, discriminant_field, 'favored hypothesis']]

    if only_statistically_significant_samples:
        statistically_significant_samples = df[sample_field].unique().tolist()
        samples_of_interest = statistically_significant_samples if samples_of_interest is None\
                              else samples_of_interest + statistically_significant_samples

    if samples_of_interest is not None:
        df = df[df[sample_field].isin(samples_of_interest)]

    if discriminant_values_of_interest is not None:
        df = df[df[discriminant_field].isin(discriminant_values_of_interest)]

    logging.info('Merging with original values.')
    return df.merge(sample_data_df[[sample_field, discriminant_field, occurrences_field, 'total']])\
             .sort_values(conditional_probability_label, ascending=False)


def resolve_and_plot_joint_distribution_with_quantiles(data, X_field, Y_field, event_field='count', chart_field=None,
                              X_values_of_interest=None, Y_values_of_interest=None,
                              significance_level=.01, qx=None, qy=None, add_x_label=False, add_y_label=False):

    chart_field = chart_field if chart_field is not None else '{}_prob_given_{}'.format(X_field, Y_field)

    df = data.copy()
    if (str(data[X_field].dtype) not in ['object', 'category']):
        df[X_field] = pd.cut(df[X_field], qx if qx is not None else df[X_field].nunique())

    if (str(data[Y_field].dtype) not in ['object', 'category']):
        df[Y_field] = pd.cut(df[Y_field], qy if qy is not None else df[Y_field].nunique())


    joint_data = joint_distribution(df, X_field, Y_field, event_field)

    plot_joint_distribution(joint_data, X_field, Y_field, chart_field,
                            X_values_of_interest, Y_values_of_interest, significance_level, add_x_label, add_y_label)
    return joint_data