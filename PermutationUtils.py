#############################################################################
# Utility classes and functions for permutation testing
# Python 3.6
#############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



####################
# Compute the count and proportion of a categorical attribute given subject labels
#
# $labelledSubj_df: a dataframe containing attributes describing subjects and labels
#                   for each subject
# $attributeColName: the column name in the dataframe for the attribute
# $labeColname: the column name in the dataframe for the subject group label
# $countColname: the column name to be created for the count of each attribute within
#                   each subject group label
# $propColName: the column name to be created for the proportion of each attribute
#               within each subject group label
# $proportionality: [labelGroup / attribute], specifies whether the proportion is
#                   taken to be within each subject label group, or over the different
#                   categories of the attribute
#
# Returns: pandas dataframe of the counted attributes per label
####################
def CountAttributePerLabel(labelledSubj_df, attributeColName, labelColName,
                            countColName='counts', propColName='proportions',
                            proportionality='labelGroup'):
    # Make a deep copy of of the input dataframe
    info_df = labelledSubj_df.copy(deep=True)

    # Count the attributes per group label
    count_df = info_df.groupby([labelColName,attributeColName]).size().reset_index(name=countColName)

    # Compute the proportions per subject label group
    grouped_df = count_df.groupby([labelColName,attributeColName]).agg({countColName: 'sum'})
    if proportionality == 'labelGroup':
        prop_df = grouped_df.groupby(level=0).apply(lambda x: 1 * x / float(x.sum())).reset_index()
    elif proportionality == 'attribute':
        prop_df = grouped_df.groupby(level=1).apply(lambda x: 1 * x / float(x.sum())).reset_index()
    else:
        raise ValueError('The "proportionality" parameter must only be set to "labelGroup" or "attribute"')

    # Transfer proportions and return
    count_df[propColName] = prop_df[countColName].values
    return count_df



####################
# Generate a categorical null distribution
#
# $attributeCount_df: pd.DataFrame containing each attribute and (subject group)
#                       label to allow for the generation of a null distribution
# $attributeColName: the column name containing value that specifies the attribute category
# $labelColName: the column name containing value which specifies the subject group label
# $countColname: the column name containing the count for each attribute-lable combination
# $N_SAMPLES: number of sample to generate for each (group) label
# $proportionality: [None/'labelGroup'] whether or not to output the pure count for each
#                   attribute - group label combination, or divide by the # of individuals
#                   in each group label
# $rngSeed: seed used for rng, can be fixed for reproducibility
#
# Returns:
#   NullAttDistribution: a numpy ndarray containing the null distribution values generated
#   clusLabelsArr: list containing the group label values
#   attCategoryArr: list containing the attribute category values
####################
def GenerateCatNullDistribution(attributeCount_df, attributeColName, labelColName,
                                countColName='counts', N_SAMPLES=100,
                                proportionality='labelGroup', rngSeed=None):

    #N_SAMPLES=100, sampleOut='proportionality',

    # Get the cluster labels and group size per cluster label
    clusLabelSumCount_df = attributeCount_df.groupby([labelColName]).sum().reset_index()
    clusLabelsArr = clusLabelSumCount_df[labelColName].values
    groupSizePerClusterLabel = clusLabelSumCount_df[countColName].values

    # Compute the (whole-population) probability of each attribute category occuring
    attSumCount_df = attributeCount_df.groupby([attributeColName]).sum().reset_index()
    attCategoryArr = attSumCount_df[attributeColName].values
    attCategoryCount = attSumCount_df[countColName].values
    attCategoryProb = np.divide(attCategoryCount, np.sum(attCategoryCount))

    # Initialize np array to store output [samples, attribute categories, cluster label groups]
    NullAttDistribution = np.empty((N_SAMPLES, len(attCategoryProb), len(clusLabelsArr)))

    # Initialize random number generator
    rng = np.random.RandomState(seed=rngSeed)

    # Generate null distribution for each cluster label group
    for clusLabelIdx in range(len(clusLabelsArr)):
        NullAttDistribution[:,:,clusLabelIdx] = rng.multinomial(groupSizePerClusterLabel[clusLabelIdx],
                                                                attCategoryProb,
                                                                size=N_SAMPLES)
        if proportionality is None:
            continue
        elif proportionality == 'labelGroup':
            NullAttDistribution[:,:,clusLabelIdx] = np.divide(NullAttDistribution[:,:,clusLabelIdx],
                                                              groupSizePerClusterLabel[clusLabelIdx])
        else:
            raise ValueError('The "proportionality" parameter must only be set to "labelGroup" or None')

    return NullAttDistribution, clusLabelsArr, attCategoryArr
