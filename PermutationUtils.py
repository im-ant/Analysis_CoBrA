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
#                           indexing: [trial, attribute category count, cluster label index]
#   clusLabelsArr: list containing the group label values
#   attCategoryArr: list containing the attribute category values
####################
def GenerateCatNullDistribution(attributeCount_df, attributeColName, labelColName,
                                countColName='counts', N_SAMPLES=100,
                                proportionality='labelGroup', rngSeed=None):

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





####################
# Compute the frequency of the observed value based on a null distribution
#
# $attributeCount_df: pd.DataFrame containing each attribute and (subject group)
#                       label to allow for the generation of a null distribution
# $attributeColName: the column name containing value that specifies the attribute category
# $labelColName: the column name containing value which specifies the subject group label
# $countColname: the column name containing the count for each attribute-lable combination
# $observationColName: the observed value (in the attributeCount_df) to be compared
#                       against the (to-be generated) null distribution
# $proportionality: [None/'labelGroup'] for the null distribution generator;
#                   whether or not to output the pure count for each attribute - group
#                   label combination, or divide by the # of individuals in each
#                   group label
# $N_SAMPLES: number of null samples to generate for each (group) label
# $rngSeed: seed used for rng in null distribution generation, can be fixed
#           for reproducibility
#
# Returns: dataframe containing the percentile and standard deviation of the observation
#           given the generated null distribution
####################
def GetObservationFrequency(attributeCount_df, attributeColName, labelColName,
                            countColName='counts', observationColName='proportions',
                            proportionality='labelGroup',
                            N_SAMPLES=100, rngSeed=None):
    ## Get the null distribution and the attribute and cluster labels ##
    NullDist, labelGroups, attributeCategories = GenerateCatNullDistribution(attributeCount_df,
                                                                             attributeColName=attributeColName,
                                                                             labelColName=labelColName,
                                                                             countColName=countColName,
                                                                             N_SAMPLES=N_SAMPLES,
                                                                             proportionality=proportionality,
                                                                             rngSeed=rngSeed)

    ## Compute the frequency of observation ##
    # Array to store output
    propNullSmallerArr = np.empty(len(attributeCount_df))
    stdevFromNullMean = np.empty(len(attributeCount_df))

    # Iterate through each attribute - label
    for index, row in attributeCount_df.iterrows():
        # Find index of the label group and attribute
        labGroupIdx = np.where(labelGroups == row[labelColName])[0][0]
        attributeIdx = np.where(attributeCategories == row[attributeColName])[0][0]

        # Get observation
        observ = row[observationColName]

        # Compute porportion in the null distribution that is smaller than the observed value
        numNullSmaller = np.sum(np.less(NullDist[:,attributeIdx,labGroupIdx],observ))
        propNullSmallerArr[index] = np.divide(numNullSmaller, N_SAMPLES)

        # Compute stdev of observation from mean
        nullStdev = np.std(NullDist[:,attributeIdx,labGroupIdx])
        distanceFromNullMean = np.abs(observ - np.mean(NullDist[:,attributeIdx,labGroupIdx]))
        stdevFromNullMean[index] = np.divide(distanceFromNullMean, nullStdev)

    # Append array to dataframe
    out_df = attributeCount_df.copy(deep=True)
    out_df['proportionNullSmaller'] = propNullSmallerArr
    out_df['stdevFromNullMean'] = stdevFromNullMean
    return out_df



####################
# Compute the frequency of the observed value based on a null distribution
#
# $attributeCount_df: pd.DataFrame containing each attribute and (subject group)
#                       label to allow for the generation of a null distribution
# $attributeColName: the column name containing value that specifies the attribute category
# $labelColName: the column name containing value which specifies the subject group label
# $countColname: the column name containing the count for each attribute-lable combination
# $observationColName: the observed value (in the attributeCount_df) to be compared
#                       against the (to-be generated) null distribution
# $proportionality: [None/'labelGroup'] for the null distribution generator;
#                   whether or not to output the pure count for each attribute - group
#                   label combination, or divide by the # of individuals in each
#                   group label
# $N_SAMPLES: number of null samples to generate for each (group) label
# $rngSeed: seed used for rng in null distribution generation, can be fixed
#           for reproducibility
#
# Returns: matplotlib object with boxplot of null distribution and red dot
#           indicating the observation
####################
def BoxplotNullDistribution(attributeCount_df, attributeColName, labelColName,
                            countColName='counts', observationColName='proportions',
                            proportionality='labelGroup',
                            N_SAMPLES=100, rngSeed=None):

    ## Get the null distribution and the attribute and cluster labels ##
    NullDist, labelGroups, attributeCategories = GenerateCatNullDistribution(attributeCount_df,
                                                                             attributeColName=attributeColName,
                                                                             labelColName=labelColName,
                                                                             countColName=countColName,
                                                                             N_SAMPLES=N_SAMPLES,
                                                                             proportionality=proportionality,
                                                                             rngSeed=rngSeed)
    ## Get the group size per label
    clusLabelSumCount_df = attributeCount_df.groupby([labelColName]).sum().reset_index()
    groupSizePerClusterLabel = clusLabelSumCount_df[countColName].values

    ## Generate plots (Plot each label as a separate subplot) ##
    fig, axarr = plt.subplots(1, len(labelGroups), sharey=True)
    for i, groupLabel in enumerate(labelGroups):
        # Plot the generated null distribution
        #plt.subplot(1, len(labelGroups), i+1)
        axarr[i].boxplot(NullDist[:,:,i], labels=attributeCategories,
                            positions=[x for x in range(len(attributeCategories))])
        axarr[i].set_title('Label Group %s (subject N=%d)' % (str(groupLabel), groupSizePerClusterLabel[i]))
        axarr[i].set_xlabel(attributeColName)
        axarr[i].set_ylabel(observationColName)
        #plt.ylim([0,1])

        # Find and plot the observed value
        for j, attributeCat in enumerate(attributeCategories):
            # Search the input dataframe for the value I want
            wantedRow = attributeCount_df[(attributeCount_df[labelColName]==groupLabel) &
                                          (attributeCount_df[attributeColName]==attributeCat)]
            if not wantedRow.empty:
                observ = wantedRow[observationColName].values[0]
            else:
                observ = 0
            # Plot the observed value
            axarr[i].scatter(j, observ, c='red')

    return fig, axarr
