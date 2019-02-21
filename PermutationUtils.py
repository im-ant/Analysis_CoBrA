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
####################
def CountAttributePerLabel(labelledSubj_df, attributeColName, labelColName,
                            countColName='count', propColName='proportions',
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
