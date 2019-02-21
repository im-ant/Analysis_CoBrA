#############################################################################
# Utility classes and functions for longitudinal analysis of data
# Python 3.6 
#############################################################################
import numpy as np
import pandas as pd
from collections import deque


####################
# Create a longitudinal data structure
# [subject ID] : record 1 <--> record 2 <--> record 3 <--> ...
#
# $dataframe: the dataframe containing subject records; each row is a
#               record of some subject at some timepoint, does not have
#               to be sorted by time
# $idColName: the string name of the column containing subject identification
#               (e.g. 'PTID'), which is used as the key to the dictionary
#               containing longitudinal records of the particular subject
#               (column name in the dataframe DataFrame)
# $timeColName: the string name of the column containing the timepoint of
#               each record, which is used to sort the dataframe to get
#               longitudinal data that is sorted by time. The column values
#               should be numeric types so comparisons can be made.
#               (column name in the dataframe DataFrame)
# $relativeTimeColName: the string name of the column containing the relative
#                       timepoint of each record, which is computed as the
#                       longitudinal dataframe is initialized (if it is set to
#                       None then it will not be initialized)
#
# Return: a dictionary with the following hashmap --> linked list structure:
#   [subject ID] : record 1 <--> record 2 <--> record 3 <--> ...
#
# Note: O(n log n) to sort (n records), O(n) to generate the data struct.
#
####################
def constructDataStruct(dataframe, idColName, timeColName,
                        relativeTimeColName=None):
    # Copy and sort the input normal dataframe
    df = dataframe.copy(deep=True)
    df.sort_values(timeColName, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Initialize the dictionary to store
    longiDict = {}

    # Itearte over each row of the (time-sorted) dataframe
    for i, row in df.iterrows():
        # Get the current subject ID / dictionary key
        subjKey = row[idColName]

        # If the current subject is already in the longitudinal data struct
        if subjKey in longiDict:
            # Do not include if the time is repeated
            if longiDict[subjKey][-1][timeColName] == row[timeColName]:
                continue
            # If the time is not repeated, compute the relative time to baseline
            if relativeTimeColName != None:
                # Compute relative time
                baselineTimeValue = longiDict[subjKey][0][timeColName]
                currentTimeValue = row[timeColName]
                currentRelativeTimeValue = currentTimeValue - baselineTimeValue
                # Add to current row
                row[relativeTimeColName] = currentRelativeTimeValue

            # Add current row record
            longiDict[subjKey].append(row)

        # If the current subject is not in the longitudinal data struct
        else:
            # Set the relative time column to 0 for first record
            if relativeTimeColName != None:
                row[relativeTimeColName] = 0
            # Add subject to dictioanry
            longiDict[subjKey] = deque()
            longiDict[subjKey].append(row)

    # Return
    return longiDict


####################
# Go from a longitudinal data structure back to a dataframe
####################
def dataStruct2Df(longiStruct):
    # List for all the series items converted to dictionaries
    dictsList = []

    # Iterate over all records and concatenate
    for subjKey in longiStruct:
        for subjRec in longiStruct[subjKey]:
            dictsList.append(subjRec.to_dict())

    # Go from a list of series to a dataframe and return
    longiDf = pd.DataFrame(dictsList)
    longiDf.reset_index(drop=True, inplace=True)
    return longiDf


####################
# Make a deep copy of the longitudinal data structure
####################
def deepcopy(longiStruct):
    # Initialize the dictionary to store
    cpDict = {}

    # Iterate over each subject in the input data structure
    for subjKey in longiStruct:
        # Initialize this subject
        cpDict[subjKey] = deque()
        # Add records of this subject to dict
        for subjRec in longiStruct[subjKey]:
            cpDict[subjKey].append( subjRec.copy(deep=True) )

    return cpDict


####################
# Given a longitudinal data-struct, get back a list of available column values
#   * typically to get back a list of timepoints
#
# $longiStruct: the longitudinal data structure
# $colName: column name to look for
####################
def getAvailableColValues(longiStruct, colName):
    # Set of timepoints
    colValSet = set([])

    # Iterate through each subject records
    for subjKey in longiStruct:
        for subjRec in longiStruct[subjKey]:
            # Add the set of times
            colValSet.add(subjRec[colName])

    # Sort and set of timepoints and return
    colValList = list(colValSet)
    colValList.sort()
    return colValList


####################
# Given a longitudinal data-struct, get a binary matrix indicating availability
# for each subject's record being present
#
# $wantedColNames: list of string denoting the column names I do not want to be
#                   null for this record
####################
def getAvaibilityMat(longiStruct, timeColname, wantedColNames):
    # Get the list of available timepoints
    tpsList = getAvailableColValues(longiStruct, timeColname)

    # Create a boolean matrix for availability
    availMat = np.full( (len(longiStruct),len(tpsList)) , False)

    # Iterate through the data structure to fill availability
    for i, subjKey in enumerate(longiStruct):
        for j, subjRec in enumerate(longiStruct[subjKey]):
            # Skip if the wanted columns are full
            if subjRec[wantedColNames].isnull().values.any():
                continue
            # Else, fill boolean array
            colIdx = tpsList.index( subjRec[timeColname] )
            availMat[i, colIdx] = True

    # Return
    return availMat


####################
# Given a longitudinal data-struct, get the relevant attributes in a Longitudinal
# structure
# [subject ID]: [attribute 1]: [(t1, value), (t2, value), (t3, value), ...]
#               [attribute 2]: [(t1, value), (t3, value), (t7, value), ...]
#
# $longiStruct: longitudinal data structure
# $timeColName: string, column name denoting time attribute (value within
#               the column must be a numeric type)
# $attributeColNames: list of strings denoting column names of the attributes
#                       which I wish to extract from the longitudinal dataframe
#
####################
def getAttributeStruct(longiStruct, timeColname, attributeColNames):
    # Initialize dictionary to store each subject
    subjAttriDict = {}

    # Iterate through each subject
    for i, subjKey in enumerate(longiStruct):
        # Initialize attribute dictionary
        subjAttriDict[subjKey] = {attriKey: None for attriKey in attributeColNames}

        # Iterate through all records of current subject and through each attribute
        for subjRec in longiStruct[subjKey]:
            for attriCol in attributeColNames:
                # Skip if the current attribute is non-existant
                if np.isnan(subjRec[attriCol]):
                    continue

                # Find the attribute value and timepoint
                curTpVal = subjRec[timeColname]
                curAttriVal = subjRec[attriCol]

                # Check if the current subject - attribute combination has been init
                if subjAttriDict[subjKey][attriCol] == None:
                    subjAttriDict[subjKey][attriCol] = []

                # Add to the attribute dictionary
                subjAttriDict[subjKey][attriCol].append( (curTpVal,curAttriVal) )

    return subjAttriDict

####################
# Given an attribute data-structure, filter for the attributes that meet a criteria,
# then apply a function to get the same-length parameters from the attribute list
#
# $attStruct : subject - attribute data structure
# $transformFunc: the function to transform a subject-attribute list-of-time-tuple
#                   to some features for subsequent processing
#                   (must output iterable!)
# $filterFunc: the function to apply over each subject-attribute list-of-time-tuple
#               to apply quality control
#
# Returns a subset of the original attribute data structure, with fitted features
# instead of the list-of-time-tuple
#
####################
def fitAttributeParam(attStruct, transformFunc, filterFunc=None):
    # Construct a new dictionary to get the attribute parameters
    subjAttParamDict = {}

    # Iterate through each subject
    for i, subjKey in enumerate(attStruct):
        # Initialize a temporarly attribute parameter dictionary
        curAttriParamDict = {attriKey: None for attriKey in attStruct[subjKey]}

        # Iterate through all attributes to see if all passes
        hasOneGoodAttribute = False
        for attriKey in attStruct[subjKey]:
            # Get the current list of tuple data
            curListOfDataTuple = attStruct[subjKey][attriKey]
            # Filtering NOT passed if: there is a filtering function AND returned false
            if filterFunc!=None and (filterFunc(curListOfDataTuple)==False):
                continue

            # If filter passed, fit parameters
            curParams = transformFunc(curListOfDataTuple)
            # Save params
            curAttriParamDict[attriKey] = curParams
            hasOneGoodAttribute = True

        # If there are good parameters, save the dictionary
        if hasOneGoodAttribute:
            subjAttParamDict[subjKey] = curAttriParamDict

    # Return
    return subjAttParamDict


####################
# Given an uniform-length (typically feature fitted) attribute data-structure,
# transform it to a pandas DataFrame
#
# $fitAttStruct: subject-feature data structure
# $idColname: what to name the column containing identifying information (i.e. the
#               keys to the upper most layer of the attribute dictionary)
#
####################
def fittedAttributeStruct2df(fitAttStruct, idColName,
                            featColNameFunc = lambda attri,idx: str(attri)+'_f_'+str(idx)):
    ## Initialize dictionary to convert to dataframe later ##
    dfDict = {}
    # Initialize all keys
    dfDict[idColName] = list(fitAttStruct.keys())
    # List to keep tract of the max index for each attribute
    attriMaxIndex = {}
    # Initialize all feature columns for the output dataframe
    for i, subjKey in enumerate(fitAttStruct):
        for j, attriKey in enumerate(fitAttStruct[subjKey]):
            # Skip if feature not present
            if fitAttStruct[subjKey][attriKey] is None:
                continue
            # Else iterate
            for k, featureList in enumerate(fitAttStruct[subjKey][attriKey]):
                # Construct dataframe column name
                curDfColName = featColNameFunc(attriKey, k)
                # If column not initialized, initialize
                if curDfColName in dfDict:
                    continue
                dfDict[curDfColName] = []
                # Keep track of the max index for each attribute
                if attriKey not in attriMaxIndex:
                    attriMaxIndex[attriKey] = 0
                attriMaxIndex[attriKey] = max(attriMaxIndex[attriKey], k)


    ## Fill dataframe dictionary ##
    # Iterate through all subjects and attributes
    for i, subjKey in enumerate(fitAttStruct):
        for j, attriKey in enumerate(fitAttStruct[subjKey]):
            # Get the list of features with this subject-attribute combination
            curFeatureList = fitAttStruct[subjKey][attriKey]
            # Enumerate over the indeces for the features
            for k in range(0, attriMaxIndex[attriKey]+1):
                # Consruct current feature column name
                curDfColName = featColNameFunc(attriKey, k)
                if curFeatureList is None:
                    dfDict[curDfColName].append(None)
                else:
                    dfDict[curDfColName].append(curFeatureList[k])

    ## convert to a dataframe ##
    featDf = pd.DataFrame.from_dict(dfDict)
    return featDf
