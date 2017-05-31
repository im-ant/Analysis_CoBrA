"""
Script for some image utilites that I use.

Author: Anthony Chen
Started: May 29, 2017
"""
import os
import sys
import time
import nibabel as nib
import numpy as np

""" Test function for sanity chck """
def hello_world():
    print "Hello world!"


"""========================================================================
# Function that find the voxel-wise intensity standard deviation across a
# number of subjects in model space (memory saving method)

Note:
    - put on hold; the processing time for 100 subjects will be ~39.4 hours
    - will try to find alternative / faster methods for now

Reference I/O workflow
    img = nib.load(in_file_list[0])
    data = img.get_data()
    out_img = nib.Nifti1Image(data, img.affine)
    out_img.to_filename(mean_out_path)
========================================================================"""
def multiSubj_stdev_bySlice(subj_list):
    #Initialize the first subject's image as a reference for the image dimensions
    ref_img = nib.load(subj_list[0])
    shape = ref_img.shape # <type 'tuple'> containing int representations of x, y and z size
    datatype = ref_img.get_data_dtype()
    #Initialize the returned output array based on the ref_img's shape
    stdev_matrix = np.zeros( shape, dtype=datatype )

    #Iterate through each coronal slice (each z-dimension value)
    for z_idx in range(0,shape[2]):
        #Initialize a blank matrix which uses different subjects as the 3rd dimension
        slice_matrix_shape = ( shape[0], shape[1], len(subj_list) )
        slice_matrix = np.zeros( slice_matrix_shape, dtype=datatype )
        #Iterate through each subject and initialize the slice_matrix
        for subj_idx, subj_file in enumerate(subj_list):
            #Progree printline (does not print on new line each time!)
            sys.stdout.write('\rSlice %d out of %d; ' % (z_idx+1, shape[2]) )
            sys.stdout.write('subject %d out of %d  ' % (subj_idx+1, len(subj_list)) )
            sys.stdout.write('(%f %% done)' % ( (( z_idx+1.0)*len(subj_list)+(subj_idx+1))/(shape[2]*len(subj_list))*100.0 ) )
            sys.stdout.flush()

            #Load the subject image
            temp_img = nib.load(subj_file)
            temp_data = temp_img.get_data()

            #Copy the specific slice from this subject into my slice_matrix
            slice_matrix[:,:,subj_idx] = np.copy(temp_data[:,:,z_idx])

        #Find the standard deviation of all the subjects
        stdev_slice = np.std(slice_matrix, axis=2, dtype=datatype)
        print np.shape(stdev_slice)

        break #TODO: testline remove
    #TODO:
    print
    #TODO: testline remove all of below
    print type(shape)
    print shape
    print type(datatype)
    print datatype
    print np.shape(stdev_matrix)
    print stdev_matrix.dtype


"""========================================================================
# Function that find the voxel-wise intensity standard deviation across a
# number of subjects in model space (memory intensive method)

Note:
    - process is memory intensive as it loads all subject brains
========================================================================"""
def multiSubj_stdev_wholeBrain(subj_list, output_file_path):
    #Initialize the first subject's image as a reference for the image dimensions
    ref_img = nib.load(subj_list[0])
    shape = ref_img.shape + (len(subj_list),) #4D shape - 4th dimension is the list of subjects
    datatype = ref_img.get_data_dtype()

    #Let the user know the size RAM needed for the processing
    print "You'll need approximately",
    print "%.2f GB" % ((ref_img.get_data().nbytes * (len(subj_list)+1)) / 1000000000.0) ,
    print "of RAM to store all subject images"

    #Initialize calculation matrix
    calc_matrix = np.zeros( shape, dtype=datatype )
    #Iterate through each subject and initialize matrix
    for subj_idx, subj_file in enumerate(subj_list):
        #Print lines to let the user know progress
        sys.stdout.write('\rIntializing image data, subject %d out of %d; ' % (subj_idx+1, len(subj_list)) )
        sys.stdout.flush()
        #Open image
        subj_img = nib.load(subj_list[subj_idx])
        subj_data = subj_img.get_data()
        #Append image to 4d matrix
        calc_matrix[:,:,:,subj_idx] = np.copy(subj_data)
    #Let user know progress
    print '\nAll image initialized, beginnning standard deviation calculation'

    #Calcuate stdev
    std_matrix = np.std(calc_matrix, axis=3)
    #Output file
    print 'Writing output file to: %s' % output_file_path
    std_img = nib.Nifti1Image(std_matrix, ref_img.get_affine()) #Use the first image's affine; good practice?
    std_img.to_filename(output_file_path)
