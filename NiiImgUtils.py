"""
Script for some image utilites that I use.

Author: Anthony Chen
Started: May 29, 2017
"""
import os
import sys
import glob
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
    - put on hold; the processing time for 100 subjects will be ~39.4 hours
    - will try to find alternative / faster methods for now

Reference I/O workflow
    img = nib.load(in_file_list[0])
    data = img.get_data()
    out_img = nib.Nifti1Image(data, img.affine)
    out_img.to_filename(mean_out_path)
========================================================================"""
def multiSubj_stdev_wholeBrain(subj_list):
    



#TODO: delete all below (test lines)
test_dir = '/data/chamal/projects/anthony/nmf_parcellation/cortical_tractMap/seg5_tract2voxel_probability_labels/model_space/'
test_subj_list = []
test_subj_list.append(os.path.join(test_dir, '100307/100307_region_seg_pct_modelSpace.nii.gz'))
test_subj_list.append(os.path.join(test_dir, '100408/100408_region_seg_pct_modelSpace.nii.gz'))
multiSubj_stdev(test_subj_list)
