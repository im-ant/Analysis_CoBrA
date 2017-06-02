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
    - currently fails (likely due to memory) on 100 subjects with float64
        (does not currently work)
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
    print np.sum(std_matrix) #TODO: testline delete
    #Output file
    print 'Writing output file to: %s' % output_file_path
    std_img = nib.Nifti1Image(std_matrix, ref_img.get_affine()) #Use the first image's affine; good practice?
    std_img.to_filename(output_file_path)


"""========================================================================
# Function that find the voxel-wise intensity standard deviation across a
# number of subjects in model space in an iterative method
========================================================================"""
def multiSubj_stdev_iterative(subj_list, mean_path, stdev_path):
    #Check if the mean file is present, if not create one
    if not os.path.isfile(mean_path):
        print "Sample mean file not found, creating mean file..."
        multiSubj_mean(subj_list, mean_path)

    #Open the mean image and first image as reference
    print "Opening mean image..."
    mean_img = nib.load(mean_path)
    mean_matrix = mean_img.get_data()
    print "Initializing & processing first image..."
    ref_img = nib.load(subj_list[0])
    shape = ref_img.shape
    datatype = ref_img.get_data_dtype()
    #Initialize the 4d calculation matrix and load the first subj's data
    calc_matrix = np.zeros( (shape + (2,)), dtype=datatype )
    calc_matrix[:,:,:,0] = np.power( np.subtract( ref_img.get_data(), mean_matrix ) , 2 )
    #Iterate through the 2nd --> last subjects' data
    for subj_idx, subj_file in enumerate(subj_list):
        #Skip the first subject as it is initialized already
        if subj_idx == 0:
            continue
        #Print lines to let the user know progress
        sys.stdout.write('\rProcessing image data, subject %d out of %d; ' % (subj_idx+1, len(subj_list)) )
        sys.stdout.flush()
        #Open image, calculate square difference and store in index 1 of the 4th dimension
        curr_img = nib.load(subj_file)
        calc_matrix[:,:,:,1] = np.power( np.subtract( curr_img.get_data(), mean_matrix ) , 2 )
        #Sum the image and store the value in index 0 of the 4th dimension
        calc_matrix[:,:,:,0] = np.sum(calc_matrix, axis=3)
        #Zero the index 1 of the 4th dimension (just in case), remove if too much processing
        calc_matrix[:,:,:,1] = np.zeros(shape, dtype=datatype)

    print "\nCalculating overall stdev..."
    #Divide each element by the # of subjects to get the variance
    var_matrix = np.true_divide(calc_matrix[:,:,:,0], (len(subj_list)) )
    stdev_matrix = np.sqrt(var_matrix)

    print "Writing to output file..."
    #Write to output path
    print np.shape(stdev_matrix) #TODO: testline delete
    print np.sum(stdev_matrix) #TODO: testline delete
    stdev_img = nib.Nifti1Image(stdev_matrix, ref_img.get_affine())
    stdev_img.to_filename(stdev_path)
    print "Done!"


"""========================================================================
# Function that find the voxel-wise intensity standard deviation across a
# number of subjects in model space using Welford's method for variance

Note:
- Taken from wikipedia, this seems to be the best iterative method by far
- It gave the same output on a 3 subject sample as the wholeBrain_stdev method
========================================================================"""
def multiSubj_stdev_welford(subj_list, stdev_path):
    # Correctness checking step
    if len(subj_list) < 2:
        print "Welford's method cannot calculate variance for <2 subjects, exiting..."
        return 1
    #Initialize the first subject's data as reference
    print "Initializing reference image"
    ref_img = nib.load(subj_list[0])
    shape = ref_img.shape
    datatype = ref_img.get_data_dtype()

    #Initialize related variables for Welford's method of calculating variance
    n = 0.0 #element counter
    mean_matrix = np.zeros(shape , dtype=datatype )
    m2_matrix = np.zeros(shape , dtype=datatype )

    #welford's method - one pass method for variance and mean
    for subj_idx, subj_file in enumerate(subj_list):
        #Print lines to let the user know progress
        sys.stdout.write('\rProcessing image data, subject %d out of %d; ' % (subj_idx+1, len(subj_list)) )
        sys.stdout.flush()
        #Open image and get current subject's data
        curr_data = nib.load(subj_file).get_data()
        ### The below code was taken from wikipedia for welford's method ###
        n += 1.0
        delta = np.subtract(curr_data, mean_matrix)
        mean_matrix = np.add( mean_matrix , np.true_divide(delta, n) )
        delta2 = np.subtract( curr_data, mean_matrix )
        m2_matrix = np.add (m2_matrix, np.multiply(delta, delta2) )
        ### The above code was taken from wikipedia for welford's method ###

    #Calculate the variance and standard deviation
    print "\nCalculating overall variance & standard deviation..."
    var_matrix = np.true_divide(m2_matrix, n) #Should be (n-1), but it seems like np.std uses n?
    stdev_matrix = np.sqrt(var_matrix)
    #Write to output path
    stdev_img = nib.Nifti1Image(stdev_matrix, ref_img.get_affine())
    stdev_img.to_filename(stdev_path)
    print "Done!"



#TODO: all testline below
def stdev_testing():
    #Need to add print lines to the above methods for effective testing
    in_list = ['/data/chamal/projects/anthony/nmf_parcellation/cortical_tractMap/seg8_tract2voxel_probability_labels/model_space/100307/100307_region_seg_pct_modelSpace.nii.gz',\
     '/data/chamal/projects/anthony/nmf_parcellation/cortical_tractMap/seg8_tract2voxel_probability_labels/model_space/100408/100408_region_seg_pct_modelSpace.nii.gz',\
     '/data/chamal/projects/anthony/nmf_parcellation/cortical_tractMap/seg8_tract2voxel_probability_labels/model_space/101006/101006_region_seg_pct_modelSpace.nii.gz']
    mean_path = '/data/chamal/projects/anthony/nmf_parcellation/cortical_tractMap/seg8_tract2voxel_probability_labels/model_space/seg8_winPct_avg_modelSpace.nii.gz'
    stdev_path = '/data/chamal/projects/anthony/nmf_parcellation/cortical_tractMap/seg8_tract2voxel_probability_labels/model_space/seg8_winPct_stdev_modelSpace.nii.gz'
    print "\n================ WHOLE BRAIN CALCULATION ================"
    multiSubj_stdev_wholeBrain(in_list, stdev_path)
    print "\n================ ITERATIVE CALCULATION ================"
    multiSubj_stdev_welford(in_list, stdev_path)
    #multiSubj_stdev_iterative(in_list, mean_path, stdev_path)

#TODO: testline ends



"""========================================================================
# Function that find the voxel-wise intensity mean across a number of
# subjects in model space

Note:
    - done in an iterative fashion so it should be good for RAM
========================================================================"""
def multiSubj_mean(subj_list, output_file_path):
    #Initialize the first subject's image as a reference for the image dimensions
    print "Initializing subject 1 / reference subject..."
    ref_img = nib.load(subj_list[0])
    shape = ref_img.shape
    datatype = ref_img.get_data_dtype()

    #Initialize the 4d calculation matrix, load the first subject's data into it
    calc_matrix = np.zeros( (shape + (2,)), dtype=datatype )
    calc_matrix[:,:,:,0] = ref_img.get_data()
    #Iterate through the 2nd --> last subject's data
    for subj_idx, subj_file in enumerate(subj_list):
        #Skip the first subject as it is initialized already
        if subj_idx == 0:
            continue
        #Print lines to let the user know progress
        sys.stdout.write('\rProcessing image data, subject %d out of %d; ' % (subj_idx+1, len(subj_list)) )
        sys.stdout.flush()
        #Open image and store in index 1 of the 4th dimension
        curr_img = nib.load(subj_file)
        calc_matrix[:,:,:,1] = curr_img.get_data()
        #Sum the image and store the value in index 0 of the 4th dimension
        calc_matrix[:,:,:,0] = np.sum(calc_matrix, axis=3)
        #Zero the index 1 of the 4th dimension (just in case), remove if too much processing
        calc_matrix[:,:,:,1] = np.zeros(shape, dtype=datatype)

    print "\nCalculating overall average..."
    #Divide each element by the # of subjects to get the mean matrix
    avg_matrix = np.true_divide(calc_matrix[:,:,:,0], len(subj_list))

    print "Writing to output file..."
    #Write to output path
    avg_img = nib.Nifti1Image(avg_matrix, ref_img.get_affine())
    avg_img.to_filename(output_file_path)
    print "Done!"
