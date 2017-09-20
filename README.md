# Analysis_CoBrA
Some analysis scripts to use around the lab.

---
## Analysis Functions

#### Stability_Utils.py
Functions to analyze stability of clusters (e.g. from unsupervised clustering)
- `countCommonEdge (vec1, vec2, pointQuant)`
  - Counts the number of common edges shared between vertices in two graphs, as suggested in the [Ben-Hur 2002 Paper](http://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf) to find stability of clustered data
  - Optimized for memory usage, reducing space complexity from *O(n^2)* to *O(n)*. This is very useful when dealing with large graphs. Run-time is still reasonable through vectorized implementation.
  - See script comment for parameters

#### NiiImgUtils.py
Function to analyze multi-subject nifty (.nii) images
- `multiSubj_stdev_welford (subj_list, stdev_path)`
  - Uses [Welford's method of computing variance](http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/) to find voxel-wise standard deviation across a population of subjects
  - Memory efficient, once pass method in which the space complexity does not exceed *O(n)*.
- `multiSubj_mean(subj_list, output_file_path)`
  - Uses an iterative method to compute the voxel-wise mean across a population of subjects
  - Memory efficient method with *O(n)* space complexity

---
## Misc Scripts
Below are other more specialized scripts for analysis.

#### Tractography_Analysis.ipynb
ipython-notebook for some tractography statistics, analyze the output files from Chris' TractREC diffusion tractography scripts.

#### ImgMeanStdev_Calculator.ipynb
ipython-notebook processing script to find the means and standard deviation of multiple large structural MRI files. Uses the functions from _NiiImgUtils_.
