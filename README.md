# niistats

## Synopsis
Compute NIfTI image statistics

## Usage

```bash
niistats [options] dirs nii o
```
- `dirs`: list of subject directories containing images
- `nii`: relative path from subject directory to NIfTI image
- `o`: output CSV file to store results

## Options
- `-h`: display help message, and exit
- `-seg`: relative path from subject directory to segmentation
- `-seg_labels`: report statistics from specified labels in segmentation image 
using a comma separated list (like 3,4) or inclusive ranges with a colon (like 5:10)
- `-slices`: report statistics, slice-by-slice, from specified slices using a 
comma separated list (like 3,4) or inclusive ranges with a colon (like 5:10). 
`-slices` counts from _ONE_ like itk-SNAP
- `l`: lower threshold to apply to NIfTI image before calculating statistics
- `-u`: upper threshold to apply to NIfTI image before calculating statistics
- `--version`: show version, and exit

## Description
This package can be used to compute image statistics from multiple NIfTI files.
The results are saved in a CSV file e.g. `results.csv`. Any errors encountered 
during processing are saved in an accompanying CSV file e.g `results_errors.csv`.

The following statistics are computed:
- voxel count
- volume in mm3
- mean
- standard deviation
- minimum
- first centile
- fifth centile
- first quartile
- median
- third quartile
- ninety-fifth centile
- ninety-ninth centile
- maximum

If a segmentation is supplied (`-seg`) statistics will be calculated for each
label in the segmentation.

Voxels outside the lower (`-l`) and upper (`-u`) thresholds are excluded - this 
will be reflected in reductions in the voxel count and volume.

## Example Usage
Consider a project consisting of maps of the fat-fraction of the thighs and 
calves of two subjects with accompanying segmentations of the individual muscles 
organised in the following directory structure:
```bash
muscle_project/
├── common
├── subject_001
│   ├── ana
│   │   └── ff
│   │       ├── calf
│   │       │   └── fatfraction.nii.gz
│   │       └── thigh
│   │           └── fatfraction.nii.gz
│   └── roi
│       ├── calf_muscle_rois.nii.gz
│       └── thigh_muscle_rois.nii.gz
└──subject_002
    ├── ana
    │   └── ff
    │       ├── calf
    │       │   └── fatfraction.nii.gz
    │       └── thigh
    │           └── fatfraction.nii.gz
    └── roi
        ├── calf_muscle_rois.nii.gz
        └── thigh_muscle_rois.nii.gz
```
### Example 1 - Whole Calf Muscles
To extract statistics describing the fat-fraction of the muscles in the calves 
from both subjects, saving the results in `common/ana/calf_results.csv`,
you should run to following command:
```bash
niistats subject_* ana/ff/calf/fatfraction.nii.gz common/ana/calf_results.csv -seg roi/calf_muscle_rois.nii.gz
```

### Example 2 - Selected Slices of the Calf Muscles
To extract the same statistics on a slice-by-slice basis from slices 1 to 10 run:
```bash
niistats subject_* ana/ff/calf/fatfraction.nii.gz common/ana/calf_results.csv -seg roi/calf_muscle_rois.nii.gz -slice 1:10
```

### Example 3 - Thigh Muscles with Fat-Fraction Thresholds
To extract statistics describing the fat-fraction of the muscles in the thighs 
from both subjects, saving the results in `common/ana/thigh_results.csv`,
excluding any voxels where the fat-fraction is less than -10 or greater than 110,
you should run to following command:
```bash
niistats subject_* ana/ff/thigh/fatfraction.nii.gz common/ana/thigh_results.csv -seg roi/thigh_muscle_rois.nii.gz -l -10 -u 110
```


## Installing
1. Create a new virtual environment in which to install `niistats`:

    ```bash
    python3 -m venv niistats-env
    ```
   
2. Activate the virtual environment:

    ```bash
    source niistats-env/bin/activate
    ```

3. Upgrade `pip` and `build` within your virtual environment:

    ```bash
    pip install --upgrade pip
    pip install --upgrade build
    ```

4. Install using `pip`:
    ```bash
    pip install git+https://github.com/SWastling/niistats.git
    ```

## License
See [MIT license](./LICENSE)


## Authors and Acknowledgements
Dr Stephen Wastling 
([stephen.wastling@nhs.net](mailto:stephen.wastling@nhs.net))