# niistats

## Synopsis
Compute NIfTI image statistics

## Usage

```bash
niistats [options] dirs nii o
```
- `dirs`: list of subject directories containing images
- `nii`: relative path from subject directory to NIfTI image (e.g. ana/ff/calf/fatfraction.nii.gz)
- `o`: output CSV file to store results (e.g. results.csv)

## Options
- `-h`: display help message, and exit
- `-seg`: relative path from subject directory to segmentation (e.g. 
roi/muscle_seg.nii.gz)
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
- ninety-finth centile
- maximum

If a segmentation is supplied (`-seg`) statistics will be calculated for each
label in the segmentation.

Voxels outside the lower (`-l`) and upper (`-u`) thresholds are excluded - this 
will be reflected in reductions in the voxel count and volume.

## Installing
1. Create a directory to store the package e.g.:

    ```bash
    mkdir niistats
    ```

2. Create a new virtual environment in which to install `niistats`:

    ```bash
    python3 -m venv niistats-env
    ```
   
3. Activate the virtual environment:

    ```bash
    source niistats-env/bin/activate
    ```

4. Upgrade `pip` and `build`:

    ```bash
    pip install --upgrade pip
    pip install --upgrade build
    ```

5. Install using `pip`:
    ```bash
    pip install git+https://github.com/SWastling/niistats.git
    ```

## License
See [MIT license](./LICENSE)


## Authors and Acknowledgements
Dr Stephen Wastling 
([stephen.wastling@nhs.net](mailto:stephen.wastling@nhs.net))