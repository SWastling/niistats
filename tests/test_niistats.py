import csv
import pathlib

import nibabel as nib
import numpy as np
import pytest

import niistats.niistats as niistats

THIS_DIR = pathlib.Path(__file__).resolve().parent
TEST_DATA_DIR = THIS_DIR / "test_data"


@pytest.mark.parametrize(
    "selection, expected_output",
    [
        ("3", [3]),
        ("1:5", [1, 2, 3, 4, 5]),
        ("1,2,3,7,8", [1, 2, 3, 7, 8]),
        ("1:3,7:10,15", [1, 2, 3, 7, 8, 9, 10, 15]),
    ],
)
def test_get_range(selection, expected_output):
    assert niistats.get_range(selection) == expected_output


def test_check_shape_and_orientation():

    nii_obj_1 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), np.eye(4))
    nii_obj_2 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), 2 * np.eye(4))
    nii_obj_3 = nib.nifti1.Nifti1Image(np.ones((32, 32, 18)), np.eye(4))

    assert niistats.check_shape_and_orientation(nii_obj_1, nii_obj_1)
    # Different affine
    assert not niistats.check_shape_and_orientation(nii_obj_1, nii_obj_2)
    # Different matrix size
    assert not niistats.check_shape_and_orientation(nii_obj_1, nii_obj_3)


@pytest.mark.parametrize(
    "seg_opt, sliced_opt, l_thr, u_thr, expected_output",
    [
        (
            True,
            True,
            0,
            100,
            [
                "Subject",
                "NIfTI File",
                "Segmentation File",
                "Label Name",
                "Slice",
                "Lower Threshold",
                "Upper Threshold",
                "Voxel Count",
                "Volume mm3",
                "Mean",
                "Standard Deviation",
                "Minimum",
                "First Centile",
                "Fifth Centile",
                "First Quartile",
                "Median",
                "Third Quartile",
                "Ninety-Fifth Centile",
                "Ninety-Ninth Centile",
                "Maximum",
            ],
        ),
        (
            True,
            True,
            0,
            None,
            [
                "Subject",
                "NIfTI File",
                "Segmentation File",
                "Label Name",
                "Slice",
                "Lower Threshold",
                "Voxel Count",
                "Volume mm3",
                "Mean",
                "Standard Deviation",
                "Minimum",
                "First Centile",
                "Fifth Centile",
                "First Quartile",
                "Median",
                "Third Quartile",
                "Ninety-Fifth Centile",
                "Ninety-Ninth Centile",
                "Maximum",
            ],
        ),
        (
            True,
            True,
            None,
            None,
            [
                "Subject",
                "NIfTI File",
                "Segmentation File",
                "Label Name",
                "Slice",
                "Voxel Count",
                "Volume mm3",
                "Mean",
                "Standard Deviation",
                "Minimum",
                "First Centile",
                "Fifth Centile",
                "First Quartile",
                "Median",
                "Third Quartile",
                "Ninety-Fifth Centile",
                "Ninety-Ninth Centile",
                "Maximum",
            ],
        ),
        (
            True,
            False,
            None,
            None,
            [
                "Subject",
                "NIfTI File",
                "Segmentation File",
                "Label Name",
                "Voxel Count",
                "Volume mm3",
                "Mean",
                "Standard Deviation",
                "Minimum",
                "First Centile",
                "Fifth Centile",
                "First Quartile",
                "Median",
                "Third Quartile",
                "Ninety-Fifth Centile",
                "Ninety-Ninth Centile",
                "Maximum",
            ],
        ),
        (
            False,
            False,
            None,
            None,
            [
                "Subject",
                "NIfTI File",
                "Voxel Count",
                "Volume mm3",
                "Mean",
                "Standard Deviation",
                "Minimum",
                "First Centile",
                "Fifth Centile",
                "First Quartile",
                "Median",
                "Third Quartile",
                "Ninety-Fifth Centile",
                "Ninety-Ninth Centile",
                "Maximum",
            ],
        ),
    ],
)
def test_results_fieldnames(seg_opt, sliced_opt, l_thr, u_thr, expected_output):
    assert (
        niistats.results_fieldnames(seg_opt, sliced_opt, l_thr, u_thr)
        == expected_output
    )


@pytest.mark.parametrize(
    "seg_opt, sliced_opt, expected_output",
    [
        (
            True,
            True,
            [
                "Subject",
                "NIfTI File",
                "Segmentation File",
                "Label Name",
                "Slice",
                "Error",
            ],
        ),
        (
            True,
            False,
            [
                "Subject",
                "NIfTI File",
                "Segmentation File",
                "Label Name",
                "Error",
            ],
        ),
        (
            False,
            True,
            [
                "Subject",
                "NIfTI File",
                "Slice",
                "Error",
            ],
        ),
        (
            False,
            False,
            [
                "Subject",
                "NIfTI File",
                "Error",
            ],
        ),
    ],
)
def test_error_fieldnames(seg_opt, sliced_opt, expected_output):
    assert niistats.error_fieldnames(seg_opt, sliced_opt) == expected_output


def test_get_voxel_volume():

    nii_obj_1 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), np.eye(4))
    nii_obj_2 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), 2 * np.eye(4))

    assert niistats.get_vox_volume(nii_obj_1) == 1.0
    assert niistats.get_vox_volume(nii_obj_2) == 8.0


@pytest.mark.parametrize(
    "data, l_thr, u_thr, expected_output",
    [
        (np.array([0, 1, 2, 3, 4, 5]), 2, 3, np.array([0, 0, 1, 1, 0, 0])),
        (np.array([-10, -6, -5, -3, 4, 5]), -5, -3, np.array([0, 0, 1, 1, 0, 0])),
    ],
)
def test_thresh_mask(data, l_thr, u_thr, expected_output):
    assert np.all(niistats.thresh_mask(data, l_thr, u_thr) == expected_output)


def test_get_stats_1():

    a = np.append(np.arange(101), [np.nan, np.nan, np.nan])
    vox_vol = 8
    ref_stats = {
        "Voxel Count": a.size - 3,
        "Volume mm3": (a.size - 3) * vox_vol,
        "Mean": 50,
        "Standard Deviation": np.std(np.arange(101), ddof=1),
        "Minimum": 0,
        "First Centile": 1,
        "Fifth Centile": 5,
        "First Quartile": 25,
        "Median": 50,
        "Third Quartile": 75,
        "Ninety-Fifth Centile": 95,
        "Ninety-Ninth Centile": 99,
        "Maximum": 100,
    }

    assert niistats.get_stats(a, vox_vol) == ref_stats


def test_get_stats_2():
    # Case where a.size=0
    a = np.ones(10)
    a = a[a > 10]
    vox_vol = 8
    ref_stats = {
        "Voxel Count": 0,
        "Volume mm3": 0,
        "Mean": np.NaN,
        "Standard Deviation": np.NaN,
        "Minimum": np.NaN,
        "First Centile": np.NaN,
        "Fifth Centile": np.NaN,
        "First Quartile": np.NaN,
        "Median": np.NaN,
        "Third Quartile": np.NaN,
        "Ninety-Fifth Centile": np.NaN,
        "Ninety-Ninth Centile": np.NaN,
        "Maximum": np.NaN,
    }

    assert niistats.get_stats(a, vox_vol) == ref_stats


SCRIPT_NAME = "niistats"
SCRIPT_USAGE = f"usage: {SCRIPT_NAME} [-h]"


def test_prints_help_1(script_runner):
    result = script_runner.run([SCRIPT_NAME])
    assert result.success
    assert result.stdout.startswith(SCRIPT_USAGE)


def test_prints_help_2(script_runner):
    result = script_runner.run([SCRIPT_NAME, "-h"])
    assert result.success
    assert result.stdout.startswith(SCRIPT_USAGE)


def test_prints_help_for_invalid_option(script_runner):
    result = script_runner.run([SCRIPT_NAME, "-!"])
    assert not result.success
    assert result.stderr.startswith(SCRIPT_USAGE)


def test_niistats_invalid_slices_option(script_runner):

    result = script_runner.run([SCRIPT_NAME, "a", "b", "c", "-slices", "5:-5"])
    assert not result.success
    assert result.stderr.startswith("slice range specified with -slices is not valid")


def test_niistats_invalid_seg_label_option(script_runner, tmp_path):

    result = script_runner.run(
        [SCRIPT_NAME, "a", "b", "c", "-seg", "d", "-seg_labels", "5:-5"]
    )
    assert not result.success
    assert result.stderr.startswith(
        "label range specified with -seg_labels is not valid"
    )


def test_niistats_subj_dir_missing(script_runner, tmp_path):
    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run([SCRIPT_NAME, "a", "b", str(results_csv_fp)])
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "subj_dir_missing.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_no_nii_file(script_runner, tmp_path):

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run([SCRIPT_NAME, str(tmp_path), "b", str(results_csv_fp)])
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "no_nii_file.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_4D(script_runner, tmp_path):

    nii_fp = tmp_path / "four_dim.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.ones((3, 3, 3, 3)), np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME, str(tmp_path), "four_dim.nii", str(results_csv_fp)]
    )
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "four_dims.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_2D(script_runner, tmp_path):

    nii_fp = tmp_path / "two_dim.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME, str(tmp_path), "two_dim.nii", str(results_csv_fp)]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert not error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "two_dim.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader


def test_niistats_2D_invalid_slice(script_runner, tmp_path):

    nii_fp = tmp_path / "two_dim.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME, str(tmp_path), "two_dim.nii", str(results_csv_fp), "-slices", "2:5"]
    )
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "two_dim_invalid_slice.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_3d_slices(script_runner, tmp_path):
    sl1 = np.arange(100).reshape((10, 10))
    sl2 = 2 * np.arange(100).reshape((10, 10))
    data = np.stack([sl1, sl2], axis=-1)

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(data, 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME, str(tmp_path), "test.nii", str(results_csv_fp), "-slices", "1:3"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "slices_3d_results.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "slices_3d_errors.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_seg_label_no_seg_file(script_runner, tmp_path):

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path),
        str(nii_fp),
        str(results_csv_fp),
        "-seg_labels",
        "1:5"]
    )
    assert not result.success
    assert result.stderr.startswith(
        "-seg_labels option cannot be used unless a segmentation images is specified with -seg"
    )


def test_niistats_no_seg_file(script_runner, tmp_path):

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME, str(tmp_path), "test.nii", str(results_csv_fp), "-seg", "c"]
    )
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "no_seg_file.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_mismatched_geom(script_runner, tmp_path):

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    seg_fp = tmp_path / "seg.nii"
    seg_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), np.eye(4))
    seg_obj_1.to_filename(seg_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path),
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii"]
    )
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "mismatched_geom.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_no_labels(script_runner, tmp_path):

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    seg_fp = tmp_path / "seg.nii"
    seg_obj_1 = nib.nifti1.Nifti1Image(np.zeros((10, 10)), 2 * np.eye(4))
    seg_obj_1.to_filename(seg_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path),
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii"]
    )
    assert result.success
    assert not results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "no_labels_in_seg.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_label_missing(script_runner, tmp_path):

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    seg_fp = tmp_path / "seg.nii"
    seg_obj_1 = nib.nifti1.Nifti1Image(np.ones((10, 10)), 2 * np.eye(4))
    seg_obj_1.to_filename(seg_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path),
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii",
        "-seg_labels",
        "1:2"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "one_label_not_in_seg_results.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "one_label_not_in_seg_errors.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_all_labels(script_runner, tmp_path):

    nii_fp = tmp_path / "test.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    seg_fp = tmp_path / "seg.nii"
    seg_data = np.ones((10, 10))
    seg_data[5:10, :] = 2
    seg_obj_1 = nib.nifti1.Nifti1Image(seg_data, 2 * np.eye(4))
    seg_obj_1.to_filename(seg_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path),
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert not error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "all_labels_in_seg.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader


def test_niistats_2subj(script_runner, tmp_path):

    subj_1_dp = tmp_path / "subj_01"
    subj_2_dp = tmp_path / "subj_02"

    subj_1_dp.mkdir()
    subj_2_dp.mkdir()

    sl1 = np.arange(100).reshape((10, 10))
    sl2 = 2 * np.arange(100).reshape((10, 10))
    data = np.stack([sl1, sl2], axis=-1)

    nii_obj_1 = nib.nifti1.Nifti1Image(data, 2 * np.eye(4))
    nii_obj_2 = nib.nifti1.Nifti1Image(2 * data, 2 * np.eye(4))

    nii_1_fp = subj_1_dp / "test.nii"
    nii_2_fp = subj_2_dp / "test.nii"

    nii_obj_1.to_filename(nii_1_fp)
    nii_obj_2.to_filename(nii_2_fp)

    seg_data = np.ones_like(data)
    seg_data[5:10, :, 0] = 2
    seg_data[0:5, :, 1] = 3
    seg_data[5:10, :, 1] = 4

    seg_obj = nib.nifti1.Nifti1Image(seg_data, 2 * np.eye(4))

    seg_1_fp = subj_1_dp / "seg.nii"
    seg_2_fp = subj_2_dp / "seg.nii"

    seg_obj.to_filename(seg_1_fp)
    seg_obj.to_filename(seg_2_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path) + "/subj_01",
        str(tmp_path) + "/subj_02",
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii",
        "-slices",
        "1:3",
        "-seg_labels",
        "1:5"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "2subj_results.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader

    error_csv = open(error_csv_fp, newline="")
    error_reader = list(csv.DictReader(error_csv))

    ref_error_csv_fp = TEST_DATA_DIR / "2subj_errors.csv"
    ref_error_csv = open(ref_error_csv_fp, newline="")
    ref_error_reader = list(csv.DictReader(ref_error_csv))

    assert ref_error_reader == error_reader


def test_niistats_2subj_all_slice(script_runner, tmp_path):

    subj_1_dp = tmp_path / "subj_01"
    subj_2_dp = tmp_path / "subj_02"

    subj_1_dp.mkdir()
    subj_2_dp.mkdir()

    sl1 = np.arange(100).reshape((10, 10))
    sl2 = 2 * np.arange(100).reshape((10, 10))
    data = np.stack([sl1, sl2], axis=-1)

    nii_obj_1 = nib.nifti1.Nifti1Image(data, 2 * np.eye(4))
    nii_obj_2 = nib.nifti1.Nifti1Image(2 * data, 2 * np.eye(4))

    nii_1_fp = subj_1_dp / "test.nii"
    nii_2_fp = subj_2_dp / "test.nii"

    nii_obj_1.to_filename(nii_1_fp)
    nii_obj_2.to_filename(nii_2_fp)

    seg_data = np.ones_like(data)
    seg_data[5:10, :, 0] = 2
    seg_data[0:5, :, 1] = 3
    seg_data[5:10, :, 1] = 4

    seg_obj = nib.nifti1.Nifti1Image(seg_data, 2 * np.eye(4))

    seg_1_fp = subj_1_dp / "seg.nii"
    seg_2_fp = subj_2_dp / "seg.nii"

    seg_obj.to_filename(seg_1_fp)
    seg_obj.to_filename(seg_2_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path) + "/subj_01",
        str(tmp_path) + "/subj_02",
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert not error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "2subj_all_slice_results.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader


def test_niistats_2D_luthresh(script_runner, tmp_path):

    nii_fp = tmp_path / "two_dim.nii"
    nii_obj_1 = nib.nifti1.Nifti1Image(np.arange(100).reshape((10, 10)), 2 * np.eye(4))
    nii_obj_1.to_filename(nii_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path),
        "two_dim.nii",
        str(results_csv_fp),
        "-l",
        "10",
        "-u",
        "90"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert not error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "two_dim_luthresh.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader


def test_niistats_3d_luthresh(script_runner, tmp_path):

    subj_1_dp = tmp_path / "subj_01"
    subj_2_dp = tmp_path / "subj_02"

    subj_1_dp.mkdir()
    subj_2_dp.mkdir()

    sl1 = np.arange(100).reshape((10, 10))
    sl2 = 2 * np.arange(100).reshape((10, 10))
    data = np.stack([sl1, sl2], axis=-1)

    nii_obj_1 = nib.nifti1.Nifti1Image(data, 2 * np.eye(4))
    nii_obj_2 = nib.nifti1.Nifti1Image(2 * data, 2 * np.eye(4))

    nii_1_fp = subj_1_dp / "test.nii"
    nii_2_fp = subj_2_dp / "test.nii"

    nii_obj_1.to_filename(nii_1_fp)
    nii_obj_2.to_filename(nii_2_fp)

    seg_data = np.ones_like(data)
    seg_data[5:10, :, 0] = 2
    seg_data[0:5, :, 1] = 3
    seg_data[5:10, :, 1] = 4

    seg_obj = nib.nifti1.Nifti1Image(seg_data, 2 * np.eye(4))

    seg_1_fp = subj_1_dp / "seg.nii"
    seg_2_fp = subj_2_dp / "seg.nii"

    seg_obj.to_filename(seg_1_fp)
    seg_obj.to_filename(seg_2_fp)

    results_csv_fp = tmp_path / "results.csv"
    error_csv_fp = tmp_path / "results_errors.csv"

    result = script_runner.run(
        [SCRIPT_NAME,
        str(tmp_path) + "/subj_01",
        str(tmp_path) + "/subj_02",
        "test.nii",
        str(results_csv_fp),
        "-seg",
        "seg.nii",
        "-l",
        "25",
        "-u",
        "175"]
    )
    assert result.success
    assert results_csv_fp.is_file()
    assert not error_csv_fp.is_file()

    results_csv = open(results_csv_fp, newline="")
    results_reader = list(csv.DictReader(results_csv))

    ref_results_csv_fp = TEST_DATA_DIR / "3d_luthresh.csv"
    ref_results_csv = open(ref_results_csv_fp, newline="")
    ref_results_reader = list(csv.DictReader(ref_results_csv))

    assert ref_results_reader == results_reader
