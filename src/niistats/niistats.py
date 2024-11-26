"""Compute NIfTI image statistics"""

import argparse
import csv
import operator
import pathlib
import re
import sys

import importlib.metadata
import nibabel as nib
import numpy as np

__version__ = importlib.metadata.version("niistats")


def get_range(selection):
    """
    Parse user selection

    The syntax is a comma-separated list of individual numbers (like "3") or
    inclusive ranges with a colon (like "5:10").  So if you input
    5,6,9:12 you will get [5, 6, 9, 10, 11, 12]

    :param selection: string containing the series selected
    :type selection: str
    :return: list of series
    :rtype: list[int]
    """
    valid = []
    parts = selection.split(",")
    for p in parts:
        if re.search(":", p):
            lower, upper = p.split(":")
            for i in range(int(lower), int(upper) + 1):
                valid.append(i)
        else:
            valid.append(int(p))
    return valid


def check_shape_and_orientation(a_obj, b_obj):
    """
    Compare the affine and matrix size in the header of two NIfTI files

    :param a_obj: first NIfTI object
    :type a_obj: nib.nifti1.Nifti1Image
    :param b_obj: second NIfTI object
    :type b_obj: nib.nifti1.Nifti1Image
    :return: True is matching, False if not
    :rtype: bool
    """

    a_affine = a_obj.header.get_best_affine()
    a_shape = a_obj.header.get_data_shape()

    b_affine = b_obj.header.get_best_affine()
    b_shape = b_obj.header.get_data_shape()

    if np.allclose(a_affine, b_affine) and (a_shape == b_shape):
        return True
    else:
        return False


def results_fieldnames(seg, sliced, l_thr, u_thr):
    """
    Create a set of column headings (fieldnames) for CSV file

    :param seg: switch, true if user provides a segmentation image
    :type seg: bool
    :param sliced: switch, true if user asks for statistics slice-by-slice
    :type sliced: bool
    :param l_thr: lower threshold
    :type l_thr: float
    :param u_thr: upper threshold
    :type u_thr: float
    :return: fieldnames
    :rtype: list
    """

    fieldnames = [
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
    ]

    if not seg:
        fieldnames.remove("Segmentation File")
        fieldnames.remove("Label Name")

    if not sliced:
        fieldnames.remove("Slice")

    if l_thr is None:
        fieldnames.remove("Lower Threshold")

    if u_thr is None:
        fieldnames.remove("Upper Threshold")

    return fieldnames


def error_fieldnames(seg, sliced):
    """
    Create a set of column headings (fieldnames) for error CSV file

    :param seg: switch, true if user provides a segmentation image
    :type seg: bool
    :param sliced: switch, true if user asks for statistics slice-by-slice
    :type sliced: bool
    :return: fieldnames
    :rtype: list
    """

    fieldnames = [
        "Subject",
        "NIfTI File",
        "Segmentation File",
        "Label Name",
        "Slice",
        "Error",
    ]

    if not seg:
        fieldnames.remove("Segmentation File")
        fieldnames.remove("Label Name")

    if not sliced:
        fieldnames.remove("Slice")

    return fieldnames


def get_vox_volume(nii_obj):
    """
    Determine the volume of a voxel in a NIfTI object

    :param nii_obj: input NIfTI object
    :type nii_obj: nib.nifti1.Nifti1Image
    :return: voxel volume
    :rtype: float
    """
    # using pixdim as get_zooms() doesn't work for 2D data
    pixdim = nii_obj.header["pixdim"]
    # note qfac is stored in pixdim[0] so ignore it
    return pixdim[1] * pixdim[2] * pixdim[3]


def thresh_mask(nii_data, l_thr, u_thr):
    """
    Create mask based on lower and upper thresholds

    :param nii_data: data to threshold
    :type nii_data: np.ndarray
    :param l_thr: lower threshold
    :type l_thr: float
    :param u_thr: upper threshold
    :type u_thr: float
    :return: mask
    :rtype: np.ndarray
    """

    mask = np.ones_like(nii_data)
    if l_thr is not None:
        mask[np.where(nii_data < l_thr)] = 0

    if u_thr is not None:
        mask[np.where(nii_data > u_thr)] = 0

    return mask


def get_stats(a, vox_vol):
    """
    Extract descriptive statistics from input array

    :param a: input array
    :type a: np.ndarray
    :param vox_vol: voxel volume in mm
    :type vox_vol: float
    :return: statistics results
    :rtype: dict
    """

    # remove elements that are nan
    a = a[~np.isnan(a)]

    if a.size > 0:
        p = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        mean_a = np.mean(a)
        std_a = np.std(a, ddof=1)
    else:
        p = [np.nan] * 9
        mean_a = np.nan
        std_a = np.nan

    return {
        "Voxel Count": a.size,
        "Volume mm3": a.size * vox_vol,
        "Mean": mean_a,
        "Standard Deviation": std_a,
        "Minimum": p[0],
        "First Centile": p[1],
        "Fifth Centile": p[2],
        "First Quartile": p[3],
        "Median": p[4],
        "Third Quartile": p[5],
        "Ninety-Fifth Centile": p[6],
        "Ninety-Ninth Centile": p[7],
        "Maximum": p[8],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute NIfTI image statistics")

    parser.add_argument(
        "dirs",
        help="list of subject directories containing images",
        nargs="+",
        type=pathlib.Path,
    )

    parser.add_argument(
        "nii",
        help="relative path from subject directory to NIfTI image",
        type=pathlib.Path,
    )

    parser.add_argument(
        "o",
        help="output CSV file to store results",
        type=pathlib.Path,
    )

    parser.add_argument(
        "-seg",
        help="relative path from subject directory to segmentation",
        type=pathlib.Path,
    )

    parser.add_argument(
        "-seg_labels",
        help="report statistics from specified labels in segmentation image "
        "using a comma separated list (like 3,4) or inclusive ranges with "
        "a colon (like 5:10)",
        type=str,
    )

    parser.add_argument(
        "-slices",
        help="report statistics, slice-by-slice, from specified slices using a comma "
        "separated list (like 3,4) or inclusive ranges with a colon "
        "(like 5:10). -slices counts from ONE like itk-SNAP",
        type=str,
    )

    parser.add_argument(
        "-l",
        help="lower threshold to apply to NIfTI image before calculating statistics",
        type=float,
    )

    parser.add_argument(
        "-u",
        help="upper threshold to apply to NIfTI image before calculating statistics",
        type=float,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    args = parser.parse_args()

    if args.slices is not None:
        slices_opt = get_range(args.slices)
        slices_opt = np.asarray(slices_opt) - 1
        if len(slices_opt) <= 0:
            sys.stderr.write("slice range specified with -slices is not valid\n")
            sys.exit(1)
    else:
        slices_opt = []

    if args.seg_labels is not None:
        if args.seg:
            labels = get_range(args.seg_labels)
            if len(labels) <= 0:
                sys.stderr.write(
                    "label range specified with -seg_labels is not valid\n"
                )
                sys.exit(1)
        else:
            sys.stderr.write(
                "-seg_labels option cannot be used unless a segmentation images is specified with -seg\n"
            )
            sys.exit(1)

    print("extracting statistics for subject:")
    results = []
    errors = []
    for subject_dp in args.dirs:
        subject_dp = subject_dp.resolve()
        print("*", subject_dp.name)

        if subject_dp.is_dir():
            nii_fp = subject_dp / args.nii

            if nii_fp.is_file():
                nii_obj = nib.load(nii_fp)
                nii_data = nii_obj.get_fdata()

                # for 2D images pad to make 3D with one slice
                if len(nii_data.shape) == 2:
                    nii_data = nii_data[..., np.newaxis]

                if len(nii_data.shape) <= 3:
                    num_slices = nii_data.shape[2]
                    slices_nii = range(num_slices)
                    voxel_volume = get_vox_volume(nii_obj)

                    # mask based on lower and upper thresholds
                    mask = thresh_mask(nii_data, args.l, args.u)

                    if args.seg:
                        # segmentation provided
                        seg_fp = subject_dp / args.seg

                        if seg_fp.is_file():
                            seg_obj = nib.load(seg_fp)

                            if check_shape_and_orientation(nii_obj, seg_obj):
                                seg_data = seg_obj.get_fdata()

                                # for 2D images pad to make 3D with one slice
                                if len(seg_data.shape) == 2:
                                    seg_data = seg_data[..., np.newaxis]

                                labels_in_seg = np.unique(seg_data)
                                labels_in_seg = labels_in_seg[np.nonzero(labels_in_seg)]

                                if labels_in_seg.any():
                                    if args.seg_labels is None:
                                        # use all the labels if the user didn't specify a selection
                                        labels = labels_in_seg

                                    for label in labels:
                                        if label in labels_in_seg:
                                            if len(slices_opt) > 0:
                                                # slice-by-slice
                                                for sl in slices_opt:
                                                    if sl in slices_nii:
                                                        nii_sl = nii_data[:, :, sl]
                                                        seg_sl = (
                                                            seg_data[:, :, sl]
                                                            * mask[:, :, sl]
                                                        )
                                                        roi_sl = nii_sl[seg_sl == label]

                                                        stats = get_stats(
                                                            roi_sl, voxel_volume
                                                        )
                                                        results.append(
                                                            {
                                                                "Subject": subject_dp.name,
                                                                "NIfTI File": args.nii,
                                                                "Segmentation File": args.seg,
                                                                "Label Name": label,
                                                                "Slice": sl + 1,
                                                                "Lower Threshold": args.l,
                                                                "Upper Threshold": args.u,
                                                                **stats,
                                                            }
                                                        )
                                                    else:
                                                        errors.append(
                                                            {
                                                                "Subject": subject_dp.name,
                                                                "NIfTI File": args.nii,
                                                                "Segmentation File": args.seg,
                                                                "Label Name": label,
                                                                "Slice": sl + 1,
                                                                "Error": "Slice not in NIfTI image",
                                                            }
                                                        )
                                            else:
                                                # all slices together
                                                seg_data = seg_data * mask
                                                roi_data = nii_data[seg_data == label]
                                                stats = get_stats(
                                                    roi_data, voxel_volume
                                                )
                                                results.append(
                                                    {
                                                        "Subject": subject_dp.name,
                                                        "NIfTI File": args.nii,
                                                        "Segmentation File": args.seg,
                                                        "Label Name": label,
                                                        "Lower Threshold": args.l,
                                                        "Upper Threshold": args.u,
                                                        **stats,
                                                    }
                                                )
                                        else:
                                            errors.append(
                                                {
                                                    "Subject": subject_dp.name,
                                                    "NIfTI File": args.nii,
                                                    "Segmentation File": args.seg,
                                                    "Label Name": label,
                                                    "Slice": "",
                                                    "Error": "Label not found in segmentation image",
                                                }
                                            )
                                else:
                                    errors.append(
                                        {
                                            "Subject": subject_dp.name,
                                            "NIfTI File": args.nii,
                                            "Segmentation File": args.seg,
                                            "Error": "No labels found in segmentation",
                                        }
                                    )
                            else:
                                errors.append(
                                    {
                                        "Subject": subject_dp.name,
                                        "NIfTI File": args.nii,
                                        "Segmentation File": args.seg,
                                        "Error": "NIfTI and segmentation have mismatched geometry",
                                    }
                                )
                        else:
                            errors.append(
                                {
                                    "Subject": subject_dp.name,
                                    "NIfTI File": args.nii,
                                    "Segmentation File": args.seg,
                                    "Error": "Segmentation file not found",
                                }
                            )
                    else:
                        # no segmentation
                        if len(slices_opt) > 0:
                            # slice-by-slice
                            for sl in slices_opt:
                                if sl in slices_nii:
                                    nii_sl = nii_data[:, :, sl]
                                    mask_sl = mask[:, :, sl]
                                    stats = get_stats(nii_sl[mask_sl > 0], voxel_volume)
                                    results.append(
                                        {
                                            "Subject": subject_dp.name,
                                            "NIfTI File": args.nii,
                                            "Slice": sl + 1,
                                            "Lower Threshold": args.l,
                                            "Upper Threshold": args.u,
                                            **stats,
                                        }
                                    )
                                else:
                                    errors.append(
                                        {
                                            "Subject": subject_dp.name,
                                            "NIfTI File": args.nii,
                                            "Slice": sl + 1,
                                            "Error": "Slice not in NIfTI image",
                                        }
                                    )
                        else:
                            # all slices together
                            stats = get_stats(nii_data[mask > 0], voxel_volume)
                            results.append(
                                {
                                    "Subject": subject_dp.name,
                                    "NIfTI File": args.nii,
                                    "Lower Threshold": args.l,
                                    "Upper Threshold": args.u,
                                    **stats,
                                }
                            )
                else:
                    errors.append(
                        {
                            "Subject": subject_dp.name,
                            "NIfTI File": args.nii,
                            "Error": "NIfTI image should not have more than 3-dimensions",
                        }
                    )
            else:
                errors.append(
                    {
                        "Subject": subject_dp.name,
                        "NIfTI File": args.nii,
                        "Error": "NIfTI file not found",
                    }
                )
        else:
            errors.append(
                {
                    "Subject": subject_dp.name,
                    "Error": "Subject directory not found",
                }
            )

    if len(slices_opt) > 0:
        if args.seg:
            results = sorted(results, key=operator.itemgetter("Label Name", "Slice"))
        else:
            results = sorted(results, key=operator.itemgetter("Slice"))
    else:
        if args.seg:
            results = sorted(results, key=operator.itemgetter("Label Name"))

    output_fp = args.o.resolve()
    if results:
        with open(output_fp, "w", newline="") as csvfile:
            fieldnames = results_fieldnames(
                args.seg, len(slices_opt) > 0, args.l, args.u
            )
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()

            for row in results:
                writer.writerow(row)

    if errors:
        error_output_fp = output_fp.parent / (output_fp.stem + "_errors.csv")
        with open(error_output_fp, "w", newline="") as csvfile:
            fieldnames = error_fieldnames(args.seg, len(slices_opt) > 0)
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()

            for row in errors:
                writer.writerow(row)


if __name__ == "__main__":  # pragma: no cover
    main()
