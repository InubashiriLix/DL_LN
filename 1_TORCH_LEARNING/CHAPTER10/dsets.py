#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :10.1.py
# @Time        :2024/10/6 下午10:09
# @Author      :InubashiriLix
"""
the file that manage the data should be single so that the following works can
work better.
"""
import os
import logging
import copy
import numpy as np
import functools
from collections import namedtuple
import csv
import SimpleITK as stik
import torch
from torch.utils.data import Dataset

from utils.util import XyzTuple, xyz2irc
from utils.disk import getCache

from setuptools import glob


# init the logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache("part2ch10_raw")
# the candidateInfo provides a clear way to define the data type
# may be an api principle ??
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)
# it contains the information of the nodule

# save the data to the cache
subset_path = 'G:/LUNA16/luna/subset*/*mhd'


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True) -> list:
    mhd_list = glob.glob(subset_path)
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # concreate the data in the annotations.csv
    diameter_dict = {}
    with open('G:/LUNA16/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            # skip the headers
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            # append the data to the dict if the key does not exist (setdefault)
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    # use the data in candidates.csv to create the full candidate list
    candidateInfo_list = []

    with open('G:/LUNA16/candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # check if the series_uid exists, it should be in the first place
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # then it exists
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            "G:/LUNA16/luna/subset*/{}.mhd".format(series_uid)
        )[0]

        ct_mhd_image = stik.ReadImage(mhd_path)
        ct_a = np.array(stik.GetArrayFromImage(ct_mhd_image), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)
        # print(ct_a.shape)
        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd_image.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd_image.GetSpacing())
        self.direction_a = np.array(ct_mhd_image.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)
        # find the center of the irc
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(round(start_ndx + width_irc[axis]))
            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        # return the ct chunk and the irc center
        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


# TODO: WTF with this decorator???
# @raw_cache.memorize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc



class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None
                 ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        # separate the training and validation set
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info("{!r}: {} {} sample".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training"
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )


if __name__ == '__main__':
    # validation for the getCandidateInfoList,
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
    positiveInfo_list = [x for x in candidateInfo_list if x[0]]
    diameter_list = [x[1] for x in positiveInfo_list]

    # pritn teh test data
    for i in range(0, len(diameter_list), 100):
        print("{:4}  {:4.1f}mm".format(i, diameter_list[i]))

    from vis import findPositiveSamples, showCandidate, plt
    PositiveSample_list = findPositiveSamples()
    showCandidate(PositiveSample_list[11][2])
    plt.show()

