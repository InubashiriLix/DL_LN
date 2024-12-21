#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :selftest.py
# @Time        :2024/10/6 下午11:16
# @Author      :InubashiriLix

import csv
import os
from collections import namedtuple
import glob
import functools

subset_path = 'G:/LUNA16/luna/subset*/*.mhd'
annotation_path = 'G:/LUNA16/annotations.csv'
candidates_path = 'G:/LUNA16/candidates.csv'

CandidatesInfoTuple = namedtuple(
    'CandidatesInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)


@functools.lru_cache(1)
def getCandidateToList(on_disk_bool = True):
    # get the list of the data into the cache
    mhd_list = glob.glob(subset_path)
    presentOnDisk_set = {os.path.split(p)[:-4][0] for p in mhd_list}

    # analyze the annotation first
    annotationDict = {}
    with open(annotation_path, 'r') as f:
        # skip the header
        for row in list(csv.reader(f))[1:]:
            annotationSeriesId = row[0]
            annotationCoordxyz = tuple([float(i) for i in row[1:4]])
            annotationDiameter = float(row[4])

            annotationDict.setdefault(annotationSeriesId, []).append(
                (annotationCoordxyz, annotationDiameter))

    # analyze the candidates then
    candidateList = []  # final output
    with open(candidates_path, 'r') as f:
        # skip the headers, too
        for row in list(csv.reader(f))[1:]:
            # check if the data has been in the disk
            if candidateSeriesId not in presentOnDisk_set and on_disk_bool:
                continue

            candidateSeriesId = row[0]
            candidateCoordxyz = tuple(float(i) for i in row[1: 4])
            candidateHasNodule_bool = bool(int(row[4]))

            # check and correct the bondaries
            for annotationTup in annotationDict.get(candidateSeriesId):
                annotationTupCoordxyz, annotationTupDiameter = annotationTup
                for i in range(3):
                    if abs(annotationTupCoordxyz[i] - candidateCoordxyz[i]) > (annotationTupDiameter / 4):
                        break
                else:
                    candidateCoordxyz = annotationTupCoordxyz

                # the data has been corrected now
                candidateList.append(CandidatesInfoTuple(
                    candidateHasNodule_bool,
                    annotationTupDiameter,
                    candidateSeriesId,
                    candidateCoordxyz
                ))

    candidateList = candidateList.sort(reverse=True)
    return candidateList

