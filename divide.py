import random
import json

import numpy as np
import pandas as pd

# TODO
AHP_B_WEIGHTS = [.2, .3, .5]
AHP_C_WEIGHTS = [.1, .1, .1, .1, .1, .1, .4]

C1_DELIM = [182.5, 56.5, 21.5, 1.5, 0.5]
C2_DELIM = [0.5, 28.5, 98.5, 182.5, 364.5]
C3_DELIM = [5.5, 3.5, 2.5, 1.5, 0.5]
C4_DELIM = [40000., 20000., 10000., 4000., 2000.]
C5_DELIM = [10.5, 8.5, 6.5, 4.5, 2.5]
C6_DELIM = [7., 12., 18., 24., 40.]
C7_DELIM = [2.5, 4.5, 6.5, 9.5, 12.5]

# [low risk, medium risk, high risk]
LABEL_ID = [0, 1, 2]
LABEL_PROPORTION = [.3, .4, .3]

# [train set, test set]
SET_ID = [0, 1]
SET_PROPORTION = [.35, .65]

DATA_PATH = 'data/data.xlsx'
TRAIN_PATH = 'data/train.json'
TEST_PATH = 'data/test.json'
USECOLS = [
    '最长故障天数C1',
    '未反馈故障天数C2',
    '故障设施种类C3',
    '服务面积C4',
    '服务范围C5',
    '执业总时长C6',
    '执业总人次C7'
]


data = pd.read_excel(DATA_PATH, usecols=USECOLS)

labels = np.array(random.choices(population=LABEL_ID, weights=LABEL_PROPORTION, k=data.__len__()))

low_indices = np.where(labels == 0)[0]
low_set = random.choices(population=SET_ID, weights=SET_PROPORTION, k=len(low_indices))

medium_indices = np.where(labels == 1)[0]
medium_set = random.choices(population=SET_ID, weights=SET_PROPORTION, k=len(medium_indices))

high_indices = np.where(labels == 2)[0]
high_set = random.choices(population=SET_ID, weights=SET_PROPORTION, k=len(high_indices))

sets = np.zeros_like(labels)

for i, index in enumerate(low_indices):
    sets[index] = low_set[i]
for i, index in enumerate(medium_indices):
    sets[index] = medium_set[i]
for i, index in enumerate(high_indices):
    sets[index] = high_set[i]

train_set = []
test_set = []

for i, row in data.iterrows():

    entry = {'id': i, 'label': int(labels[i])}

    scores = []

    c1 = row[USECOLS[0]]
    score1 = \
        0 if c1 > C1_DELIM[0] else (
            1 if c1 > C1_DELIM[1] else (
                2 if c1 > C1_DELIM[2] else (
                    3 if c1 > C1_DELIM[3] else (
                        4 if c1 > C1_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score1 *= AHP_B_WEIGHTS[0] * AHP_C_WEIGHTS[0]
    scores.append(score1)

    c2 = row[USECOLS[1]]
    score2 = \
        0 if c2 < C2_DELIM[0] else (
            1 if c2 < C2_DELIM[1] else (
                2 if c2 < C2_DELIM[2] else (
                    3 if c2 < C2_DELIM[3] else (
                        4 if c2 < C2_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score2 *= AHP_B_WEIGHTS[0] * AHP_C_WEIGHTS[1]
    scores.append(score2)

    c3 = row[USECOLS[2]]
    score3 = \
        0 if c3 > C3_DELIM[0] else (
            1 if c3 > C3_DELIM[1] else (
                2 if c3 > C3_DELIM[2] else (
                    3 if c3 > C3_DELIM[3] else (
                        4 if c3 > C3_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score3 *= AHP_B_WEIGHTS[0] * AHP_C_WEIGHTS[2]
    scores.append(score3)

    c4 = row[USECOLS[3]]
    score4 = \
        0 if c4 > C4_DELIM[0] else (
            1 if c4 > C4_DELIM[1] else (
                2 if c4 > C4_DELIM[2] else (
                    3 if c4 > C4_DELIM[3] else (
                        4 if c4 > C4_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score4 *= AHP_B_WEIGHTS[1] * AHP_C_WEIGHTS[3]
    scores.append(score4)

    c5 = row[USECOLS[4]]
    score5 = \
        0 if c5 > C5_DELIM[0] else (
            1 if c5 > C5_DELIM[1] else (
                2 if c5 > C5_DELIM[2] else (
                    3 if c5 > C5_DELIM[3] else (
                        4 if c5 > C5_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score5 *= AHP_B_WEIGHTS[1] * AHP_C_WEIGHTS[4]
    scores.append(score5)

    c6 = row[USECOLS[5]]
    score6 = \
        0 if c6 < C6_DELIM[0] else (
            1 if c6 < C6_DELIM[1] else (
                2 if c6 < C6_DELIM[2] else (
                    3 if c6 < C6_DELIM[3] else (
                        4 if c6 < C6_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score6 *= AHP_B_WEIGHTS[2] * AHP_C_WEIGHTS[5]
    scores.append(score6)

    c7 = row[USECOLS[6]]
    score7 = \
        0 if c7 < C7_DELIM[0] else (
            1 if c7 < C7_DELIM[1] else (
                2 if c7 < C7_DELIM[2] else (
                    3 if c7 < C7_DELIM[3] else (
                        4 if c7 < C7_DELIM[4] else (
                            5
                        )
                    )
                )
            )
        )
    score7 *= AHP_B_WEIGHTS[2] * AHP_C_WEIGHTS[6]
    scores.append(score7)

    entry['scores'] = scores

    if sets[i] == 0:
        train_set.append(entry)
    else:
        test_set.append(entry)

with open(TRAIN_PATH, 'w') as f:
    train_json = json.dumps(train_set, indent=4)
    f.write(train_json)
    f.close()

with open(TEST_PATH, 'w') as f:
    test_json = json.dumps(test_set, indent=4)
    f.write(test_json)
    f.close()

