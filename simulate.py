import random

import json

from tqdm import tqdm

from divide import LABEL_ID, LABEL_PROPORTION, AHP_B_WEIGHTS, AHP_C_WEIGHTS

NUM = 10000

sim_data = []

for i in tqdm(range(NUM)):

    entry = {'id': i}

    label = [0. for _ in range(len(LABEL_ID))]
    risk = random.choices(population=LABEL_ID, weights=LABEL_PROPORTION, k=1)
    label[risk[0]] = 1.
    entry['label'] = label

    scores = random.choices(population=[0, 1, 2, 3, 4, 5], k=7)
    scores[0] *= AHP_B_WEIGHTS[0] * AHP_C_WEIGHTS[0]
    scores[1] *= AHP_B_WEIGHTS[0] * AHP_C_WEIGHTS[1]
    scores[2] *= AHP_B_WEIGHTS[0] * AHP_C_WEIGHTS[2]
    scores[3] *= AHP_B_WEIGHTS[1] * AHP_C_WEIGHTS[3]
    scores[4] *= AHP_B_WEIGHTS[1] * AHP_C_WEIGHTS[4]
    scores[5] *= AHP_B_WEIGHTS[2] * AHP_C_WEIGHTS[5]
    scores[6] *= AHP_B_WEIGHTS[2] * AHP_C_WEIGHTS[6]
    entry['scores'] = scores

    sim_data.append(entry)

with open('data/sim_train.json', 'w') as f:
    train_json = json.dumps(sim_data, indent=4)
    f.write(train_json)
    f.close()
