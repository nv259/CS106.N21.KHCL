import numpy as np


def ReadData(path_to_file):
    values = []
    weights = []

    with open(path_to_file, 'r') as f:
        all_lines = f.read().splitlines()
        # Remove blank lines
        all_lines = [line for line in all_lines if line]

        num_items = int(all_lines[0])
        capacities = int(all_lines[1])

        for i in range(num_items):
            # item's index is stored from 2
            item = all_lines[i + 2].split()
            value, weight = int(item[0]), int(item[1])

            values.append(value)
            weights.append(weight)

    return values, weights, capacities

def dp(values, weights, capacities):
    # Init
    f = np.zeros(capacities + 1)
    for i, weight in enumerate(weights):
        f[weight] = values[i]

    # Dynamic programming
    for i in range(capacities):
        for index, weight in enumerate(weights):
            if weight > i:
                continue
            f[i] = max(f[i], f[i - weight] + values[index])

    print(f.max())

def main():
    path_to_file = "./kplib/00Uncorrelated/n00050"
    values, weights, capacities = ReadData(path_to_file + '/R01000/s000.kp')
    dp(values, weights, capacities)


if __name__ == '__main__':
    main()