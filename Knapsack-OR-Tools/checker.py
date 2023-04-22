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
    f = np.zeros([len(weights) + 1, capacities + 1])

    # Dynamic programming
    for i in range(1, len(weights) + 1):
        for j in range(capacities + 1):
            if weights[i - 1] > j:
                f[i][j] = f[i - 1][j]
            else:
                f[i][j] = max(f[i - 1][j], f[i - 1][j - weights[i - 1]] + values[i - 1])

    print(f.max())


def main():
    path_to_file = "./kplib/00Uncorrelated/n00050"
    values, weights, capacities = ReadData(path_to_file + '/R01000/s000.kp')
    dp(values, weights, capacities)


if __name__ == '__main__':
    main()