import time

from ortools.algorithms import pywrapknapsack_solver


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

    return values, [weights], [capacities]

def main():
    # Create the solver.
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

    values, weights, capacities = ReadData("./kplib/07SpannerUncorrelated/n10000/R01000/s000.kp")

    solver.Init(values, weights, capacities)
    t1 = time.time()
    computed_value = solver.Solve()
    t2 = time.time()

    print(t2-t1)

    packed_items = []
    packed_weights = []
    total_weight = 0
    print('Total value =', computed_value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    print('Total weight:', total_weight)
    print('Packed items:', packed_items)
    print('Packed_weights:', packed_weights)


if __name__ == '__main__':
    main()