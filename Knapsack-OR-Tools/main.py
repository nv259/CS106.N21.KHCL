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
    test_cases = ['n00050', 'n00100', 'n00200', 'n00500', 'n01000', 'n02000', 'n05000', 'n10000']

    # Declare the solver
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    solver.set_time_limit(180)

    with open('result.txt', 'a') as f:
        f.write(f'Group: {path_to_files}\n')

        for test_case in test_cases:
            f.write(f'----------Test case: {test_case}------------\n')
            print(f'----------Test case: {test_case}------------\n')

            # Create the data
            values, weights, capacities = ReadData(path_to_files + test_case + '/R01000/s000.kp')

            # Call the solver
            t1 = time.time()
            solver.Init(values, weights, capacities)
            computed_value = solver.Solve()
            t2 = time.time()

            packed_items = []
            packed_weights = []
            total_weight = 0
            f.write(f'Total value = {computed_value}\n')

            for i in range(len(values)):
                if solver.BestSolutionContains(i):
                    packed_items.append(i)
                    packed_weights.append(weights[0][i])
                    total_weight += weights[0][i]

            f.write(f'Total weight: {total_weight}\n')
            f.write(f'Packed items: {packed_items}\n')
            f.write(f'Packed_weights: {packed_weights}\n')
            f.write(f'Run time:{t2-t1}\n')
            f.write('--------------------------------------------\n')
        f.write('\n\n')

if __name__ == '__main__':
    # Path to test folder
    path_to_files = './kplib/12Circle/'
    main()
