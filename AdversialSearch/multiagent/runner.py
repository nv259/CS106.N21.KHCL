import glob
import os
import argparse
import itertools

import fontTools.t1Lib
import numpy as np


def trainer():
    # define grid of parameters
    param_grid = {}

    # initialize variables to store the best (combination + score)
    best_params = None
    best_score = -np.inf
    score = 0  # suppress error

    # iterate through all combinations
    for params in itertools.product(*param_grid.values()):
        # calculate score for current combination

        # update current score
        if score > best_score:
            best_score = score
            best_params = params

    with open('weights.txt', 'a') as f:
        f.write(f'Weights: {best_params}\nEstimate score: {best_score}')


def runner():
    for source_to_layout in glob.glob("./layouts/*"):
        layout = os.path.split(source_to_layout)[1][:-4]
        for evaluationFunction in ['scoreEvaluationFunction', 'betterEvaluationFunction', 'myEvaluationFunction']:
            for agent in ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']:
                print('Map:', layout)
                print('Evaluation Function:', evaluationFunction)
                print('Search algorithm:', agent)

                depth = 3
                if agent == 'AlphaBetaAgent':
                    depth = 4

                os.system(f"python pacman.py -l {layout} -p {agent} -a depth={depth},evalFn={evaluationFunction} "
                          f"-n 10 -q -t -f")
                print('\n' + '-'*20 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runner (trainer) for pacman.py')
    parser.add_argument('--grid_search', action='store_true', help='using grid search to find optimal weights')

    args = parser.parse_args()

    if args.grid_search:
        trainer()
    else:
        runner()
