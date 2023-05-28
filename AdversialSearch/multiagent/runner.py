import glob
import json
import os
import argparse
import itertools

import numpy as np
from pacman import my_trainer

def trainer():
    # define grid of parameters
    param_grid = {
        "w_score": [1],
        "w_junctions": np.arange(start=10, stop=20, step=2),
        "w_capsule": np.arange(start=-50, stop=-10, step=10),
        "w_remaining_food": np.arange(start=-50, stop=0, step=10),
        "w_ghost_edible": np.arange(start=-100, stop=-10, step=10),
        "w_ghost": np.arange(start=100, stop=10, step=-10),
        # "w_food": np.arange(start=-100, stop=-10, step=10),
        "w_food": np.arange(start=-10, stop=-100, step=-10)
    }

    # initialize variables to store the best (combination + score)
    best_params = None
    best_score = -np.inf
    score = 0  # suppress error
    argv = ['-n', '5', '-l', "layout", '-p', "agentType", '-a', 'depth=4,evalFn=myEvaluationFunction',
            '-f', '--isTraining', '--frameTime=0', '-q']
    with open('./log.txt', 'w') as log:
        count = 0
        # iterate through all combinations
        for params in itertools.product(*param_grid.values()):
            # Push current weights for multiAgentSearch
            count = count + 1
            log.write(f"Param set: {count}\n")
            current_weights = {
                "w_score": params[0],
                "w_junctions": int(params[1]),
                "w_capsule": int(params[2]),
                "w_remaining_food": int(params[3]),
                "w_ghost_edible": int(params[4]),
                "w_ghost": int(params[5]),
                "w_food": int(params[6]),
            }
            with open('./weights.json', 'w') as f:
                json.dump(current_weights, f, indent=4)
            
            # calculate score for current combination
            score = 0
            # for source_to_layout in glob.glob("./layouts/*"):
            #     layout = os.path.split(source_to_layout)[1][:-4]
            for layout in ['contestClassic']:
                log.write(f"Layout: {layout}\n")
                print("Layout:", layout, '\n')
                for agentType in ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']:
                    print("Agent:", agentType)
                    argv[3] = layout
                    argv[5] = agentType
                    argv[7] = 'depth=4,evalFn=myEvaluationFunction' if agentType == 'AlphaBetaAgent' \
                        else 'depth=3,evalFn=myEvaluationFunction'
                    score = score + my_trainer(argv)
                    print()

            print('-'*20 + '\n' + '-'*20)
            score = score / 6
            log.write(f"Current score: {score}\n")
            print("Current score:", score)
            # update current score
            if score > best_score:
                best_score = score
                best_params = params
                log.write(f"Best score: {score}")
                print("Current Best score:", score)
                print("-"*20)

                weights = {
                    "w_score": params[0],
                    "w_junctions": int(params[1]),
                    "w_capsule": int(params[2]),
                    "w_remaining_food": int(params[3]),
                    "w_ghost_edible": int(params[4]),
                    "w_ghost": int(params[5]),
                    "w_food": int(params[6]),
                }

                with open('./best_weights.json', 'w') as f:
                    json.dump(weights, f, indent=4)

            log.write('-----------------------------------------\n')
            log.write('-----------------------------------------\n')
            print('-'*20 + '\n' + '-'*20)


def runner():
    for source_to_layout in glob.glob("./layouts/*"):
        layout = os.path.split(source_to_layout)[1][:-4]
        for evaluationFunction in ['scoreEvaluationFunction', 'myEvaluationFunction']:
            for agent in ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']:
                print('Map:', layout)
                print('Evaluation Function:', evaluationFunction)
                print('Search algorithm:', agent)

                depth = 3
                if agent == 'AlphaBetaAgent':
                    depth = 4

                os.system(f"python pacman.py -l {layout} -p {agent} -a depth={depth},evalFn={evaluationFunction} "
                          f"-n 2 -f")
                print('\n' + '-'*20 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runner (trainer) for pacman.py')
    parser.add_argument('--grid-search', action='store_true', help='using grid search to find optimal weights')

    args = parser.parse_args()

    if args.grid_search:
        trainer()
    else:
        runner()
