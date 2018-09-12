import numpy as np
import math
from scipy.optimize import minimize


class RunPredictor:

    def __init__(self, statistics_matrix, runs_scored, overs_remaining, wickets_remaining, which_innings):
        self._statistics_matrix = statistics_matrix
        self._runs_scored = runs_scored
        self._wickets_remaining = wickets_remaining
        self._overs_remaining = overs_remaining
        self._which_innings = which_innings

    @staticmethod
    def func_to_minimize_euclidean_distance_for_all_data_points(params, args):
        euclidean_distance = []
        param_L = params[10]
        runs_scored = args[0]
        overs_remaining = args[1]
        wickets_remaining = args[2]
        which_innings = args[3]
        for i in range(len(wickets_remaining)):
            if which_innings[i] == 1:
                run = runs_scored[i]
                over = overs_remaining[i]
                wicket = wickets_remaining[i]
                if run > 0:
                    predicted_run = params[wicket-1] * (1.0 - math.exp(-1*param_L*over/params[wicket-1]))
                    euclidean_distance.append(math.pow(predicted_run - run, 2))
        # print np.sum(euclidean_distance)
        return np.sum(euclidean_distance)

    @staticmethod
    def func_to_minimize_euclidean_distance(params, args):
        euclidean_distance = []
        param_L = params[10]
        statistics_matrix = args[0]
        for i in range(statistics_matrix.shape[0]):
            for j in range(statistics_matrix.shape[1]):
                if statistics_matrix[i][j] > 0:
                    predicted_run = params[j] * (1.0 - math.exp(-1 * param_L * (i + 1) / params[j]))
                    euclidean_distance.append(math.pow(predicted_run - statistics_matrix[i][j], 2))
        # print np.sum(euclidean_distance)
        return np.sum(euclidean_distance)

    def fit_run_prediction_function(self, initial_param):
        constraints = (
            {'type': 'ineq',
             'fun': lambda params: params[1] - params[0]},
            {'type': 'ineq',
             'fun': lambda params: params[2] - params[1]},
            {'type': 'ineq',
             'fun': lambda params: params[3] - params[2]},
            {'type': 'ineq',
             'fun': lambda params: params[4] - params[3]},
            {'type': 'ineq',
             'fun': lambda params: params[5] - params[4]},
            {'type': 'ineq',
             'fun': lambda params: params[6] - params[5]},
            {'type': 'ineq',
             'fun': lambda params: params[7] - params[6]},
            {'type': 'ineq',
             'fun': lambda params: params[8] - params[7]},
            {'type': 'ineq',
             'fun': lambda params: params[9] - params[8]}
        )
        # params = minimize(self.func_to_minimize_euclidean_distance, initial_param,
        #                   args=[self._statistics_matrix], method='L-BFGS-B')

        params = minimize(self.func_to_minimize_euclidean_distance_for_all_data_points, initial_param,
                          args=[self._runs_scored,
                                self._overs_remaining,
                                self._wickets_remaining,
                                self._which_innings],
                          method='L-BFGS-B')
        print params
        return params['x'], params['fun']
