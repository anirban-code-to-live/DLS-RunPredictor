import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import RunPredictor as rp


def evaluate_run_prediction_func(z, l, u):
    return z * (1 - np.exp(-l*u/z))


if __name__ == '__main__':
    cricket_data = pd.read_csv('../data/04_cricket_1999to2011.csv')
    print cricket_data.info()
    winning_team = cricket_data['Winning.Team'].values
    batting_team = cricket_data['At.Bat'].values
    which_innings = cricket_data['Innings'].values
    print which_innings.tolist().count(1)
    current_score = cricket_data['Total.Runs'].values
    innings_total_runs = cricket_data['Innings.Total.Runs'].values
    runs_remaining = innings_total_runs - current_score
    total_overs = cricket_data['Total.Overs'].values
    overs_done = cricket_data['Over'].values
    overs_remaining = total_overs - overs_done
    wickets_remaining = cricket_data['Wickets.in.Hand'].values
    run_wicket_over_mat = np.zeros((50, 11), dtype=float)
    count_wicket_over_mat = np.zeros((50, 11), dtype=float)
    for i in range(len(overs_remaining)):
        if which_innings[i] == 1:
            wicket = wickets_remaining[i]
            over = overs_remaining[i]
            run = runs_remaining[i]
            run_wicket_over_mat[over - 1][wicket - 1] += run
            count_wicket_over_mat[over - 1][wicket - 1] += 1
            run_wicket_over_mat[49][9] += innings_total_runs[i]
            count_wicket_over_mat[49][9] += 1

    statistics_matrix = np.divide(run_wicket_over_mat, count_wicket_over_mat,
                                  out=np.zeros_like(run_wicket_over_mat), where=count_wicket_over_mat != 0)

    run_predictor = rp.RunPredictor(statistics_matrix, runs_remaining, overs_remaining, wickets_remaining, which_innings)
    initial_params = [10.0, 40.0, 80.0, 120.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 3]
    optimized_params, total_loss = run_predictor.fit_run_prediction_function(initial_params)

    print 'Total Loss :: ' + str(total_loss)
    print 'Parameter L :: ' + str(optimized_params[10])
    for i in range(10):
        print 'Parameter Z' + str(i+1) + ' :: ' + str(optimized_params[i])

    # Plot the resource vs overs used graphs for 10 parameters
    plt.figure(1)
    plt.xlim((0, 50))
    plt.ylim((0, 1))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs remaining')
    plt.ylabel('Percentage of resource remaining')
    max_resource_list = evaluate_run_prediction_func(optimized_params[9], optimized_params[10], 50)
    x = np.arange(0, 51, 1)
    modified_x = np.array([50.0 - i for i in x])
    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#234a21', '#876e34', '#a21342']
    for i in range(10):
        y = 100*evaluate_run_prediction_func(optimized_params[i], optimized_params[10], modified_x)/max_resource_list
        plt.plot(x, y, c=color_list[i], label='Z['+str(i+1)+']')
        plt.legend()

    y_linear = [-2*i + 100 for i in x]
    plt.plot(x, y_linear, '#5631a2')
    plt.savefig('run_prediction_functions.pdf')

