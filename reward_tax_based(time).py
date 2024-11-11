

from matplotlib import pyplot as plt
import random as rand
import math
import numpy as np
import networkx as nx
from tqdm import trange
import random
from numba import jit


@jit(nopython=True)
def initialization_strategy(I_c, total_attribute):
    for i in range(size):
        temp = rand.random()
        if temp <= I_c:
            total_attribute[0][i] = 0  # cooperator
            total_attribute[1][i] = 0
        elif I_c * 2 >= temp > I_c:
            total_attribute[0][i] = 1  # defector
            total_attribute[1][i] = 1
        else:
            total_attribute[0][i] = 2  # rewarding cooperator
            total_attribute[1][i] = 2


@jit(nopython=True)
def initialization_payoff(total_attribute, neighborsArray, r, reward_amount, tax_rate):
    updatePayoff(total_attribute, neighborsArray, r, reward_amount)
    updateFitness(total_attribute, neighborsArray, tax_rate)


@jit(nopython=True)
def fermiFunction(cost1, cost2):
    K = 0.1
    prob = 1 / (1 + math.exp((cost1 - cost2) / K))
    return prob


@jit(nopython=True)
def updatePayoff(total_attribute, neighborsArray, r, reward_amount):
    for i in range(size):
        total_payoff = 0
        cooperator_num = 0
        rewarder_num = 0
        # Counting the payoffs made by player i within a group with i as the focal player
        for j in neighborsArray[i]:
            if total_attribute[0][j] == 0:
                cooperator_num += 1
            elif total_attribute[0][j] == 2:
                rewarder_num += 1
        if total_attribute[0][i] == 0:
            total_payoff += r * (cooperator_num + rewarder_num + 1) / 5 - 1 + reward_amount * rewarder_num / (
                    cooperator_num + rewarder_num + 1)
        elif total_attribute[0][i] == 1:
            total_payoff += r * (cooperator_num + rewarder_num) / 5
        else:
            total_payoff += r * (
                    cooperator_num + rewarder_num + 1) / 5 - 1 - reward_amount + reward_amount * (1 + rewarder_num) / (
                                    cooperator_num + rewarder_num + 1)

        # counting player i's payoffs within the remaining four groups
        for j in neighborsArray[i]:
            cooperator_num = 0
            rewarder_num = 0
            if total_attribute[0][j] == 0:
                cooperator_num += 1
            elif total_attribute[0][j] == 2:
                rewarder_num += 1

            for k in neighborsArray[j]:
                if k == i:
                    continue
                if total_attribute[0][k] == 0:
                    cooperator_num += 1
                elif total_attribute[0][k] == 2:
                    rewarder_num += 1
            if total_attribute[0][i] == 0:
                total_payoff += r * (cooperator_num + rewarder_num + 1) / 5 - 1 + reward_amount * rewarder_num / (
                        cooperator_num + rewarder_num + 1)
            elif total_attribute[0][i] == 1:
                total_payoff += r * (cooperator_num + rewarder_num) / 5
            else:
                total_payoff += r * (
                        cooperator_num + rewarder_num + 1) / 5 - 1 - reward_amount + reward_amount * rewarder_num / (
                                        cooperator_num + rewarder_num + 1)
        total_attribute[2][i] = total_payoff


@jit(nopython=True)
def updateFitness(total_attribute, neighborsArray, tax_rate):
    for i in range(size):
        # if the payoff is negative, player i pays nothing
        if total_attribute[2][i] <= 0:
            if total_attribute[0][i] == 0 or total_attribute[0][i] == 1:
                total_attribute[3][i] = total_attribute[2][i]
            else:
                # if player i is a rewarding cooperator, it will get a tax incentive
                receive_tax = calculateTax(i, total_attribute, neighborsArray, tax_rate)
                # update fitness
                total_attribute[3][i] = total_attribute[2][i] + receive_tax

        else:
            # if player i has a positive payoff, it pays a proportional tax of p
            if total_attribute[0][i] == 0 or total_attribute[0][i] == 1:
                total_attribute[3][i] = total_attribute[2][i] * (1 - tax_rate)
                # if there are no rewarding cooperators within the group of the tax-paying cooperators and defectors,
                # the surplus tax is calculated
                calculateRemainTax(i, total_attribute, neighborsArray, tax_rate)
            else:
                # if player i is a rewarding cooperator, it will get a tax incentive
                receive_tax = calculateTax(i, total_attribute, neighborsArray, tax_rate)
                total_attribute[3][i] = total_attribute[2][i] * (1 - tax_rate) + receive_tax

    # distribution of surplus tax to all rewarding cooperators
    rewarder_num = 0
    global_tax = sum(total_attribute[4])
    for i in range(size):
        if total_attribute[0][i] == 2:
            rewarder_num += 1
    for i in range(size):
        if total_attribute[0][i] == 2:
            total_attribute[3][i] += global_tax / rewarder_num


@jit(nopython=True)
def calculateRemainTax(i, total_attribute, neighborsArray, tax_rate):
    # judge whether there is a rewarding cooperator in the group with player i as the focal player
    # count the surplus tax if there is no rewarding cooperator
    rewarder_num = 0
    for j in neighborsArray[i]:
        if total_attribute[0][j] == 2:
            rewarder_num += 1
    if rewarder_num == 0:
        total_attribute[4][i] += total_attribute[2][i] * tax_rate / 5
    # counting surplus tax within the remaining four groups
    for j in neighborsArray[i]:
        if total_attribute[0][j] == 2:
            continue
        rewarder_num = 0
        for k in neighborsArray[j]:
            if total_attribute[0][k] == 2:
                rewarder_num += 1
        if rewarder_num == 0:
            total_attribute[4][i] += total_attribute[2][i] * tax_rate / 5


@jit(nopython=True)
def calculateTax(i, total_attribute, neighborsArray, tax_rate):
    receive_tax = 0  # total amount of tax distributed to i
    rewarder_num = 0

    # counting the tax revenue that i receives within a group with i as the focal player
    if total_attribute[2][i] > 0:
        total_tax = total_attribute[2][i] * tax_rate / 5
    else:
        total_tax = 0
    for j in neighborsArray[i]:
        if total_attribute[2][j] > 0:
            total_tax += total_attribute[2][j] * tax_rate / 5
        if total_attribute[0][j] == 2:
            rewarder_num += 1
    receive_tax += total_tax / (rewarder_num + 1)

    # calculate the tax that i receives within the remaining four groups
    for j in neighborsArray[i]:
        if total_attribute[2][j] > 0:
            total_tax = total_attribute[2][j] * tax_rate / 5
        else:
            total_tax = 0
        if total_attribute[0][j] == 2:
            rewarder_num = 1
        else:
            rewarder_num = 0
        for k in neighborsArray[j]:
            if total_attribute[0][k] == 2 and k != i:
                rewarder_num += 1
            if total_attribute[2][k] > 0:
                total_tax += total_attribute[2][k] * tax_rate / 5
        receive_tax += total_tax / (rewarder_num + 1)

    return receive_tax


@jit(nopython=True)
def number_of_cooperation(total_attribute):
    number = 0
    for i in range(size):
        if total_attribute[0][i] == 0:
            number += 1
    return number


@jit(nopython=True)
def number_of_defect(total_attribute):
    number = 0
    for i in range(size):
        if total_attribute[0][i] == 1:
            number += 1
    return number


@jit(nopython=True)
def number_of_rewarder(total_attribute):
    number = 0
    for i in range(size):
        if total_attribute[0][i] == 2:
            number += 1
    return number


@jit(nopython=True)
def strategyUpdate(total_attribute, neighborsArray):
    # synchronous update
    for i in range(size):
        # randomly select a neighbor
        random_node = random.randrange(4)
        imitation_node = neighborsArray[i][random_node]
        # calculating probability using the Fermi equation
        prob = fermiFunction(total_attribute[3][i], total_attribute[3][imitation_node])
        if random.random() <= prob:
            total_attribute[1][i] = total_attribute[0][imitation_node]  # imitate neighbor's strategy
        else:
            total_attribute[1][i] = total_attribute[0][i]
    for i in range(size):
        total_attribute[0][i] = total_attribute[1][i]


@jit(nopython=True)
def monte_carlo_simulation(total_attribute, I_c, cooperation, defect, rewarder, r, neighborsArray, reward_amount,
                           tax_rate):
    initialization_strategy(I_c, total_attribute)
    for epoch in range(max_epoch):
        cooperation[epoch] += number_of_cooperation(total_attribute) / size / average
        defect[epoch] += number_of_defect(total_attribute) / size / average
        rewarder[epoch] += number_of_rewarder(total_attribute) / size / average

        initialization_payoff(total_attribute, neighborsArray, r, reward_amount, tax_rate)
        total_attribute[4, :] = 0

        strategyUpdate(total_attribute, neighborsArray)

    return cooperation, defect, rewarder


def is_plot_cooperation_ratio(cooperation, defect, rewarder):
    plt.figure()
    plt.ylim(0, 1.05)
    plt.semilogx(range(1, 1 + len(cooperation)), cooperation, label='C')
    plt.semilogx(range(1, 1 + len(defect)), defect, label='D')
    plt.semilogx(range(1, 1 + len(rewarder)), rewarder, label='R')
    plt.ylabel('proportion')
    plt.xlabel('time step')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    # parameter
    I_c = 1 / 3  # three strategies evenly distributed
    max_epoch = 10000  # maximum step size
    r = 4  # synergy factor
    reward_amount = 0.2  # reward cost
    tax_rate = 0.06  # tax rate parameter
    accumulation = 100
    average = 1
    L = 200
    size = pow(L, 2)

    # initialization
    # 0:strategy, 1:temp strategy, 2:payoff, 3:fitness, 4:remain tax
    total_attribute = np.zeros((5, size), dtype=np.float32)
    # lattice network
    G = nx.grid_2d_graph(int(math.sqrt(size)), int(math.sqrt(size)), periodic=True)
    neighborsList = []
    neighborsArray = []
    for i in G.nodes():
        temp = []
        for j in list(G.adj[i]):
            temp.append(j[0] * L + j[1])
        neighborsList.append(temp)
        neighborsArray.append(temp)
    degreeArray = np.ones(size, dtype=int) * 4
    neighborsArray = np.asarray(neighborsArray, dtype='int32')

    # results
    cooperation = np.zeros(max_epoch, dtype=np.float32)
    defect = np.zeros(max_epoch, dtype=np.float32)
    rewarder = np.zeros(max_epoch, dtype=np.float32)

    for avg in trange(average):
        monte_carlo_simulation(total_attribute, I_c, cooperation, defect, rewarder, r, neighborsArray, reward_amount,
                               tax_rate)

    # visualization
    is_plot_cooperation_ratio(cooperation, defect, rewarder)
