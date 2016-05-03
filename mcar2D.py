# Q-learning for mountain car problem by Luiz Celiberto
#celibertojr@gmail.com
# follows approach described in Sutton/Singh papers
# based in the C version of Sridhar Mahadevan in 1996


import random
import StringIO
from math import *

buf = StringIO.StringIO()

#HQL - heuristic ?
heuristic = False

# training parameters
runs = 100
max_trials = 10000
value_plot_step_size = 2000  # output value function once in N trials

# Mcar 2d = [pos,vel, action]
# Mcar definitions
gravity = -0.0025  # // acceleration due to gravity
grid_res = 100
# car position is limited to the following range
pos_range = [-1.2, 0.60]
# car velocity is limited to the following range
vel_range = [-0.07, 0.07]
goal = 0.5  # above this value means goal reached
pos_step = (pos_range[1] - pos_range[0]) / grid_res
vel_step = (vel_range[1] - vel_range[0]) / grid_res

simulatev = 0
simulatep = 0

# QL parameters
epsilon = 0.1
alpha = 0.2
gamma = 0.9
actions = 3  # action 0,1,2,3
reward = -1
exploration_rate = 0.1  # percentage of randomness
beta = 0.1  # learning rate

QL = {}
evol_data = {}

#H parameters
H = {}
mi = 1 # heuristically ponderation
deltaH = 10 # value of H


##### Car  Basic Functions #####
# generate random starting position
def random_pos():
    prand = random.uniform(0, 2)
    return pos_range[1] - pos_range[0] * prand + pos_range[0]  # scale position into legal range


# generate random starting position
def random_vel():
    vrand = random.uniform(0, 2)
    return vel_range[1] - vel_range[0] * vrand + vel_range[0]  # scale velocity into legal range

############# HQL functions ############################

def accelerate():
    posH=pos_range[0]
    velH=vel_range[0]
    for p1 in range(posH, (pos_range[1]-pos_range[0])/grid_res):
        for v1 in range(velH, (vel_range[1] - vel_range[0]) / grid_res):
            if (v1<0):
                H[(p1,v1,2)] = deltaH
            else:
                H[(p1, v1, 1)] = deltaH

def resetH():
    for l1 in range(0, grid_res + 2):
        for l2 in range(0, grid_res + 2):
            for l3 in range(0, actions):
                H[(l1, l2, l3)] = 0


# given a position and velocity bin, pick the highest Q+H-value action with high probability
def choose_action_h(pos, vel):
    bact = random.randint(0, 2)
    rvalue = random.uniform(0, 1)
    if rvalue < exploration_rate:  # do a random action
        return random.randint(0, 2)
    else:
        for a in range(0, actions):
            if qvalue(pos, vel, a) + mi*H[(pos, vel, a)] > qvalue(pos, vel, bact)+mi*H[(pos, vel, a)]:
                bact = a
        return bact


#### QL  functions ############################

def resetQ():
    for l1 in range(0, grid_res + 2):
        for l2 in range(0, grid_res + 2):
            for l3 in range(0, actions):
                QL[(l1, l2, l3)] = random.uniform(0, 1)
    print "Tabela Resetada"


# / see if car is up the hill
def rewards(pcar):
    localreward = 0
    if pcar > goal:
        localreward = 100
    return localreward


def update_position_velocity(a):
    # action 0,1,2,3 0 backward 1 forward 2 coast
    global simulatev
    global simulatep
    global newv
    global newp
    oldv = simulatev  # preserve old values
    oldp = simulatep

    if a == 0:  # backward
        aval = -1
    else:
        aval = a  # coast = 0, forward = +1, backward = -1;

    newv = oldv + (0.001 * aval) + (gravity * cos(3 * oldp))  # update equation for velocity

    newp = simulatep + newv  # update equation for position
    #print newp, newv

    if newv < vel_range[0]:  # clip velocity if necessary to keep it within range
        newv = vel_range[0]

    if newv > vel_range[1]:
        newv = vel_range[1]

    if newp < pos_range[0]:  # clip position and velocity if necessary to keep it within range
        newp = pos_range[0]
        newv = 0  # reduce velocity to 0 if position was out of bounds

    if newp > pos_range[1]:
        newp = pos_range[1]
        newv = 0

    #print newp, newv
    simulatep = newp
    simulatev = newv  # update state to new values


# compute qvalue at any desired state
def qvalue(pos, vel, a):
    pos_in_grid = int(((pos - pos_range[0]) / (pos_range[1] - pos_range[0])) * grid_res)
    vel_in_grid = int(((vel - vel_range[0]) / (vel_range[1] - vel_range[0])) * grid_res)
    # print pos_in_grid, vel_in_grid
    qv = QL[(pos_in_grid, vel_in_grid, a)]
    return qv


# given a position and velocity bin, pick the highest Q-value action with high probability
def choose_action(pos, vel):
    bact = random.randint(0, 2)
    rvalue = random.uniform(0, 1)
    if rvalue < exploration_rate:  # do a random action
        return random.randint(0, 2)
    else:
        for a in range(0, actions):
            if qvalue(pos, vel, a) > qvalue(pos, vel, bact):
                bact = a
        return bact


def best_qvalue(pos, vel):
    bvalue = qvalue(pos, vel, 2)  # 2 coast action
    for a in range(0, actions):
        if qvalue(pos, vel, a) > bvalue:
            bvalue = qvalue(pos, vel, a)

        return bvalue


# update the Q-Learning algorithm
def QLupdate(reward, act, oldp, oldv, newp, newv):
    pos_in_grid = int(((oldp - pos_range[0]) / (pos_range[1] - pos_range[0])) * grid_res)
    vel_in_grid = int(((oldv - vel_range[0]) / (vel_range[1] - vel_range[0])) * grid_res)
    # print pos_in_grid,vel_in_grid
    best_new_qval = best_qvalue(newp, newv)
    QL[(pos_in_grid, vel_in_grid, act)] = (1 - beta) * QL[(pos_in_grid, vel_in_grid, act)] + (beta) * (
        reward + gamma * best_new_qval)  # Q-Learning update rule


# goal ?
def reached_goal(pos):
    if pos > goal:
        return 1
    else:
        return 0


def record_evolution(run, trial, steps):
    evol_data[(run, trial)] = steps


############### FILE ###########################################

def write_evol_data():
    file = open("evolution.txt", "w")

    for trial in range(0, max_trials):
        desvio_quad = 0
        sum = 0

        for run in range(0, runs):
            sum += evol_data[(run, trial)]

        average = sum / runs

        for run in range(0, runs):
            desvio_quad += pow((evol_data[(run, trial)] - average), 2)

        desvio_quad = desvio_quad / runs

        file.write(str(trial))
        file.write(' ')
        file.write(str(average))
        file.write(' ')
        file.write(str(desvio_quad))
        file.write('\n')
    file.close()


def generate_vvalue_plot(l):

    file = open("v-table.txt", "w")

    for pos in range(0, grid_res):
        pvalue = pos_range[0] + pos_step * pos
        for vel in range(0, grid_res):
            vvalue = vel_range[0] + vel_step * vel
            value = best_qvalue(pvalue, vvalue)
            if (value < 0):
                value = -value

            file.write(str(value))
            file.write('"\t"')

        file.close()

##############################################################


def run_trials():
    for run in range(0, runs):
        # contRUN +=1
        global simulatep
        global simulatev
        resetQ()  # reset Q table
        carV = 0
        if heuristic:
            resetH()
            accelerate()



        for trial in range(0, max_trials):
            iterations = 0
            randP = random.uniform(0, 1)
            initP = (pos_range[1] - pos_range[0]) * randP + pos_range[0]  # scale position into legal range
            simulatep = initP
            simulatev = carV

            while True:
                OLDsimulatep = simulatep
                OLDsimulatev = simulatev

                if heuristic: # H value to accelerate
                    action = choose_action_h(simulatep, simulatev)
                else:
                    action = choose_action(simulatep, simulatev)

                update_position_velocity(action)  # move the car
                r = rewards(simulatep)
                QLupdate(r, action, OLDsimulatep, OLDsimulatev, simulatep, simulatev)
                iterations += 1
                mygoal = reached_goal(simulatep)
                if mygoal == 1:
                    record_evolution(run, trial, iterations)
                    print trial, run, simulatev, simulatep, r, action
                    break
                else:
                    print trial, run, simulatev, simulatep, r, action



#### EXECUTE PROGRAM ###############################

run_trials()
write_evol_data()
print " Finish ALL ! "

# print random_pos()
# print random_vel()
# print CARreward(random_pos())
