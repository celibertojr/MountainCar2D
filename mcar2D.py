# Q-learning for mountain car problem by Luiz Celiberto
#celibertojr@gmail.com
# follows approach described in Sutton/Singh papers
# based in the work of Sridhar Mahadevan in 1996


import StringIO
import random
from math import *
import time


buf = StringIO.StringIO()


class Learning(object):
    def __init__(self):
    # QL parameters
        self.positivereward = 1 # the value of positive reward.
        self.negativereward = -1 #the value of negative reward.
        self.alpha = 0.2
        self.gamma = 0.9
        self.actions = 3  # action 0,1,2
        self.exploration_rate = 0.1  # percentage of randomness

        self.QLlearning = False

        # training parameters

        self.value_plot_step_size = 2000  # output value function once in N trials

        # Mcar 2d = [pos,vel, action]
        # Mcar definitions
        self.gravity = -0.0025  # // acceleration due to gravity

        self.grid_res = 100
        # car position is limited to the following range
        self.pos_range = [-1.2, 0.60]
        # car velocity is limited to the following range
        self.vel_range = [-0.07, 0.07]
        self.goal = 0.6  # above this value means goal reached
        self.pos_step = (self.pos_range[1] - self.pos_range[0]) / self.grid_res
        self.vel_step = (self.vel_range[1] - self.vel_range[0]) / self.grid_res

        self.simulatev = 0 #initial value
        self.simulatep = 0 #initial value

        self.QL = {}
        self.evol_data = {}



        ##### Car  Basic Functions #####

    def random_pos(self): #random position
        prand = random.uniform(0, 2)
        return self.pos_range[1] - self.pos_range[0] * prand + self.pos_range[0]  # scale position into legal range

    def random_vel(self): #random velocity
        vrand = random.uniform(0, 2)
        return self.vel_range[1] - self.vel_range[0] * vrand + self.vel_range[0]  # scale velocity into legal range




    ########### QL  and MountainCar functions ############################

    def resetQ(self):
        for l1 in range(0, self.grid_res+1):
            for l2 in range(0, self.grid_res+1):
                for l3 in range(0, self.actions):
                    self.QL[(l1, l2, l3)] = random.uniform(0, 1)
                    #self.QL[(l1, l2, l3)] = 0
        print "Reseted"


    # / see if car is up the hill
    def rewards(self,pcar):
        localreward = self.negativereward
        if pcar >= self.goal:
            localreward = self.positivereward
        return localreward


    def update_position_velocity(self,a):
    # action 0,1,2,3 0 backward 1 forward 2 coast
        oldv = self.simulatev  # preserve old values
        oldp = self.simulatep

        if a == 0:  # backward
            aval = -1
        else:
            aval = a  # coast = 0, forward = +1, backward = -1;

        newv = oldv + (0.001 * aval) + (self.gravity * cos(3 * oldp))  # update equation for velocity

        newp = self.simulatep + newv  # update equation for position
        #print newp, newv #debug

        if newv < self.vel_range[0]:  # clip velocity if necessary to keep it within range
            newv = self.vel_range[0]

        if newv > self.vel_range[1]:
            newv = self.vel_range[1]

        if newp < self.pos_range[0]:  # clip position and velocity if necessary to keep it within range
            newp = self.pos_range[0]
            newv = 0  # reduce velocity to 0 if position was out of bounds

        if newp > self.pos_range[1]:
            newp = self.pos_range[1]
            newv = 0

        #print newp, newv
        self.simulatep = newp
        self.simulatev = newv  # update state to new values


    # compute qvalue at any desired state
    def qvalue(self,pos, vel, a):
        pos_in_grid = int(((pos - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0])) * self.grid_res)
        vel_in_grid = int(((vel - self.vel_range[0]) / (self.vel_range[1] - self.vel_range[0])) * self.grid_res)
        # print pos_in_grid, vel_in_grid  #debug
        qv = self.QL[(pos_in_grid, vel_in_grid, a)]
        return qv


    # given a position and velocity bin, pick the highest Q-value action with high probability
    def choose_action(self,pos, vel):
        bact = random.randint(0, 2)
        rvalue = random.uniform(0, 1)
        if rvalue < self.exploration_rate:  # do a random action
            return random.randint(0, 2)
        else:
            for a in range(0, self.actions):
                if self.qvalue(pos, vel, a) > self.qvalue(pos, vel, bact):
                    bact = a
        return bact


    def best_qvalue(self,pos, vel):
        bvalue = self.qvalue(pos, vel, 2)  # 2 coast action
        for a in range(0, self.actions):
            if self.qvalue(pos, vel, a) > bvalue:
                bvalue = self.qvalue(pos, vel, a)

        return bvalue


    # update the Q-Learning algorithm
    # Q(s,a)=Q(s,a)+alpha*(reward + gama*maxQ(s',a') - Q(s,a))
    def QLupdate(self,reward, act, oldp, oldv, newp, newv):
        pos_in_grid = int(((oldp - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0])) * self.grid_res)
        vel_in_grid = int(((oldv - self.vel_range[0]) / (self.vel_range[1] - self.vel_range[0])) * self.grid_res)
        # print pos_in_grid,vel_in_grid #debug
        best_new_qval = self.best_qvalue(newp, newv)
        self.QL[(pos_in_grid, vel_in_grid, act)] = self.QL[(pos_in_grid, vel_in_grid, act)] + (self.alpha) * (reward + self.gamma * best_new_qval-(self.QL[(pos_in_grid, vel_in_grid, act)]))  # Q-Learning update rule


    # update the SARSA algorithm.
    # Q(s,a)=Q(s,a)+alpha*(reward + gama*Q(s',a') - Q(s,a))
    def SARSAupdate(self, reward, act, oldp, oldv, newp, newv):
        pos_in_grid = int(((oldp - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0])) * self.grid_res)
        vel_in_grid = int(((oldv - self.vel_range[0]) / (self.vel_range[1] - self.vel_range[0])) * self.grid_res)

        Newpos_in_grid = int(((newp - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0])) * self.grid_res)
        Newvel_in_grid = int(((newv - self.vel_range[0]) / (self.vel_range[1] - self.vel_range[0])) * self.grid_res)

        New_ac=self.choose_action(newp, newv)
        New_qval=self.qvalue(newp, newv, New_ac)

        self.QL[(pos_in_grid, vel_in_grid, act)] = self.QL[(pos_in_grid, vel_in_grid, act)] + (self.alpha) * (
        reward + self.gamma * New_qval - (self.QL[(pos_in_grid, vel_in_grid, act)]))  # SARSA update rule







    # goal ? if OK return 1
    def reached_goal(self,pos):
        if pos >= self.goal:
            return 1
        else:
            return 0


    def record_evolution(self,run, trial, steps): # record the data
        self.evol_data[(run, trial)] = steps


    ############### FILE ###########################################

    def write_evol_data(self,runs,max_trials): # salve the data
        file = open("evolution.txt", "w")
        mean_quad = 0

        for trial in range(0, max_trials):
            desvio_quad = 0
            sum = 0

            for run in range(0, runs):
                sum += self.evol_data[(run, trial)]

            average = sum / runs

            for run in range(0, runs):
                mean_quad += pow((self.evol_data[(run, trial)] - average), 2)

            mean_quad = mean_quad / runs

            file.write(str(trial))
            file.write(' ')
            file.write(str(average))
            file.write(' ')
            file.write(str(mean_quad))
            file.write('\n')

        file.close()


    def generate_vvalue_plot(self):

        file = open("v-table.txt", "w")

        for pos in range(0, self.grid_res-1):
            pvalue = self.pos_range[0] + self.pos_step * pos
            for vel in range(0, self.grid_res-1):
                vvalue = self.vel_range[0] + self.vel_step * vel
                value = self.best_qvalue(pvalue, vvalue)
                if (value < 0):
                    value = value*(-1)

                file.write(str(value))
                file.write('\t')
            file.write('\n')

        file.close()

##############################################################


    def run_trials(self,runs,max_trials):
        for run in range(0, runs):

            self.resetQ()  # reset Q table
            carV = 0


            for trial in range(0, max_trials):
                iterations = 0
                randP = random.uniform(0, 1)
                initP = (self.pos_range[1] - self.pos_range[0]) * randP + self.pos_range[0]  # scale position into legal range
                self.simulatep = initP
                self.simulatev = carV

                while True:
                    OLDsimulatep = self.simulatep
                    OLDsimulatev = self.simulatev

                    action = self.choose_action(self.simulatep, self.simulatev) #choice action

                    self.update_position_velocity(action)  # move the car
                    r = self.rewards(self.simulatep) #return the reward
                    self.QLupdate(r, action, OLDsimulatep, OLDsimulatev, self.simulatep, self.simulatev)
                    #self.SARSAupdate(r, action, OLDsimulatep, OLDsimulatev, self.simulatep, self.simulatev)
                    iterations += 1
                    mygoal = self.reached_goal(self.simulatep)
                    if mygoal == 1:
                        self.record_evolution(run, trial, iterations)
                        print trial, run, self.simulatev, self.simulatep, r, action #debug
                        break
                    else:
                        print trial, run, self.simulatev, self.simulatep, r, action # debug



#### EXECUTE PROGRAM ###############################
def run():
    runs = 30
    max_trials = 2000

    start = time.time() #start count time

    agent = Learning()
    agent.run_trials(runs,max_trials)
    done = time.time() #finish count time
    agent.write_evol_data(runs,max_trials)
    agent.generate_vvalue_plot()

    elapsed = done - start
    print(elapsed)

    print " Finish ! "




if __name__ == '__main__':
    run()
