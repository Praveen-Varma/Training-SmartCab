import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import pickle
import csv
class QAgent(object):
    def __init__(self, alpha = 0.2, epsilon = 0.05, gamma = 0.4):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.QTable = dict()
    def make_decision(self, state):
        for a in Environment.valid_actions:
            if self.QTable.get((state, a)) == None:
                self.QTable[(state, a)] = 0
        q = [self.QTable.get((state, a)) for a in Environment.valid_actions]
        if random.random() < self.epsilon:
            action = random.choice(Environment.valid_actions)
        else:
            max_q = max(q)
            if q.count(max_q) > 1:
                best_actions = [i for i in range(len(Environment.valid_actions)) if q[i] == max_q]                       
                action_idx = random.choice(best_actions)
            else:
                action_idx = q.index(max_q)
            action = Environment.valid_actions[action_idx]
        return action
    def set_params(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
    def learn(self, state, action, next_state, reward):
        q = [self.QTable.get((next_state, a)) for a in Environment.valid_actions]
        future_reward = max(q)         
        if future_reward is None:
            future_reward = 0.0
        q = self.QTable.get((state, action))
        if q is None:
            q = reward
        else:
            q = (1 - self.alpha) * q +  self.alpha*(reward + self.gamma*future_reward)
        self.QTable[(state, action)] = q
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here
        self.agent = QAgent()
        self.index = 0
        self.total_reward = 0
        self.total_moves = 0
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
##        for state, value in self.agent.QTable.iteritems():
##            writer.writerow([self.index, str(state[0].light), str(state[0].oncoming), str(state[0].left), str(state[0].next_waypoint), str(state[1]), value])
        self.state = None
        self.next_waypoint = None
        self.index += 1
    def make_state(self, state, waypoint):
        return State(light = state['light'], oncoming=state['oncoming'], left = state['left'],  next_waypoint = waypoint)
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()
        deadline = self.env.get_deadline(self)
        inputs = self.env.sense(self)
        # TODO: Update state
        self.state = self.make_state(inputs, self.next_waypoint)
        #print "The state is light {} oncoming {} left {} next_waypoint {}".format(inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        # TODO: Select action according to your policy
        action = self.agent.make_decision(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        self.total_moves += 1
        # TODO: Learn policy based on state, action, reward
        next_state = self.make_state(self.env.sense(self), self.planner.next_waypoint())
        self.agent.learn(self.state, action, next_state, reward)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
def run(alpha, gamma, epsilon):
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.agent.set_params(alpha, gamma, epsilon)
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    # Now simulate it
    sim = Simulator(e, update_delay= 0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    sim.run(n_trials=1000)  # run for a specified number of trials
    writer.writerow([alpha, gamma, epsilon, a.total_reward, a.total_moves])
    with open('final_Q.csv', 'wb') as q_file:
        writer_q = csv.writer(q_file, delimiter=';')
        for state, value in a.agent.QTable.iteritems():
            print state
            writer_q.writerow([str(state[0][0]), str(state[0][1]), str(state[0][2]), str(state[0][3]), str(state[1]), value])
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
if __name__ == '__main__':
    csvfile = open('grid_search.csv', 'wb')
    writer = csv.writer(csvfile, delimiter=';')
    State = namedtuple('State', ['light', 'oncoming', 'left', 'next_waypoint'])
    run(0.2, 0.4, 0.05)
    csvfile.close()
