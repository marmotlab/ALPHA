from py2cpp import pyCallCpp
import copy
import math
import random
import sys


# import gym
import numpy as np
from gym import spaces
# from gym.envs.classic_control import rendering
from matplotlib.colors import hsv_to_rgb

import alg_parameters
from map_generator import *
# from util import normalization_feature
from alg_parameters import *
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import NoSolutionError

opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1),
           8: (-1, 1)}  # x,y operation for corresponding action
# -{0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST}
actionDict = {v: k for k, v in dirDict.items()}


class State(object):
    """ map the environment as 2 2d numpy arrays """

    def __init__(self, world0, goals, num_agents, nodes_obs, num_nodes):
        """initialization"""
        self.state = world0.copy()  # static obstacle: -1,empty: 0,agent = positive integer (agent_id)
        self.goals = goals.copy()  # empty: 0, goal = positive integer (corresponding to agent_id)
        self.num_agents = num_agents
        self.nodes_obs = nodes_obs
        self.num_nodes = num_nodes + 2
        self.agents, self.agent_goals = self.scan_for_agents()  # position of agents, and position of goals

        assert (len(self.agents) == num_agents)

    def scan_for_agents(self):
        """find the position of agents and goals"""
        agents = [(-1, -1) for _ in range(self.num_agents)]
        agent_goals = [(-1, -1) for _ in range(self.num_agents)]

        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):  # check every position in the environment
                if self.state[i, j] > 0:  # agent
                    agents[self.state[i, j] - 1] = (i, j)
                if self.goals[i, j] > 0:  # goal
                    agent_goals[self.goals[i, j] - 1] = (i, j)
        assert ((-1, -1) not in agents and (-1, -1) not in agent_goals)
        return agents, agent_goals

    def get_pos(self, agent_id):
        """agent's current position"""
        return self.agents[agent_id - 1]

    def get_goal(self, agent_id):
        """the position of agent's goal"""
        return self.agent_goals[agent_id - 1]

    def find_swap(self, curr_position, past_position, actions):
        """check if there is a swap collision"""
        swap_index = []
        for i in range(self.num_agents):
            if actions[i] == 0:  # stay can not cause swap error
                continue
            else:
                ax = curr_position[i][0]
                ay = curr_position[i][1]
                agent_index = [index for (index, value) in enumerate(past_position) if value == (ax, ay)]
                for item in agent_index:
                    if i != item and curr_position[item] == past_position[i]:
                        swap_index.append([i, item])
        return swap_index

    def joint_move(self, actions):
        """simultaneously move agents and checks for collisions on the joint action """
        imag_state = (self.state > 0).astype(int)  # map of world 0-no agent, 1- have agent
        past_position = copy.deepcopy(self.agents)  # the position of agents before moving
        curr_position = copy.deepcopy(self.agents)  # the current position of agents after moving
        agent_status = np.zeros(self.num_agents)  # use to determine rewards and invalid actions

        # imagine moving
        for i in range(self.num_agents):
            direction = self.get_dir(actions[i])
            ax = self.agents[i][0]
            ay = self.agents[i][1]  # current position

            # Not moving is always allowed
            if direction == (0, 0):
                continue

            # Otherwise, let's look at the validity of the move
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.state.shape[0] or ax + dx < 0 or ay + dy >= self.state.shape[1] or ay + dy < 0:
                # out of boundaries
                agent_status[i] = -1
                continue

            if self.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                agent_status[i] = -2
                continue

            imag_state[ax, ay] -= 1  # set the previous position to empty
            imag_state[ax + dx, ay + dy] += 1  # move to the new position
            curr_position[i] = (ax + dx, ay + dy)  # update agent's current position

        # solve collision between agents
        swap_index = self.find_swap(curr_position, past_position, actions)  # search for swapping collision
        collide_poss = np.argwhere(imag_state > 1)  # search for vertex collision
        while len(swap_index) > 0 or len(collide_poss) > 0:
            while len(collide_poss) > 0:
                agent_index = [index for (index, value) in enumerate(curr_position) if
                               all(value == collide_poss[0])]  # solve collisions one by one
                for i in agent_index:  # stop at it previous position
                    imag_state[curr_position[i]] -= 1
                    imag_state[past_position[i]] += 1
                    curr_position[i] = past_position[i]
                    agent_status[i] = -3

                collide_poss = np.argwhere(imag_state > 1)  # recheck

            swap_index = self.find_swap(curr_position, past_position, actions)

            while len(swap_index) > 0:
                couple = swap_index[0]  # solve collision one by one
                for i in couple:
                    imag_state[curr_position[i]] -= 1
                    imag_state[past_position[i]] += 1
                    curr_position[i] = past_position[i]
                    agent_status[i] = -3

                swap_index = self.find_swap(curr_position, past_position, actions)  # recheck

            collide_poss = np.argwhere(imag_state > 1)  # recheck

        assert len(np.argwhere(imag_state < 0)) == 0

        # Ture moving
        for i in range(self.num_agents):
            direction = self.get_dir(actions[i])
            # execute valid action
            if agent_status[i] == 0:
                dx, dy = direction[0], direction[1]
                ax = self.agents[i][0]
                ay = self.agents[i][1]
                self.state[ax, ay] = 0  # clean previous position
                self.agents[i] = (ax + dx, ay + dy)  # update agent's current position
                if self.goals[ax + dx, ay + dy] == i + 1:
                    agent_status[i] = 1  # reach goal
                    continue
                elif self.goals[ax + dx, ay + dy] != i + 1 and self.goals[ax, ay] == i + 1:
                    agent_status[i] = 2
                    continue  # on goal in last step and leave goal now
                else:
                    agent_status[i] = 0  # nothing happen

        for i in range(self.num_agents):
            self.state[self.agents[i]] = i + 1  # move to new position
        return agent_status

    def get_dir(self, action):
        """obtain corresponding x,y operation based on action"""
        return dirDict[action]

    def get_action(self, direction):
        """obtain corresponding action based on x,y operation"""
        return actionDict[direction]

    def task_done(self):
        """check if all agents on their goal"""
        num_complete = 0
        for i in range(1, len(self.agents) + 1):
            agent_pos = self.agents[i - 1]
            if self.goals[agent_pos[0], agent_pos[1]] == i:
                num_complete += 1
        return num_complete == len(self.agents), num_complete


# class MAPFEnv(gym.Env):
class MAPFEnv():
    """map MAPF problems to a standard RL environment"""

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, num_agents=EnvParameters.N_AGENTS, size=EnvParameters.WORLD_SIZE,
                 prob=EnvParameters.OBSTACLE_PROB):
        """initialization"""
        self.num_agents = num_agents
        self.observation_size = EnvParameters.FOV_SIZE
        self.SIZE = size  # size of a side of the square grid
        self.PROB = prob  # obstacle density
        self.max_on_goal = 0
        assert len(self.SIZE) == 2
        assert len(self.PROB) == 2

        self.set_world()
        self.all_dist_map = self.get_all_dist_map()
        
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(EnvParameters.N_ACTIONS)])
        self.viewer = None

    def is_connected(self, world0):
        """check if each agent's start position and goal position are sampled from the same connected region"""
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def first_free(world):
            for x in range(world.shape[0]):
                for y in range(world.shape[1]):
                    if world[x, y] == 0:
                        return x, y

        def flood_fill(world, k, g):
            sx, sy = world.shape[0], world.shape[1]
            if k < 0 or k >= sx or g < 0 or g >= sy:  # out of boundaries
                return
            if world[k, g] == -1:
                return  # obstacles
            world[k, g] = -1
            flood_fill(world, k + 1, g)
            flood_fill(world, k, g + 1)
            flood_fill(world, k - 1, g)
            flood_fill(world, k, g - 1)

        i, j = first_free(world0)
        flood_fill(world0, i, j)
        if np.any(world0 == 0):
            return False
        else:
            return True

    def get_obstacle_map(self):
        """get obstacle map"""
        return (self.world.state == -1).astype(int)

    def get_goals(self):
        """get all agents' goal position"""
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.get_goal(i))
        return result

    def get_positions(self):
        """get all agents' position"""
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.get_pos(i))
        return result
    
    # def get_all_dist_map(self):
    #
    #     open_list = list()
    #
    #     num_dist_map = self.world.num_nodes - 2 + self.num_agents
    #     for i in range(num_dist_map):
    #         if i < self.world.num_nodes - 2:
    #             x, y = tuple(self.world.nodes_obs[i])
    #         else:
    #             x, y = tuple(self.world.get_goal(i - self.world.num_nodes + 2 + 1))
    #         open_list.append((x, y))
    #
    #     return pyCallCpp(self.world.state, open_list)

    def get_all_dist_map(self):
        """
        get all dist map to cover all possible calculation
        """
        num_dist_map = self.world.num_nodes - 2 + self.num_agents
        map_size = (self.world.state.shape[0], self.world.state.shape[1])
        dist_map = np.ones((num_dist_map, *map_size), dtype=np.int32) * 999999
        for i in range(num_dist_map):
            open_list = list()
            if i < self.world.num_nodes - 2:
                x, y = tuple(self.world.nodes_obs[i])
            else:
                x, y = tuple(self.world.get_goal(i - self.world.num_nodes + 2 + 1))
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up[0] >= 0 and self.get_obstacle_map()[up] == 0 and dist_map[i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    if up not in open_list:
                        open_list.append(up)

                down = x + 1, y
                if down[0] < map_size[0] and self.get_obstacle_map()[down] == 0 and dist_map[
                    i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    if down not in open_list:
                        open_list.append(down)

                left = x, y - 1
                if left[1] >= 0 and self.get_obstacle_map()[left] == 0 and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    if left not in open_list:
                        open_list.append(left)

                right = x, y + 1
                if right[1] < map_size[1] and self.get_obstacle_map()[right] == 0 and dist_map[
                    i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    if right not in open_list:
                        open_list.append(right)
        # print("num_nodes", len(self.world.nodes_obs))
        # print("num_nodes", self.world.num_nodes)
        # print("dis_map shape", dist_map.shape)
        return dist_map

    def set_world(self):
        """randomly generate a new task"""

        def get_connected_region(world0, regions_dict, x0, y0):
            # ensure at the beginning of an episode, all agents and their goal at the same connected region
            sys.setrecursionlimit(1000000)
            if (x0, y0) in regions_dict:  # have done
                return regions_dict[(x0, y0)]
            visited = set()
            sx, sy = world0.shape[0], world0.shape[1]
            work_list = [(x0, y0)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                if i < 0 or i >= sx or j < 0 or j >= sy:
                    continue
                if world0[i, j] == -1:
                    continue  # crashes
                if world0[i, j] > 0:
                    regions_dict[(i, j)] = visited
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x0, y0)] = visited
            return visited

        def padding_world(world):
            labeled_image, count = skimage.measure.label(world, background=-1, connectivity=1, return_num=True)
            num_comp = np.zeros(count)
            for i in range(labeled_image.shape[0]):
                for j in range(labeled_image.shape[1]):
                    for seg in range(1, count + 1):
                        if labeled_image[i, j] == seg:
                            num_comp[seg - 1] = num_comp[seg - 1] + 1
            list_num_comp = num_comp.tolist()
            max_index = list_num_comp.index(max(list_num_comp))
            padding_world = copy.deepcopy(labeled_image)
            for seg in range(1, count + 1):
                for i in range(labeled_image.shape[0]):
                    for j in range(labeled_image.shape[1]):
                        if labeled_image[i, j] == 0:
                            padding_world[i, j] = -1
                        elif labeled_image[i, j] == max_index + 1:
                            padding_world[i, j] = 0
                        else:
                            padding_world[i, j] = -1
            return padding_world

        # prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
        #                             self.PROB[1])  # sample a value from triangular distribution
        # size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
        #                         p=[.5, .25, .25])  # sample a value according to the given probability
        # # prob = self.PROB
        # # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
        # # here is the map without any agents nor goals
        # world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
        # for PRIMAL1 map
        # world = random_generator(SIZE_O=self.SIZE, PROB_O=self.PROB)
        # world = padding_world(world)
        # for MAZE map
        # world = maze_generator(env_size=self.SIZE, wall_components=(7, 8), obstacle_density=self.PROB)
        world, nodes_obs = house_generator(env_size=self.SIZE)
        world = world.astype(int)
        num_nodes = len(nodes_obs)
        # print(world)
        # randomize the position of agents
        agent_counter = 1
        agent_locations = []
        while agent_counter <= self.num_agents:
            x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
            if world[x, y] == 0:
                world[x, y] = agent_counter
                agent_locations.append((x, y))
                agent_counter += 1

        # randomize the position of goals
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions = dict()
        while goal_counter <= self.num_agents:
            agent_pos = agent_locations[goal_counter - 1]
            valid_tiles = get_connected_region(world, agent_regions, agent_pos[0], agent_pos[1])
            x, y = random.choice(list(valid_tiles))
            if goals[x, y] == 0 and world[x, y] != -1:
                # ensure new goal does not at the same grid of old goals or obstacles
                goals[x, y] = goal_counter
                goal_counter += 1
        self.world = State(world, goals, self.num_agents, nodes_obs, num_nodes)

    def observe(self, agent_id, nodes_obs, world):
        """return one agent's observation"""

        def get_specific_nodes(nodes, agent_id, world):
            # get access the agent and its goal's pos
            agent_pos = self.world.get_pos(agent_id)
            agent_pos = list(agent_pos)
            agent_gpos = self.world.get_goal(agent_id)
            agent_gpos = list(agent_gpos)
            # add agent pos and its goal pos to the graph
            # before this part, the nodes only contain map generated nodes
            nodes.append(agent_pos)
            nodes.append(agent_gpos)
            # the distance between agent to goal
            self.all_dist_map = np.array(self.all_dist_map)
            agent_to_goal = self.all_dist_map[agent_id - (self.num_agents + 1), agent_pos[0], agent_pos[1]]
            # add dimension to node  - distance (A* - Manhattan distance)
            for node_q in range(len(nodes)):
                # if the current node is the agent
                if node_q == len(nodes) - 2:
                    mh_dist_goal = manhattan_distance(nodes[-2], nodes[-1])
                    a_to_goal = self.all_dist_map[agent_id - (self.num_agents + 1), nodes[-2][0], nodes[-2][1]]
                    dis_to_agent = 0
                    dis_to_goal = a_to_goal - mh_dist_goal
                    extra_detour = 0
                    nodes[node_q].append(dis_to_agent)
                    nodes[node_q].append(dis_to_goal)
                    nodes[node_q].append(extra_detour)
                # if the current node is the agent's goal
                elif node_q == len(nodes) - 1:
                    mh_dist_agent = manhattan_distance([nodes[-2][0], nodes[-2][1]], nodes[-1])
                    a_to_agent = self.all_dist_map[agent_id - (self.num_agents + 1), nodes[-2][0], nodes[-2][1]]
                    dis_to_agent = a_to_agent - mh_dist_agent
                    dis_to_goal = 0
                    extra_detour = 0
                    nodes[node_q].append(dis_to_agent)
                    nodes[node_q].append(dis_to_goal)
                    nodes[node_q].append(extra_detour)
                # if the node is a normal node
                else:
                    mh_dist_agent = manhattan_distance(nodes[node_q], nodes[-2])
                    mh_dist_goal = manhattan_distance(nodes[node_q], nodes[-1])
                    if mh_dist_agent == 0:
                        a_to_agent = 0
                    else:
                        a_to_agent = self.all_dist_map[node_q, nodes[-2][0], nodes[-2][1]]
                    if mh_dist_goal == 0:
                        a_to_goal = 0
                    else:
                        a_to_goal = self.all_dist_map[node_q, nodes[-1][0], nodes[-1][1]]
                    dis_agent = a_to_agent - mh_dist_agent
                    dis_goal  = a_to_goal  - mh_dist_goal
                    extra_detour = a_to_agent + a_to_goal - agent_to_goal
                    nodes[node_q].append(dis_agent)
                    nodes[node_q].append(dis_goal)
                    nodes[node_q].append(extra_detour)
            # change all nodes as relative coordinates instead of global coordinates
            nodes = np.array(nodes)
            curr_agent_pos = self.world.get_pos(agent_id)
            for node_id in range(len(nodes)):
                for node_coord in range(2):
                    nodes[node_id][node_coord] = nodes[node_id][node_coord] - curr_agent_pos[node_coord]
            nodes = nodes.tolist()
            while len(nodes) > NetParameters.NUM_NODES:
                leave_node_index = random.randrange(len(nodes) - 2)
                nodes.remove(nodes[leave_node_index])
            return nodes

        assert (agent_id > 0)
        top_left = (self.world.get_pos(agent_id)[0] - self.observation_size // 2,
                    self.world.get_pos(agent_id)[1] - self.observation_size // 2)  # (top, left)
        obs_shape = (self.observation_size, self.observation_size)
        goal_map = np.zeros(obs_shape)  # own goal
        poss_map = np.zeros(obs_shape)  # agents
        goals_map = np.zeros(obs_shape)  # other observable agents' goal
        obs_map = np.zeros(obs_shape)  # obstacle
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):  # top and bottom
            for j in range(top_left[1], top_left[1] + self.observation_size):  # left and right
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of boundaries
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    continue
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] == agent_id:
                    # own position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.goals[i, j] == agent_id:
                    # own goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append(self.world.state[i, j])
                    poss_map[i - top_left[0], j - top_left[1]] = 1

        for agent in visible_agents:
            x, y = self.world.get_goal(agent)
            # project the goal out of FOV to the boundary of FOV
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.world.get_goal(agent_id)[0] - self.world.get_pos(agent_id)[0]  # distance on x axes
        dy = self.world.get_goal(agent_id)[1] - self.world.get_pos(agent_id)[1]  # distance on y axes
        mag = (dx ** 2 + dy ** 2) ** .5  # total distance
        if mag != 0:  # normalized
            dx = dx / mag
            dy = dy / mag
        # prepare a world only for graph generating, in which only contain obs
        # world_only_obs = -1 * self.get_obstacle_map()
        # print(nodes_obs)
        nodes = get_specific_nodes(nodes_obs, agent_id, world)
        # the current/ego agent index in nodes and agent_nodes/goal_nodes
        in_node_index = len(nodes) - 2
        in_agent_index = agent_id - 1
        return [poss_map, goal_map, goals_map, obs_map], [dx, dy, mag], nodes, in_node_index, in_agent_index

    def get_intention(self, intent_world, start_positions, goal_positions, intent_steps=11):
        agent_input = []
        agent_path = []
        for i in range(self.num_agents):
            if start_positions[i] != goal_positions[i]:
                agent_path, _ = astar_4(intent_world, tuple(start_positions[i]), goal_positions[i])
                # get access the currect path
                agent_path = agent_path[::-1]
            while len(agent_path) < intent_steps:
                agent_path.append(goal_positions[i])
            new_agent_path = agent_path[: intent_steps]
            mean = np.mean(new_agent_path, axis=0)
            variance = np.var(new_agent_path, axis=0)
            dx_frontier = new_agent_path[-1][0] - start_positions[i][0]  # distance on x axes
            dy_frontier = new_agent_path[-1][1] - start_positions[i][1]  # distance on y axes
            mag = (dx_frontier ** 2 + dy_frontier ** 2) ** .5  # total distance
            if mag != 0:  # normalized
                dx_frontier = dx_frontier / mag
                dy_frontier = dy_frontier / mag
            agent_input.append([start_positions[i][0], start_positions[i][1], mean[0], mean[1], variance[0], variance[1], dx_frontier, dy_frontier, mag])
        return agent_input

    def _reset(self, num_agents):
        """restart a new task"""
        self.num_agents = num_agents
        self.max_on_goal = 0
        if self.viewer is not None:
            self.viewer = None

        self.set_world()  # back to the initial situation
        self.all_dist_map = self.get_all_dist_map()
        return False

    def _reload(self, world_with_agent, world_with_goals, num_agents, nodes_obs):
        self.num_agents = num_agents
        self.max_on_goal = 0
        if self.viewer is not None:
            self.viewer = None
        self.set_world()
        num_nodes = len(nodes_obs)
        self.world = State(world_with_agent, world_with_goals, num_agents, nodes_obs.tolist(), num_nodes)
        self.all_dist_map = self.get_all_dist_map()

    def astar(self, world, start, goal, robots):
        """A* function for single agent"""
        for (i, j) in robots:
            world[i, j] = 1
        try:
            path = od_mstar.find_path(world, [start], [goal], inflation=1, time_limit=5 * 60)
        except NoSolutionError:
            path = None
        for (i, j) in robots:
            world[i, j] = 0
        return path

    def get_blocking_reward(self, agent_id):
        """calculates how many agents are prevented from reaching goal and returns the blocking penalty"""
        other_agents = []
        other_locations = []
        inflation = 10
        top_left = (self.world.get_pos(agent_id)[0] - self.observation_size // 2,
                    self.world.get_pos(agent_id)[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        for agent in range(1, self.num_agents):
            if agent == agent_id:
                continue
            x, y = self.world.get_pos(agent)
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agent not in FOV
                continue
            other_agents.append(agent)
            other_locations.append((x, y))

        num_blocking = 0
        world = self.get_obstacle_map()
        for agent in other_agents:
            other_locations.remove(self.world.get_pos(agent))
            # before removing
            path_before = self.astar(world, self.world.get_pos(agent), self.world.get_goal(agent),
                                     robots=other_locations + [self.world.get_pos(agent_id)])
            # after removing
            path_after = self.astar(world, self.world.get_pos(agent), self.world.get_goal(agent),
                                    robots=other_locations)
            other_locations.append(self.world.get_pos(agent))
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                num_blocking += 1
        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking

    def list_next_valid_actions(self, agent_id, prev_action=0):
        """obtain the valid actions that can not lead to colliding with obstacles and boundaries
        or backing to previous position at next time step"""
        available_actions = [0]  # staying still always allowed

        agent_pos = self.world.get_pos(agent_id)
        ax, ay = agent_pos[0], agent_pos[1]

        for action in range(1, EnvParameters.N_ACTIONS):  # every action except 0
            direction = self.world.get_dir(action)
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.world.state.shape[0] or ax + dx < 0 or ay + dy >= self.world.state.shape[
                    1] or ay + dy < 0:  # out of boundaries
                continue
            if self.world.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                continue
            # otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:  # back to previous position
            available_actions.remove(opposite_actions[prev_action])
        return available_actions

    def joint_step(self, actions, num_step):
        """execute joint action and obtain reward"""
        action_status = self.world.joint_move(actions)
        valid_actions = [action_status[i] >= 0 for i in range(self.num_agents)]
        #     2: action executed and agent leave its own goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of boundaries
        #    -2: collision with obstacles
        #    -3: collision with agents

        # initialization
        blockings = np.zeros((1, self.num_agents), dtype=np.float32)
        rewards = np.zeros((1, self.num_agents), dtype=np.float32)
        obs = np.zeros((1, self.num_agents, 4, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        # set for graph
        graph_nodes = np.zeros((1, self.num_agents, NetParameters.NUM_NODES, NetParameters.NUM_FEATURE), dtype=np.float32)
        agent_intent = np.zeros((1, self.num_agents, self.num_agents, NetParameters.NUM_INTENTION_FEATURE), dtype=np.float32)
        # set to get the index of the graph
        node_index = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        agent_index = np.zeros((1, self.num_agents, 1), dtype=np.float32)

        next_valid_actions = []
        num_blockings = 0
        leave_goals = 0
        num_collide = 0
        # calculate the agent future feature
        agent_feature = self.get_intention(-1 * self.get_obstacle_map(), self.get_positions(), self.get_goals())

        for i in range(self.num_agents):
            if actions[i] == 0:  # staying still
                if action_status[i] == 1:  # stayed on goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                    if self.num_agents < 32:  # do not calculate A* for increasing speed
                        x, num_blocking = self.get_blocking_reward(i + 1)
                        num_blockings += num_blocking
                        rewards[:, i] += x
                        if x < 0:
                            blockings[:, i] = 1
                elif action_status[i] == 0:  # stayed off goal
                    rewards[:, i] = EnvParameters.IDLE_COST  # stop penalty
                elif action_status[i] == -3 or action_status[i] == -2 or action_status[i] == -1:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_collide += 1

            else:  # moving
                if action_status[i] == 1:  # reached goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                elif action_status[i] == -2 or action_status[i] == -1 or action_status[i] == -3:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_collide += 1
                elif action_status[i] == 2:  # leave own goal
                    rewards[:, i] = EnvParameters.ACTION_COST
                    leave_goals += 1
                else:  # nothing happen
                    rewards[:, i] = EnvParameters.ACTION_COST
            nodes_obs_inner = copy.deepcopy(self.world.nodes_obs)
            state = self.observe(i + 1, nodes_obs_inner, -1 * self.get_obstacle_map())
            obs[:, i, :, :, :] = state[0]
            vector[:, i, : 3] = state[1]
            graph_nodes[:, i, :, :] = np.pad(state[2], ((0, NetParameters.NUM_NODES - len(state[2])), (0, 0)), 'constant')
            node_index[:, i, :] = state[3]
            agent_index[:, i, :] = state[4]
            # the dynamic feature
            agent_intent[:, i, :, :] = agent_feature

            next_valid_actions.append(self.list_next_valid_actions(i + 1, actions[i]))

        # graph_nodes = normalization_feature(graph_nodes, self.world.num_nodes)

        done, num_on_goal = self.world.task_done()
        if num_on_goal > self.max_on_goal:
            self.max_on_goal = num_on_goal
        if num_step >= EnvParameters.EPISODE_LEN - 1:
            done = True
        return obs, vector, graph_nodes, agent_intent, node_index, agent_index, rewards, \
               done, next_valid_actions, blockings, valid_actions, num_blockings, leave_goals, num_on_goal,\
               self.max_on_goal, num_collide, action_status

    # def create_rectangle(self, x, y, width, height, fill, permanent=False):
    #     """draw a rectangle to represent an agent"""
    #     ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
    #     rect = rendering.FilledPolygon(ps)
    #     rect.set_color(fill[0], fill[1], fill[2])
    #     rect.add_attr(rendering.Transform())
    #     if permanent:
    #         self.viewer.add_geom(rect)
    #     else:
    #         self.viewer.add_onetime(rect)
    #
    # def create_circle(self, x, y, diameter, size, fill, resolution=20):
    #     """draw a circle to represent a goal"""
    #     c = (x + size / 2, y + size / 2)
    #     dr = math.pi * 2 / resolution
    #     ps = []
    #     for i in range(resolution):
    #         x = c[0] + math.cos(i * dr) * diameter / 2
    #         y = c[1] + math.sin(i * dr) * diameter / 2
    #         ps.append((x, y))
    #     circ = rendering.FilledPolygon(ps)
    #     circ.set_color(fill[0], fill[1], fill[2])
    #     circ.add_attr(rendering.Transform())
    #     self.viewer.add_onetime(circ)
    #
    # def init_colors(self):
    #     """the colors of agents and goals"""
    #     c = {a + 1: hsv_to_rgb(np.array([a / float(self.num_agents), 1, 1])) for a in range(self.num_agents)}
    #     return c
    #
    # def _render(self, mode='human', close=False, screen_width=800, screen_height=800, action_probs=None):
    #     if close:
    #         return
    #     # values is an optional parameter which provides a visualization for the value of each agent per step
    #     size = screen_width / max(self.world.state.shape[0], self.world.state.shape[1])
    #     colors = self.init_colors()
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         self.reset_renderer = True
    #     if self.reset_renderer:
    #         self.create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6), permanent=True)
    #         for i in range(self.world.state.shape[0]):
    #             start = 0
    #             end = 1
    #             scanning = False
    #             write = False
    #             for j in range(self.world.state.shape[1]):
    #                 if self.world.state[i, j] != -1 and not scanning:  # free
    #                     start = j
    #                     scanning = True
    #                 if (j == self.world.state.shape[1] - 1 or self.world.state[i, j] == -1) and scanning:
    #                     end = j + 1 if j == self.world.state.shape[1] - 1 else j
    #                     scanning = False
    #                     write = True
    #                 if write:
    #                     x = i * size
    #                     y = start * size
    #                     self.create_rectangle(x, y, size, size * (end - start), (1, 1, 1), permanent=True)
    #                     write = False
    #     for agent in range(1, self.num_agents + 1):
    #         i, j = self.world.get_pos(agent)
    #         x = i * size
    #         y = j * size
    #         color = colors[self.world.state[i, j]]
    #         self.create_rectangle(x, y, size, size, color)
    #         i, j = self.world.get_goal(agent)
    #         x = i * size
    #         y = j * size
    #         color = colors[self.world.goals[i, j]]
    #         self.create_circle(x, y, size, size, color)
    #         if self.world.get_goal(agent) == self.world.get_pos(agent):
    #             color = (0, 0, 0)
    #             self.create_circle(x, y, size, size, color)
    #     if action_probs is not None:
    #         for agent in range(1, self.num_agents + 1):
    #             # take the a_dist from the given data and draw it on the frame
    #             a_dist = action_probs[agent - 1]
    #             if a_dist is not None:
    #                 for m in range(EnvParameters.N_ACTIONS):
    #                     dx, dy = self.world.get_dir(m)
    #                     x = (self.world.get_pos(agent)[0] + dx) * size
    #                     y = (self.world.get_pos(agent)[1] + dy) * size
    #                     s = a_dist[m] * size
    #                     self.create_circle(x, y, s, size, (0, 0, 0))
    #     self.reset_renderer = False
    #     result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
    #     return result

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    env = MAPFEnv(num_agents=EnvParameters.N_AGENTS)

    kk = []
    yy = []
    for i in range(1, 9):
        k = env.world.get_pos(i)
        kk.append(k)
        y = env.world.get_goal(i)
        yy.append(y)
    print(np.array(kk))
    print(np.array(yy))
    # print(env.world.state)

    plt.ion()
    plt.imshow(env.world.state)
    plt.pause(0.05)
    plt.close()
    plt.ioff()
