import ray
import torch
from model import Model
from mapf_gym import MapfGym
from alg_parameters import *
from util import OneEpPerformance, make_gif
import numpy as np
import os
import pickle

# path to the trained model folder (the one that contains net_checkpoint.pkl)
MODEL_PATH = 'models/a2lpha/alpha_maze_829-09-262238/final'
# test instances: <TEST_DIR>/32length_{N_AGENTS}agents_0.3density.pth, 200 cases per file
TEST_DIR = './32_32_house_0.2_0.3'
NUM_RUNS = 200
NUM_CPUS = 25


@ray.remote(num_cpus=1)
def test_model(num_episode):
    # load the fully trained model
    net_path_checkpoint = MODEL_PATH + "/net_checkpoint.pkl"
    # during the testing, we are gonna to use cpu
    # load the trained model
    net_dict = torch.load(net_path_checkpoint, map_location=torch.device('cpu'), weights_only=False)
    test_device = torch.device('cpu')
    test_model = Model(0, test_device)
    test_model.network.load_state_dict(net_dict['model'])
    test_model.network.eval()
    with torch.no_grad():
        success_status, makespan, num_reached = test_one_case(model=test_model, test_episode=num_episode)
    return success_status, makespan, num_reached


def test_one_case(model, test_episode):
    """
        The env_size should be a tuple as env_size = (20, 20)
        num_agents = EnvParameters.N_AGENTS (50, 100, 150, 200, 250, 300)
        model is the evaluation model - the trained network
        device: run the execution on cpu or gpu
        greedy: choose the action greedy or randomly
    """
    print(f"current_episode is {test_episode}")
    # load the parameters from args, i.e., the item in tests
    # env_info[0] = world with agents; env_info[1] = world with goals; env_info[2] = obstacle map;
    # env_info[3] = agents;            env_info[4] = agents goals;     env_info[5] = nodes_obs;
    # env_info[6] = num_agents
    # general tests
    with open(os.path.join(TEST_DIR, '32length_{}agents_0.3density.pth'.format(EnvParameters.N_AGENTS)), 'rb') as f:
        env_info = pickle.load(f)

    fully_arrived = 0
    one_ep_step = 0
    num_reached = 0
    # episodeFrames = []
    oneEpisodePerformance = OneEpPerformance()

    # Build the env directly from the (solvable) saved case. Calling
    # MapfGym.__init__ here first generated a throwaway RANDOM map/agent layout
    # and ran A* on it before replicate() ever loaded the case; that random
    # layout is not guaranteed solvable, so it crashed with "No Path Exists"
    # at random. replicate() rebuilds every attribute __init__ would have set.
    env = MapfGym.__new__(MapfGym)
    env.replicate(-1 * env_info[test_episode][2], env_info[test_episode][3], env_info[test_episode][4])

    done = False
    while not done:
        obs, vector, svo, conflict_index, same_index, msg, graph_nodes, agent_intent, node_index = env.getAllObservations()
        # episodeFrames.append(env._render())
        actions, pre_block, _, _, ps, svo_output = model.evaluate(
            obs, vector, svo, conflict_index, same_index, msg, graph_nodes, agent_intent, node_index, num_agent=env.n_agents
        )
        actionStatus, fixedActions = env.getActionStatus(actions, svo_output, ps=ps)
        oneEpisodePerformance.invalid += len(env.getStaticColl(actionStatus))
        svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(fixedActions, actionStatus)
        oneEpisodePerformance.block += np.sum(blockings)
        oneEpisodePerformance.numLeaveGoal += np.sum(leaveGoals)
        oneEpisodePerformance.numCollide += np.sum(numCollide)
        oneEpisodePerformance.numStep += 1
        for i in range(env.n_agents):
            if (pre_block[i] < 0.5) == blockings[:, i]:
                oneEpisodePerformance.wrongBlocking += 1
        oneEpisodePerformance.episodeReward += np.sum(baseRewards)
        goalsReached, truelly_done = env.jointStep(fixedActions)
        if truelly_done or ((oneEpisodePerformance.numStep + 1) % EnvParameters.EPISODE_LEN == 0):
            done = True
        oneEpisodePerformance.maxGoals = max(oneEpisodePerformance.maxGoals, np.sum(goalsReached))
        if np.sum(goalsReached) == env.n_agents:
            fully_arrived = 1
        one_ep_step = oneEpisodePerformance.numStep
        num_reached = oneEpisodePerformance.maxGoals

        # if done:
        #     episodeFrames.append(env._render())
        #     if not os.path.exists(RecordingParameters.TEST_GIFS_PATH):
        #         os.makedirs(RecordingParameters.TEST_GIFS_PATH)
        #     # print("frames:", len(episode_frames))
        #     images = np.array(episodeFrames[:-1])
        #     name = str(test_episode)
        #     make_gif(images, RecordingParameters.TEST_GIFS_PATH + "/" + name + '.gif')

    return fully_arrived, one_ep_step, num_reached



if __name__ == "__main__":
    # init some metric
    total_success = 0
    total_step = 0
    total_reach = 0
    # how many env should test in parallel
    ray.init(num_cpus=NUM_CPUS)
    num_runs = NUM_RUNS
    results = ray.get([test_model.remote(i) for i in range(num_runs)])
    for result in results:
        if result[0] == 1:
            total_success += result[0]
        total_step = total_step + result[1]
        total_reach = total_reach + result[2]
    print(f"Map type is house; env size is 32; num_agent is {EnvParameters.N_AGENTS}; test set is {TEST_DIR}")
    print("The max steps is", EnvParameters.EPISODE_LEN)
    print(f"success rate is: {total_success / num_runs}")
    print("the average step is: ", total_step / num_runs)
    print("the reach rate is: ", total_reach / (EnvParameters.N_AGENTS * num_runs))
