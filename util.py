import copy
import random

import imageio
import numpy as np
import torch
import wandb

from alg_parameters import *


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def write_to_tensorboard(global_summary, step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True,
                         greedy=True):
    """record performance using tensorboard"""
    # summary = SummaryWriter()
    if imitation_loss is not None:
        global_summary.add_scalar(tag='Loss/Imitation_loss', scalar_value=imitation_loss[0], global_step=step)
        global_summary.add_scalar(tag='Grad/Imitation_grad', scalar_value=imitation_loss[1], global_step=step)
        # global_summary.flush()
        return
    if evaluate:
        if greedy:
            global_summary.add_scalar(tag='Perf_greedy_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

        else:
            global_summary.add_scalar(tag='Perf_random_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        global_summary.add_scalar(tag='Perf/Reward', scalar_value=performance_dict['per_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
        global_summary.add_scalar(tag='Perf/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
        global_summary.add_scalar(tag='Perf/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
        global_summary.add_scalar(tag='Perf/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                global_summary.add_scalar(tag='Grad/' + name, scalar_value=val, global_step=step)
            else:
                global_summary.add_scalar(tag='Loss/' + name, scalar_value=val, global_step=step)
    # global_summary.flush()


def write_to_wandb(step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True, greedy=True):
    """record performance using wandb"""
    if imitation_loss is not None:
        wandb.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        wandb.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
    if evaluate:
        if greedy:
            wandb.log({'Perf_greedy_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_greedy_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_block': performance_dict['per_block']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
            wandb.log({'Perf_greedy_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
            wandb.log({'Perf_greedy_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_collide': performance_dict['per_num_collide']}, step=step)

        else:
            wandb.log({'Perf_random_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_random_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_random_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            wandb.log({'Perf_random_eval/Num_block': performance_dict['per_block']}, step=step)
            wandb.log({'Perf_random_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
            wandb.log({'Perf_random_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
            wandb.log({'Perf_random_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
            wandb.log({'Perf_random_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
            wandb.log({'Perf_random_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
            wandb.log({'Perf_random_eval/Num_collide': performance_dict['per_num_collide']}, step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        wandb.log({'Perf/Reward': performance_dict['per_r']}, step=step)
        wandb.log({'Perf/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        wandb.log({'Perf/Episode_length': performance_dict['per_episode_len']}, step=step)
        wandb.log({'Perf/Num_block': performance_dict['per_block']}, step=step)
        wandb.log({'Perf/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
        wandb.log({'Perf/Final_goals': performance_dict['per_final_goals']}, step=step)
        wandb.log({'Perf/Half_goals': performance_dict['per_half_goals']}, step=step)
        wandb.log({'Perf/Block_accuracy': performance_dict['per_block_acc']}, step=step)
        wandb.log({'Perf/Max_goals': performance_dict['per_max_goals']}, step=step)
        wandb.log({'Perf/Num_collide': performance_dict['per_num_collide']},
                  step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                wandb.log({'Grad/' + name: val}, step=step)
            else:
                wandb.log({'Loss/' + name: val}, step=step)


def make_gif(images, file_name):
    """record gif"""
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("wrote gif")


def reset_env(env, num_agent):
    """reset environment"""
    done = env._reset(num_agent)
    prev_action = np.zeros(num_agent)
    valid_actions = []
    obs = np.zeros((1, num_agent, 4, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
    vector = np.zeros((1, num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)
    # set for graph
    graph_nodes = np.zeros((1, num_agent, NetParameters.NUM_NODES, 5), dtype=np.float32)
    agent_intent = np.zeros((1, num_agent, num_agent, NetParameters.NUM_INTENTION_FEATURE), dtype=np.float32)
    # find the index of the ego agent
    node_index = np.zeros((1, num_agent, 1), dtype=np.float32)
    agent_index = np.zeros((1, num_agent, 1), dtype=np.float32)
    # calculate the agent future feature
    agent_feature = env.get_intention(-1 * env.get_obstacle_map(), env.get_positions(), env.get_goals())

    for i in range(num_agent):
        valid_action = env.list_next_valid_actions(i + 1)
        nodes_obs_inner = copy.deepcopy(env.world.nodes_obs)
        s = env.observe(i + 1, nodes_obs_inner, -1 * env.get_obstacle_map())
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        vector[:, i, -1] = prev_action[i]
        graph_nodes[:, i, :, :] = np.pad(s[2], ((0, NetParameters.NUM_NODES - len(s[2])), (0, 0)), 'constant')
        node_index[:, i, :] = s[3]
        agent_index[:, i, :] = s[4]
        agent_intent[:, i, :, :] = agent_feature
        valid_actions.append(valid_action)
        train_valid[i, valid_action] = 1
    # graph_nodes = normalization_feature(graph_nodes, env.world.num_nodes)
    return done, valid_actions, obs, vector, graph_nodes, agent_intent, node_index, agent_index, train_valid


def reset_env_load(env, world_with_agent, world_with_goals, num_agent, nodes_obs):
    """reset environment"""
    done = False
    env._reload(world_with_agent, world_with_goals, num_agent, nodes_obs)
    prev_action = np.zeros(num_agent)
    valid_actions = []
    obs = np.zeros((1, num_agent, 4, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
    vector = np.zeros((1, num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)
    # set for graph
    graph_nodes = np.zeros((1, num_agent, NetParameters.NUM_NODES, NetParameters.NUM_FEATURE), dtype=np.float32)
    agent_intent = np.zeros((1, num_agent, num_agent, NetParameters.NUM_INTENTION_FEATURE), dtype=np.float32)
    # find the index of the ego agent
    node_index = np.zeros((1, num_agent, 1), dtype=np.float32)
    agent_index = np.zeros((1, num_agent, 1), dtype=np.float32)
    # calculate the agent future feature
    agent_feature = env.get_intention(-1 * env.get_obstacle_map(), env.get_positions(), env.get_goals())

    for i in range(num_agent):
        valid_action = env.list_next_valid_actions(i + 1)
        nodes_obs_inner = copy.deepcopy(env.world.nodes_obs)
        s = env.observe(i + 1, nodes_obs_inner, -1 * env.get_obstacle_map())
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        vector[:, i, -1] = prev_action[i]
        graph_nodes[:, i, :, :] = np.pad(s[2], ((0, NetParameters.NUM_NODES - len(s[2])), (0, 0)), 'constant')
        node_index[:, i, :] = s[3]
        agent_index[:, i, :] = s[4]
        agent_intent[:, i, :, :] = agent_feature
        valid_actions.append(valid_action)
        train_valid[i, valid_action] = 1
    # graph_nodes = normalization_feature(graph_nodes, env.world.num_nodes)
    return done, valid_actions, obs, vector, graph_nodes, agent_intent, node_index, agent_index, train_valid


def one_step(env, one_episode_perf, actions, pre_block, num_agent):
    """run one step"""
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)
    obs, vector, graph_nodes, agent_intent, node_index, agent_index, rewards, done, next_valid_actions, \
    blockings, _, num_blockings, leave_goals, num_on_goal, max_on_goal, num_collide, action_status\
        = env.joint_step(actions, one_episode_perf['num_step'])

    one_episode_perf['block'] += num_blockings
    one_episode_perf['num_leave_goal'] += leave_goals
    one_episode_perf['num_collide'] += num_collide
    vector[:, :, -1] = actions
    for i in range(num_agent):
        train_valid[i, next_valid_actions[i]] = 1
        if (pre_block[i] < 0.5) == blockings[:, i]:
            one_episode_perf['wrong_blocking'] += 1
    one_episode_perf['num_step'] += 1
    return rewards, next_valid_actions, obs, vector, graph_nodes, agent_intent,\
           node_index, agent_index, train_valid, done, blockings, num_on_goal, \
           one_episode_perf, max_on_goal, action_status


def update_perf(one_episode_perf, performance_dict, num_on_goals, max_on_goals, num_agent):
    """record batch performance"""
    performance_dict['per_r'].append(one_episode_perf['episode_reward'])
    performance_dict['per_valid_rate'].append(
        ((one_episode_perf['num_step'] * num_agent) - one_episode_perf['invalid']) / (
                one_episode_perf['num_step'] * num_agent))
    performance_dict['per_episode_len'].append(one_episode_perf['num_step'])
    performance_dict['per_block'].append(one_episode_perf['block'])
    performance_dict['per_leave_goal'].append(one_episode_perf['num_leave_goal'])
    performance_dict['per_num_collide'].append(one_episode_perf['num_collide'])
    performance_dict['per_final_goals'].append(num_on_goals)
    performance_dict['per_block_acc'].append(
        ((one_episode_perf['num_step'] * num_agent) - one_episode_perf['wrong_blocking']) / (
                one_episode_perf['num_step'] * num_agent))
    performance_dict['per_max_goals'].append(max_on_goals)
    return performance_dict


def normalization_feature(data, num_nodes):
    a = data[0]
    normalize_data = []
    for agent_nodes in a:
        nodes = copy.deepcopy(agent_nodes)
        # print(nodes.shape)
        for j in range(agent_nodes.shape[1]):
            b = []
            for i in range(num_nodes):
                b.append(abs(agent_nodes[i][j]))
            # mean = np.mean(b)
            # std = np.std(b)
            max_value = np.max(b)
            threshold = 1e-5
            # print(f"std is {std}")
            # print(f"mean is {mean}, std is {std}")
            # print(f"max_value is {max_value}")
            for k in range(num_nodes):
                nodes[k][j] = nodes[k][j] / (max_value + threshold)
                # print(f"nodes kj is {nodes[k][j]}")
        normalize_data.append(nodes)
    normalize_data = np.array([normalize_data])
    # print(f"final data is {normalize_data}")
    return normalize_data
