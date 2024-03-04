import datetime

""" Hyperparameters """


class EnvParameters:
    N_AGENTS = 8   # number of agents used in training todo 8
    N_ACTIONS = 5
    EPISODE_LEN = 256  # maximum episode length in training
    FOV_SIZE = 9
    WORLD_SIZE = (10, 40)   # todo: (10, 40)
    OBSTACLE_PROB = (0.0, 0.3)  # todo: (0, 0.3)
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 0.0
    COLLISION_COST = -2
    BLOCKING_COST = -1


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95      # discount factor
    LAM = 0.95        # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    N_EPOCHS = 10
    N_ENVS = 32   # number of processes  todo 32
    N_MAX_STEPS = 3e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 8  # number of time steps per process per data collection todo 2**10
    MINIBATCH_SIZE = int(2 ** 8)   # todo 2**10
    DEMONSTRATION_PROB = 0.1  # imitation learning rate  todo: 0.1


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 4    # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 4     # [dx, dy, d total, action t-1]
    NUM_NODES = 50    # There are always to many nodes in PRIMAL1 map, if we select PRIMAL2 map, we can decrease the num of nodes
    MAX_NEIGHBOR = 50
    EMBEDDING_DIM = 512
    NUM_FEATURE = 5
    NUM_INTENTION_FEATURE = 9


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 2


class RecordingParameters:
    RETRAIN = False
    WANDB = True   # set as True when real training started
    TENSORBOARD = False
    TXT_WRITER = True
    ENTITY = 'ALPHA_PLUS'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'ALPHA_ICRA'
    EXPERIMENT_NAME = 'Intention_frontier_11'
    EXPERIMENT_NOTE = 'In this version, there is no FA layer but only 1 layers of self-attention'
    SAVE_INTERVAL = 5e5  # interval of saving model0
    BEST_INTERVAL = 0  # interval of saving model0 with the best performance
    GIF_INTERVAL = 1e6  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TEST_GIFS_PATH = './test_gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TXT_NAME = 'alg.txt'
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss', 'valid_loss',
                 'blocking_loss', 'clipfrac',
                 'grad_norm', 'advantage']


all_args = {'N_AGENTS': EnvParameters.N_AGENTS, 'N_ACTIONS': EnvParameters.N_ACTIONS,
            'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'FOV_SIZE': EnvParameters.FOV_SIZE,
            'WORLD_SIZE': EnvParameters.WORLD_SIZE,
            'OBSTACLE_PROB': EnvParameters.OBSTACLE_PROB,
            'ACTION_COST': EnvParameters.ACTION_COST,
            'IDLE_COST': EnvParameters.IDLE_COST, 'GOAL_REWARD': EnvParameters.GOAL_REWARD,
            'COLLISION_COST': EnvParameters.COLLISION_COST,
            'BLOCKING_COST': EnvParameters.BLOCKING_COST,
            'lr': TrainingParameters.lr, 'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
            'CLIPRANGE': TrainingParameters.CLIP_RANGE, 'MAX_GRAD_NORM': TrainingParameters.MAX_GRAD_NORM,
            'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
            'VALUE_COEF': TrainingParameters.VALUE_COEF,
            'POLICY_COEF': TrainingParameters.POLICY_COEF,
            'VALID_COEF': TrainingParameters.VALID_COEF, 'BLOCK_COEF': TrainingParameters.BLOCK_COEF,
            'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
            'N_MAX_STEPS': TrainingParameters.N_MAX_STEPS,
            'N_STEPS': TrainingParameters.N_STEPS, 'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
            'DEMONSTRATION_PROB': TrainingParameters.DEMONSTRATION_PROB,
            'NET_SIZE': NetParameters.NET_SIZE, 'NUM_CHANNEL': NetParameters.NUM_CHANNEL,
            'GOAL_REPR_SIZE': NetParameters.GOAL_REPR_SIZE, 'VECTOR_LEN': NetParameters.VECTOR_LEN,
            'SEED': SetupParameters.SEED, 'USE_GPU_LOCAL': SetupParameters.USE_GPU_LOCAL,
            'USE_GPU_GLOBAL': SetupParameters.USE_GPU_GLOBAL,
            'NUM_GPU': SetupParameters.NUM_GPU, 'RETRAIN': RecordingParameters.RETRAIN,
            'WANDB': RecordingParameters.WANDB,
            'TENSORBOARD': RecordingParameters.TENSORBOARD, 'TXT_WRITER': RecordingParameters.TXT_WRITER,
            'ENTITY': RecordingParameters.ENTITY,
            'TIME': RecordingParameters.TIME, 'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
            'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
            'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,
            'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL, "BEST_INTERVAL": RecordingParameters.BEST_INTERVAL,
            'GIF_INTERVAL': RecordingParameters.GIF_INTERVAL, 'EVAL_INTERVAL': RecordingParameters.EVAL_INTERVAL,
            'EVAL_EPISODES': RecordingParameters.EVAL_EPISODES, 'RECORD_BEST': RecordingParameters.RECORD_BEST,
            'MODEL_PATH': RecordingParameters.MODEL_PATH, 'GIFS_PATH': RecordingParameters.GIFS_PATH,
            'SUMMARY_PATH': RecordingParameters.SUMMARY_PATH,
            'TXT_NAME': RecordingParameters.TXT_NAME}
