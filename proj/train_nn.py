import getopt
import sys
import numpy as np
from cgolai.cgol import Model, CgolProblem
from cgolai.ai import RL


def run(filename):
    rl = train()
    test_res, base_res = report(rl, filename)
    print_report(test_res, base_res, full=True)


def print_report(test_res, base_res, full=False):
    print()
    print('base')
    total_steps = 0
    total_rewards = 0
    total_maxouts = 0
    for i, (steps, rewards, maxout) in enumerate(base_res):
        reward = sum(rewards)
        total_steps += steps
        total_rewards += reward
        if maxout:
            total_maxouts += 1
        if full:
            print('base trial:', i, end='; ')
            print('steps:', steps, end='; ')
            print('reward:', reward, end='; ')
            print('maxout:', maxout)
    print('total steps:', total_steps)
    print('total rewards:', total_rewards)
    print('total maxouts:', total_maxouts)
    print()

    print()
    print('test')
    total_steps = 0
    total_rewards = 0
    total_maxouts = 0
    for i, (steps, rewards, maxout) in enumerate(test_res):
        reward = sum(rewards)
        total_steps += steps
        total_rewards += reward
        if maxout:
            total_maxouts += 1
        if full:
            print('test trial:', i, end='; ')
            print('steps:', steps, end='; ')
            print('reward:', reward, end='; ')
            print('maxout:', maxout)
    print('total steps:', total_steps)
    print('total rewards:', total_rewards)
    print('total maxouts:', total_maxouts)
    print()


def report(rl, filename):
    model = rl.get_problem().model
    model.set_filename("train_{}".format(filename))
    model.save()

    trials = 10
    test_res = measure(rl, model, trials, random=False)
    model.set_filename("test_{}".format(filename))
    model.save()

    base_res = measure(rl, model, trials, random=True)
    model.set_filename("base_{}".format(filename))
    model.save()
    return test_res, base_res


def measure(rl, model, trials, random=False, max_length=100):
    """

    :param rl: The trained rl model
    :param model: The model
    :param trials: number of trials
    :param random: random moves or use rl's moves
    :param max_length: max length before stopping trial
    :return: list containing results from trial. results are (steps, list of rewards, did maxout)
    """
    problem = rl.get_problem()
    model.load_iter(0)
    trial_results = []
    for _ in range(trials):
        problem.reset()
        steps = 0
        rewards = []
        while not problem.is_terminal() and steps < max_length:
            steps += 1
            action, _ = rl.choose_best_action(explore=random, epsilon=1.0)
            _, reward = problem.do(action)
            rewards.append(reward)
        trial_results.append((steps, rewards, not problem.is_terminal()))
    return trial_results


def train():
    verbose = True
    LIFE = True  # for visuals
    D___ = False
    init_flip = np.array([
        [D___, LIFE, D___, D___],
        [LIFE, D___, D___, LIFE],
        [D___, D___, D___, D___],
        [D___, D___, D___, D___],
    ])
    model = Model(size=init_flip.shape)
    problem = CgolProblem(model, init_flip=init_flip)
    rl = RL(problem, shape=[None, *(3*[problem.length]), None], verbose=verbose,
            mu=0.01,
            batches=100,
            batch_size=50,
            max_steps=1000,
            # epsilon_decay_factor=0.9,
            # epsilon_init=0.5,
            )
    rl.train(iterations=100)
    return rl


def parse_args(config_opts, config_defaults):
    """
    Args:
        config_defaults dict:
            '-{arg}': (Name[, Set Value[, cast]])
            Example: {'-f': ("file", ), '-c': ("ctrl",True)}
        config_opts:
            getopts options argument
            'cf:' for example
    """
    config = {}

    # proc opts
    optlist, args = getopt.getopt(sys.argv[1:], config_opts)
    optlist = dict(optlist)
    for k, v in optlist.items():
        if optlist[k] == '':  # fill default
            optlist[k] = config_defaults[k][1]
        if len(config_defaults[k]) == 3:  # cast
            optlist[k] = config_defaults[k][2](optlist[k])
        config[config_defaults[k][0]] = optlist[k]

    # proc args
    if len(args) > 0:
        raise Exception("wasn't expecting arguments")

    return config


if __name__ == '__main__':
    config_opts = 'f:'
    config_defaults = {
        '-f': ('filename', ),
    }
    config = parse_args(config_opts, config_defaults)
    run(**config)
