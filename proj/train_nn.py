import getopt
import sys
import numpy as np
from cgolai.cgol import Model, CgolProblem
from cgolai.ai import RL


def run():
    LIFE = True  # for visuals
    D___ = False
    names = []
    init_flips = []
    rnd_flips = []

    names.append("static flip flop 4x4")
    init_flips.append(
        np.array([
            [D___, LIFE, D___, D___],
            [LIFE, D___, D___, LIFE],
            [D___, D___, D___, D___],
            [D___, D___, D___, D___],
        ]))
    rnd_flips.append(None)

    sizes = [(4, 4), (5, 10), (10, 10), (20, 20), (50, 50), (60, 80)]
    densities = [.25, .5, .75]
    for size in sizes:
        for density in densities:
            names.append("same random flip of size {}x{} and density {}".format(*size, density))
            init_flips.append(np.random.rand(*size)<density)
            rnd_flips.append(None)

            names.append("new random flip of size {}x{} and density {}".format(*size, density))
            init_flips.append(None)
            rnd_flips.append(size)
    experiments = zip(names, init_flips, rnd_flips)

    for experiment in experiments:
        name = experiment[0]
        rl = train(init_flip=experiment[1], size=experiment[2])
        summary(name, rl)


def train(init_flip=None, size=None, verbose=False):
    if init_flip is not None:
        size = init_flip.shape
    model = Model(size=size)
    problem = CgolProblem(model, init_flip=init_flip)
    rl = RL(problem, shape=[None, *(3*[problem.length]), None], verbose=verbose,
            mu=0.01,
            batches=25,
            batch_size=50,
            max_steps=1000,
            epsilon_init=1.0,
            epsilon_decay_factor=0.9,
            epsilon_min=0.01,
            )
    rl.train(iterations=100)
    return rl


def summary(header, rl, filename=False, full=False):
    print('-~'*30, end='-\n')
    print('Title:', header)
    if filename:
        prefix = 'result {}.dat'.format(header).replace(' ', '_').lower()
    else:
        prefix = None
    test_res, base_res = report(rl, prefix)
    print_report(test_res, base_res, full=full)


def print_report(test_res, base_res, full=False):
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


def report(rl, filename=None):
    model = rl.get_problem().model
    if filename:
        model.set_filename("train_{}".format(filename))
        model.save()

    trials = 10
    test_res = measure(rl, model, trials, random=False)
    if filename:
        model.set_filename("test_{}".format(filename))
        model.save()

    base_res = measure(rl, model, trials, random=True)
    if filename:
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
    run()
