import numpy as np
from cgolai.cgol import Model, CgolProblemV2
from cgolai.ai import RL


major_name = "v2/"
density = .2


def run():
    title = "v2"
    size = 10
    batches = 500
    rl = train(size=(size, size), verbose=True, batches=batches, batch_size=30)
    rl.save(major_name+title+"_nn_model.dat")
    summary(title, rl, full=True, trials=10)


def train(size=(4, 4), verbose=False, batches=50, batch_size=20):
    model = Model(size=size)
    problem = CgolProblemV2(model, init_flip=None, density=density, high_density=1.0-density, pop_record_size=10)
    layers = [problem.length+problem.key_dim, problem.length + problem.cols + problem.rows, problem.length // 2]
    rl = RL(problem, shape=[None, *layers, None], verbose=verbose,
            mu=0.01,
            batches=batches,
            batch_size=batch_size,
            max_steps=250,
            epsilon_init=1.0,
            epsilon_decay_factor=0.9,
            epsilon_min=0.01,
            #replay_count=2,
            )
    rl.train(iterations=100)
    return rl


def summary(title, rl, full=False, trials=5):
    print('-~'*30, end='-\n')
    print('Title:', title)
    prefix = '{}{} result {}.dat'.format(major_name, '{}', title).replace(' ', '_').lower()
    test_res, base_res = report(rl, prefix, trials=trials)
    print_report(test_res, base_res, full=full)


def report(rl, filename=None, trials=10):
    model = rl.get_problem().get_model()
    if filename:
        model.set_filename(filename.format('train'))
        model.save()

    init_flips = [(np.random.rand(*model.size) < density) for _ in range(trials)]

    test_res = measure(rl, init_flips, random=False)
    if filename:
        model.set_filename(filename.format('rlagent'))
        model.save()

    base_res = measure(rl, init_flips, random=True)
    if filename:
        model.set_filename(filename.format('control'))
        model.save()
    return test_res, base_res


def print_report(test_res, base_res, full=False):
    print('control')
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
            print('control trial:', i, end='; ')
            print('steps:', steps, end='; ')
            print('reward:', reward, end='; ')
            print('maxout:', maxout)
    print('total steps:', total_steps)
    print('total rewards:', total_rewards)
    print('total maxouts:', total_maxouts)

    print()
    print('rl agent')
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


def measure(rl, init_flips, random=False, max_length=500):
    """
    :param rl: The trained rl model
    :param model: The model
    :param init_flips: number of trials
    :param random: random moves or use rl's moves
    :param max_length: max length before stopping trial
    :return: list containing results from trial. results are (steps, list of rewards, did maxout)
    """
    problem = rl.get_problem()
    model = problem.get_model()
    model.load_iter(0)
    trial_results = []
    for init_flip in init_flips:
        problem.reset(init_flip)
        steps = 0
        rewards = []
        while not problem.is_terminal() and steps < max_length:
            steps += 1
            action, _ = rl.choose_best_action(explore=random, epsilon=1.0)
            _, reward = problem.do(action)
            rewards.append(reward)
        trial_results.append((steps, rewards, not problem.is_terminal()))
    return trial_results


if __name__ == '__main__':
    run()
