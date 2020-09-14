import numpy as np
import torch
import sys
sys.path.append('..')

import config as flags
from utils import *
import torch.nn as nn
from mi_fgsm import MI_FGSM_ENS
import argparse
import math

"""
    对代码进行了一些整理，不能指定target类，target攻击是会读取之前保存的target类
"""


class LinfMGA():

    def __init__(self, model, pop_size=5, generations=1000, cross_rate=0.7,
                 mutation_rate=0.001, max_queries=2000,
                 epsilon=8. / 255, ensemble_models=None, iters=10, targeted=False):

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        # parameters about evolution algorithm
        self.pop_size = pop_size
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

        # parameters about attack
        self.epsilon = epsilon

        self.clip_min = 0
        self.clip_max = 1

        # ensemble MI-FGSM parameters, use ensemble MI-FGSM attack generate adv as initial population
        self.ensemble_models = ensemble_models
        self.iters = iters
        self.targeted = targeted
        self.max_queries = max_queries

    def is_success(self, logits, y):

        label = logits.argmax(dim=1).item()
        if self.targeted and label == y:
            return True
        if self.targeted == False and label != y:
            return True

        return False

    def fitness_helper(self, individual, x, y):

        # resize to image size
        individual = individual.copy()
        zeros = (individual == 0)
        individual[zeros] = -1

        delta = individual * self.epsilon
        delta = torch.from_numpy(delta).to(dtype=x.dtype, device=x.device)

        adv = x + delta.unsqueeze(0)
        adv = torch.clamp(adv, self.clip_min, self.clip_max)

        # only imagenet dataset needs preprocess
        logits = predict(self.model, torch2numpy(adv), logits=True)

        loss = self.loss_fn(logits, y)
        self.query += 1

        if self.is_success(logits, y):
            self.adv = adv.cpu()

        return loss.item()

    def get_fitness(self, lw, x, y):
        first, second = self.idx[0], self.idx[1]
        if self.is_change[first] == 1:
            f1 = self.fitness_helper(lw[0], x, y)
            self.pop_fitness[first] = f1
            self.is_change[first] = 0
        else:
            f1 = self.pop_fitness[first]

        if self.is_change[second] == 1:
            f2 = self.fitness_helper(lw[1], x, y)
            self.pop_fitness[second] = f2
            self.is_change[second] = 0
        else:
            f2 = self.pop_fitness[second]

        return np.array([f1, f2])

    def cross_over(self, lw):
        cross_point = np.random.rand(self.channel, self.image_size, self.image_size) < self.cross_rate
        lw[0, cross_point] = lw[1, cross_point]
        return lw

    def mutate(self, lw):

        # generate mutation point
        mutation_point = np.random.rand(self.channel, self.image_size, self.image_size) < self.mutation_rate

        # reverse the value at mutation point 1->0, 0->1
        zeros = (lw[0] == 0)
        ones = (lw[0] == 1)
        lw[0, mutation_point & zeros] = 1
        lw[0, mutation_point & ones] = 0
        return lw

    def init_pop(self, x, y):

        adversary = MI_FGSM_ENS(self.ensemble_models, epsilon=self.epsilon,
                                iters=self.iters, targeted=self.targeted)

        datas, labels = x.repeat((self.pop_size, 1, 1, 1)), y.repeat(self.pop_size)
        adv = adversary.perturb(datas, labels)

        delta = adv - x
        negative = delta <= 0
        positive = delta > 0
        delta[negative] = 0
        delta[positive] = 1

        return delta.cpu().numpy()

    def perturb(self, x, y):

        x, y = x.to(flags.device), y.to(flags.device)

        # input train_data parameter
        self.batch_size, self.channel, self.image_size, _ = x.shape
        self.adv = None
        self.query = 0
        self.pop_fitness = np.zeros(self.pop_size)
        self.is_change = np.zeros(self.pop_size)

        if self.ensemble_models is None:
            # initial population
            pop = np.random.randint(0, 2, (self.pop_size, self.channel, self.image_size, self.image_size))
        else:
            pop = self.init_pop(x, y)

        # this expense 5 queries, this thy the median always 5
        # init pop fitness, this can reduce query, cause in mga, not all individual changes in a generation
        for n in range(self.pop_size):
            self.pop_fitness[n] = self.fitness_helper(pop[n], x, y)

        for i in range(self.generations):

            self.idx = np.random.choice(np.arange(self.pop_size), size=2, replace=False)
            lw = pop[self.idx].copy()  # short for losser winner

            fitness = self.get_fitness(lw, x, y)

            # if success, abort early
            if self.adv is not None:
                return self.adv, self.query

            # in target situation, the smaller fitness is, the better
            if self.targeted:
                fidx = np.argsort(-fitness)
            else:
                fidx = np.argsort(fitness)

            lw = lw[fidx]
            lw = self.cross_over(lw)
            lw = self.mutate(lw)

            lw = lw[fidx]
            # update population
            pop[self.idx] = lw.copy()

            # losser changed, so fitness also change
            self.is_change[self.idx[fidx[0]]] = 1
            if self.query >= self.max_queries:
                return None, self.query

        return None, self.query


def attack_cifar10(args, model, models):
    val_loader = Cifar10Loader(1).get_test_loader()
    adversary = LinfMGA(model, pop_size=args.pop_size, generations=50000, cross_rate=args.cr, targeted=args.targeted,
                        mutation_rate=args.mr, max_queries=args.max_queries,
                        epsilon=args.epsilon, iters=args.iters, ensemble_models=models)
    return adversary, val_loader


def attack_imagenet(args, model, models):

    val_loader = get_imagenet_val_loader(flags.imagenet12_valk, 1, image_size=299)
    adversary = LinfMGA(model, pop_size=args.pop_size, generations=50000, cross_rate=args.cr, targeted=args.targeted,
                        mutation_rate=args.mr, max_queries=args.max_queries,
                        epsilon=args.epsilon, iters=args.iters, ensemble_models=models)

    return adversary, val_loader


if __name__ == '__main__':
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_models', type=str, nargs="+",
                        default=['vgg19_bn_cifar10'])
    parser.add_argument('--model', type=str, default='resnet50_cifar10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epsilon', type=float, default=0.03137)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--pop_size', type=int, default=5)
    parser.add_argument('--num_attack', type=int, default=2000)
    parser.add_argument('--mr', type=float, default=0.001)
    parser.add_argument('--cr', type=float, default=0.7)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--targeted', default=False, action='store_true')
    parser.add_argument('--defense_method', default=None, type=str, choices=["RP", 'JPEG', 'BitDepthReduce'])

    args = parser.parse_args()

    start = time.time()

    # initial model
    models = []
    if args.ensemble_models == "":
        models = None
    else:
        for m in args.ensemble_models:
            models.append(init_model(m))

    # defense method
    if args.defense_method:
        defense_method = args.defense_method+"()"
        input_trans = eval(defense_method)
    else:
        input_trans = None

    model = init_model(args.model)

    # imagenet use val dataset, mnist and cifar10 use the test dataset
    if args.dataset == 'cifar10':
        model = InputTransformModel(model, normalize=(flags.cifar10_mean, flags.cifar10_std), input_trans=input_trans)
        adversary, val_loader = attack_cifar10(args, model, models)
    elif args.dataset == 'imagenet':

        if args.model == "inceptionv3":
            model = InputTransformModel(model, normalize=(flags.tf_mean, flags.tf_std), input_trans=input_trans)
        else:
            model = InputTransformModel(model, normalize=(flags.torch_mean, flags.torch_std),
                                            input_trans=input_trans)

        adversary, val_loader = attack_imagenet(args, model, models)
    else:
        pass

    # log process
    log = str(args.targeted) + '_' + str(args.model)+'_mr_'+str(args.mr) + '.log'
    f = open(log, 'w')

    if args.targeted:
        if args.dataset == 'imagenet':
            target_class = np.load('imagenet_target_class.npy')
        else:
            target_class = np.load('cifar10_mnist_target_class.npy')

    success_num = total_correct = total_queries = 0
    query_list = []
    for i, (data, label) in enumerate(val_loader):

        if predict(model, data.numpy()).cpu() == label:

            # same class skip
            if args.targeted and target_class[i] == label.item():
                continue

            one_start = time.time()
            # Run attack
            if args.targeted:
                adv_img, num_queries = adversary.perturb(data, torch.tensor([target_class[i]]))
            else:
                adv_img, num_queries = adversary.perturb(data, label)

            if adv_img is not None:
                # Check if the adversarial image satisfies the constraint
                assert torch.max(torch.abs((adv_img.cpu() - data))).item() <= args.epsilon + 1e-3
                assert adv_img.max().item() <= 1. + 1e-3
                assert adv_img.min().item() >= 0. - 1e-3
                success_num += 1
                total_queries += num_queries
                query_list.append(num_queries)

            total_correct += 1
            print("index:{}, queries:{}, time:{}".format(i, num_queries, time.time()-one_start), file=f)

        f.flush()

        if i>=args.num_attack:
            break

        torch.cuda.empty_cache()

    f.close()

    print(success_num)
    print(total_correct)
    print(query_list)
    query_list = np.array(query_list)
    print('success rate:{}, average queries:{}, median:{}'.format(success_num / total_correct, total_queries / success_num, np.median(query_list)))
    print('max_queries:{}, mutation rate:{}, crossover rate:{},'
          ' ensemble_models:{}, target model:{}, targeted:{}, dataset:{}, epsilon:{}'.format(
        args.max_queries, args.mr, args.cr, args.ensemble_models,
        args.model, args.targeted, args.dataset, args.epsilon))

    print('time:{}'.format(time.time() - start))
