import os
import copy
import time
import numpy as np

import json
import torch
from tensorboardX import SummaryWriter
from utils import get_flat_model_params
import torch.nn.functional as F

from options import args_parser
from update import PerLocalUpdate
from models import MLP, CNN, CNN1, CNN2
from utils import get_dataset, exp_details, setup_seed, average_loss_acc



if __name__ == '__main__':
    seeds = [1, 10, 100, 1000, 10000]

    for seed in seeds:

        start_time = time.time()

        path_project = os.path.abspath('.')
        logger = SummaryWriter('../logs')

        args = args_parser()
        exp_details(args)

        setup_seed(seed)
        print('random seed =', seed)
        # torch.cuda.set_device(1)

        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, user_groups = get_dataset(args)
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            input_size = 784
            num_classes = 10
        local_model, model = [], []
        if args.model == 'CNN':
            global_model = CNN()
        elif args.model == 'MLP':
            global_model = MLP(input_size, num_classes)
        elif args.model == 'CNN1':
            global_model = CNN1()
        elif args.model == 'CNN2':
            global_model = CNN2()
        else:
            exit('Error: unrecognized model')

        global_model.to(args.device)
        global_weights = global_model.state_dict()
        person_model_dim = sum(p.numel() for p in global_model.parameters())
        Proj = torch.normal(mean=0, std=1., size=(10, person_model_dim))
        Proj = F.normalize(Proj, p=2, dim=1).to(args.device)
        if args.framework == 'lp-proj-2' or args.framework == 'FLAME-lp-proj-2':
            flat_w = get_flat_model_params(global_weights)
            global_weights = torch.matmul(Proj, flat_w)

        else:
            global_model.train()
        print(global_model)
        w = copy.deepcopy(global_weights)

        total_params = sum(p.numel() for p in global_model.parameters())
        print(f"Total number of parameters: {total_params}")
        train_loss_personal, test_acc_personal, train_loss_global, test_acc_global = [], [], [], []
        train_loss_hybrid, test_acc_hybrid, test_loss_hybrid = [], [], []
        test_acc_personal_variances, test_acc_global_variances = [], []
        test_loss_personal, test_loss_global = [], []
        test_loss_personal_variances, test_loss_global_variances = [], []
        test_acc_hybrid_variances, test_loss_hybrid_variances = [], []
        train_loss_personal_local, train_loss_global_local = 0, 0
        malicious_users = []
        if args.corrupted == '1' or args.corrupted == '2' or args.corrupted == '3' or args.corrupted == '4':
            malicious_users = np.random.choice(range(args.num_users), max(int(args.num_malicious * args.num_users), 1), replace=False)
        for idx in range(args.num_users):
            local_model.append(
                PerLocalUpdate(args=args, global_model=global_model, dataset=dataset, idxs=user_groups[idx],
                               logger=logger, w=w, Proj=Proj, user_id=idx, malicious_users=malicious_users))

        (train_loss_hybrid_avg, test_acc_hybrid_avg, test_loss_hybrid_avg, train_loss_global_avg,
         train_loss_personal_avg, test_acc_personal_avg, test_acc_global_avg, test_loss_personal_avg,
         test_loss_global_avg, test_acc_personal_variance, test_acc_global_variance, test_loss_personal_variance,
         test_loss_global_variance, test_acc_hybrid_variance, test_loss_hybrid_variance) = average_loss_acc(local_model,
                                                                                                            args.num_users,
                                                                                                            malicious_users)

        train_loss_hybrid.append(train_loss_hybrid_avg)
        train_loss_personal.append(train_loss_personal_avg)
        train_loss_global.append(train_loss_global_avg)

        test_acc_personal.append(test_acc_personal_avg)
        test_acc_global.append(test_acc_global_avg)
        test_acc_hybrid.append(test_acc_hybrid_avg)

        test_loss_personal.append(test_loss_personal_avg)
        test_loss_global.append(test_loss_global_avg)
        test_loss_hybrid.append(test_loss_hybrid_avg)

        test_acc_personal_variances.append(test_acc_personal_variance)
        test_acc_global_variances.append(test_acc_global_variance)
        test_acc_hybrid_variances.append(test_acc_hybrid_variance)

        test_loss_personal_variances.append(test_loss_personal_variance)
        test_loss_global_variances.append(test_loss_global_variance)
        test_loss_hybrid_variances.append(test_loss_hybrid_variance)


        for epoch in range(args.epochs):

            local_sum, local_train_losses_personal, local_test_accuracies_personal, \
            local_test_accuracies_global, local_train_losses_global = [], [], [], [], []

            m1 = max(int(args.frac_candidates * args.num_users), 1)
            m2 = max(int(args.frac * args.num_users), 1)


            global_model.train()
            idxs_candidates_users = np.random.choice(range(args.num_users), m1, replace=False)

            # Client selection:
            if args.strategy == 'random':
                idxs_users = np.random.choice(idxs_candidates_users, m2, replace=False)
            elif args.strategy == 'full':
                idxs_users = range(args.num_users)
            else:
                exit('Error: unrecognized client selection strategy.')



            print(f"\n\x1b[{35}m{'------------------------------------------   The IDs of selected clients: {}   ------------------------------------------'.format(np.sort(idxs_users))}\x1b[0m")






            lr = args.lr
            # Update personalized and global models
            for idx in idxs_users:
                local_model[idx].update_weights(global_round=epoch, global_model=global_model, w=copy.deepcopy(w), UserID=idx, lr=lr, malicious_users=malicious_users)


            if args.framework == 'FLAME':
                for key in w.keys():
                    w[key] = torch.zeros_like(w[key])
                    for i in range(0, len(local_model)):
                        w[key] += (local_model[i].wi[key] + (1 / args.rho) * local_model[i].alpha[key]) * 1 / args.num_users
            elif args.framework == 'pFedMe':
                for key in w.keys():
                    w[key] = torch.zeros_like(w[key])
                    for i in range(0, len(local_model)):
                        w[key] += local_model[i].wi[key] * 1 / args.num_users
            elif args.framework == 'FLAME-lp-proj-2':
                w = torch.zeros_like(w)
                for i in range(0, len(local_model)):
                    w += (local_model[i].wi + (1 / args.rho) * local_model[i].alpha) * 1 / args.num_users

            elif args.framework == 'lp-proj-2':
                w = torch.zeros_like(w)
                for i in range(0, len(local_model)):
                    w += local_model[i].wi * 1 / args.num_users
            elif args.framework == 'ditto':
                for key in w.keys():
                    w[key] = torch.zeros_like(w[key])
                    for i in range(0, len(local_model)):
                        w[key] += local_model[i].wi[key] * 1 / args.num_users

            if args.framework == 'lp-proj-2' or args.framework == 'FLAME-lp-proj-2':
                pass
            else:
                global_model.load_state_dict(w)

            (train_loss_hybrid_avg, test_acc_hybrid_avg, test_loss_hybrid_avg, train_loss_global_avg, train_loss_personal_avg, test_acc_personal_avg, test_acc_global_avg, test_loss_personal_avg,
             test_loss_global_avg, test_acc_personal_variance, test_acc_global_variance,test_loss_personal_variance, test_loss_global_variance, test_acc_hybrid_variance, test_loss_hybrid_variance) = average_loss_acc(local_model, args.num_users, malicious_users)

            train_loss_hybrid.append(train_loss_hybrid_avg)
            train_loss_personal.append(train_loss_personal_avg)
            train_loss_global.append(train_loss_global_avg)

            test_acc_personal.append(test_acc_personal_avg)
            test_acc_global.append(test_acc_global_avg)
            test_acc_hybrid.append(test_acc_hybrid_avg)


            test_loss_personal.append(test_loss_personal_avg)
            test_loss_global.append(test_loss_global_avg)
            test_loss_hybrid.append(test_loss_hybrid_avg)

            test_acc_personal_variances.append(test_acc_personal_variance)
            test_acc_global_variances.append(test_acc_global_variance)
            test_acc_hybrid_variances.append(test_acc_hybrid_variance)


            test_loss_personal_variances.append(test_loss_personal_variance)
            test_loss_global_variances.append(test_loss_global_variance)
            test_loss_hybrid_variances.append(test_loss_hybrid_variance)
            print(
                f"\n\x1b[{34}m{'>>> Round: {} / Hybrid model / Test accuracy: {:.2f}% / Training loss: {:.4f} / Test loss: {:.4f} / Test accuracy variance: {:.6f} / Test loss variance: {:.5f}'.format(epoch, 100 * test_acc_hybrid_avg, train_loss_hybrid_avg, test_loss_hybrid_avg, test_acc_hybrid_variance, test_loss_hybrid_variance)}\x1b[0m")
            print(
                f"\x1b[{34}m{'>>> Round: {} / Personal model / Test accuracy: {:.2f}% / Training loss: {:.4f} / Test loss: {:.4f} / Test accuracy variance: {:.6f} / Test loss variance: {:.5f}'.format(epoch, 100 * test_acc_personal_avg, train_loss_personal_avg, test_loss_personal_avg, test_acc_personal_variance, test_loss_personal_variance)}\x1b[0m")
            print(
                f"\x1b[{34}m{'>>> Round: {} / Global model / Test accuracy: {:.2f}% / Training loss: {:.4f} / Test loss: {:.4f} / Test accuracy variance: {:.6f} / Test loss variance: {:.5f}'.format(epoch, 100 * test_acc_global_avg, train_loss_global_avg, test_loss_global_avg, test_acc_global_variance, test_loss_global_variance)}\x1b[0m")


        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


        output = {}
        output['dataset'] = args.dataset
        output['framework'] = args.framework
        output['num_users'] = args.num_users
        output['seed'] = seed
        output['rho'] = args.rho
        output['Lambda'] = args.Lambda
        output['local_ep'] = args.local_ep
        output['training_loss_hybrid'] = train_loss_hybrid
        output['training_loss_personal'] = train_loss_personal
        output['training_loss_global'] = train_loss_global


        output['test_acc_personal'] = test_acc_personal
        output['test_acc_global'] = test_acc_global
        output['test_acc_hybrid'] = test_acc_hybrid

        output['test_loss_hybrid'] = test_loss_hybrid
        output['test_loss_personal'] = test_loss_personal
        output['test_loss_global'] = test_loss_global

        output['test_acc_personal_variances'] = test_acc_personal_variances
        output['test_acc_global_variances'] = test_acc_global_variances
        output['test_acc_hybrid_variances'] = test_acc_hybrid_variances



        output['test_loss_personal_variances'] = test_loss_personal_variances
        output['test_loss_global_variances'] = test_loss_global_variances
        output['test_loss_hybrid_variances'] = test_loss_hybrid_variances

        output_file = '../save/{}_{}_{}_{}_users_{}_rho_{}_lambda_{}_epoch_{}_partition_{}_q_{}_attack_{}_num_malicious_{}.json'.format(args.dataset,
                                                                                                          args.framework,
                                                                                                          args.strategy,
                                                                                                          seed,
                                                                                                          args.num_users,
                                                                                                          args.rho,
                                                                                                          args.Lambda,
                                                                                                          args.local_ep,
                                                                                                          args.partition,
                                                                                                          args.q,
                                                                                                          args.corrupted,
                                                                                                          args.num_malicious)
        with open(output_file, "w") as dataf:
            json.dump(output, dataf)





