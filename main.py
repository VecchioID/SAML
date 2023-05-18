# -*- coding:utf-8 -*-
# __author__ = 'Vecchio'

# Add models and tasks to path
import sys
sys.path.insert(0, './models')
sys.path.insert(0, './tasks')
sys.path.insert(0, './svrt-1k')

import argparse
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import log
from Model_class import Solver
from net_class import seq_dataset, check_path

saved_model_dir = './saved_models/'
check_path(saved_model_dir)

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Add models and tasks to path
sys.path.insert(0, './models')
sys.path.insert(0, './tasks')

parser = argparse.ArgumentParser(description="A model for discrete reasoning")
parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=12346)
parser.add_argument('--load_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--multi_gpu', type=bool, default=False)  # choose if to use multi-gpu
parser.add_argument('--test_every', type=int, default=5)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--trn_configs', nargs='+', type=str, default="*")
parser.add_argument('--tst_configs', nargs='+', type=str, default="*")
parser.add_argument('--silent', type=bool, default=False)
parser.add_argument('--shuffle_first', type=bool, default=False)
parser.add_argument('--checkpoint', type=str, default="./saved_models/")
parser.add_argument('--n_shapes', type=int, default=100, help="n = total number of shapes available for training and testing")
parser.add_argument('--train_set_size', type=int, default=10000)
parser.add_argument('--test_set_size', type=int, default=10000)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)

parser.add_argument('--train_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
parser.add_argument('--test_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")

parser.add_argument('--m_holdout', type=int, default=95, help="m = number of objects (out of n) withheld during training")
parser.add_argument('--model_name', type=str, default='ESBN', help="{'SAML', 'ESBN', 'MAREO', 'CoRelNet', 'NTM'}")
parser.add_argument('--task', type=str, default='dist3', help="{'same_diff', 'RMTS', 'dist3', 'identity_rules'}")
parser.add_argument('--norm_type', type=str, default='contextnorm', help="{'nonorm', 'contextnorm', 'tasksegmented_contextnorm'}")
parser.add_argument('--log_txt', type=str, default='output.txt')
parser.add_argument('--encoder', type=str, default='conv', help="{'conv', 'mlp', 'rand', 'vit'}")

# time steps
parser.add_argument('--step', type=int, default=16)
# Run number
parser.add_argument('--run', type=str, default='0')

args = parser.parse_args()
log.info('begin tackle data...')
log_txt = open(args.log_txt, mode="a", encoding="utf-8")

# Randomly assign objects to training or test set
all_shapes = np.arange(args.n_shapes)
np.random.shuffle(all_shapes)
if args.m_holdout > 0:
    train_shapes = all_shapes[args.m_holdout:]
    test_shapes = all_shapes[:args.m_holdout]
else:
    train_shapes = all_shapes
    test_shapes = all_shapes
# Generate training and test sets
task_gen = __import__(args.task)
log.info('Generating task: ' + args.task + '...')
args, train_set, test_set = task_gen.create_task(args, train_shapes, test_shapes)
# Convert to PyTorch DataLoaders
train_set = seq_dataset(train_set, args)
trn_ldr = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
test_set = seq_dataset(test_set, args)
tst_ldr = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

# Load images
all_imgs = []
for i in range(args.n_shapes):
    img_fname = './imgs/' + str(i) + '.png'
    img = torch.Tensor(np.array(Image.open(img_fname))) / 255.
    all_imgs.append(img)
all_imgs = torch.stack(all_imgs, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

# initialise model
model = Solver(args, task_gen).to(device)


# if parameters of this model is needed
# from torchsummary import summary
# summary(model, input_size=(16, 80, 80))

# define train, validate, and test functions
def train(epoch):
    model.train()
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    enum = enumerate(trn_ldr) if args.silent else tqdm(enumerate(trn_ldr))
    for batch_idx, (seq_idx, target) in enum:
        counter += 1
        image = all_imgs[seq_idx, :, :]
        image = image.to(device)
        target = target.to(device)
        loss, acc = model.train_(image, target)
        loss_all += loss
        acc_all += acc
    if not args.silent:
        print("Epoch {}: Avg Training Acc: {:.4f}, Loss: {:.6f}".format(epoch, acc_all / float(counter), loss_all / float(counter)), file=log_txt)
        log.info("Epoch {}: Avg Training Acc: {:.4f}, Loss: {:.6f}".format(epoch, acc_all / float(counter), loss_all / float(counter)))

    # 一个epoch保存一次
    check_path('./results/')
    train_prog_dir = './results/train_prog/'
    task_dir = train_prog_dir + args.task + '/'
    check_path(task_dir)
    model_dir = task_dir + args.model_name + '/'
    check_path(model_dir)
    run_dir = model_dir + '/m' + str(args.m_holdout) + '/' + 'run' + args.run + '/'
    check_path(run_dir)
    train_prog_fname = run_dir + 'epoch_' + str(epoch) + '.txt'
    train_prog_f = open(train_prog_fname, 'w')
    train_prog_f.write('epoch loss acc\n')
    # Save progress to file
    train_prog_f.write(str(epoch) + ' ' + \
                       '{:.4f}'.format(loss_all / float(counter)) + ' ' + \
                       '{:.2f}'.format(acc_all / float(counter)) + '\n')
    train_prog_f.close()

    return loss_all / float(counter)


def test(epoch):
    model.eval()
    acc_overall = 0
    start = time.time()
    acc_all = 0.0
    counter = 0
    for batch_idx, (seq_idx, target) in enumerate(tst_ldr):
        counter += 1
        image = all_imgs[seq_idx, :, :]
        image = image.to(device)
        target = target.to(device)
        acc_all += model.test_(image, target)
    average = acc_all / float(counter)
    end = time.time()
    if not args.silent:
        print("Total Test acc: {:.4f}. Tested in {:.2f} seconds.".format(average, end - start), file=log_txt)
        log.info("Total Test acc: {:.4f}. Tested in {:.2f} seconds.".format(average, end - start))
    acc_overall = acc_all
    average_overall = acc_overall / len(tst_ldr)
    if not args.silent:
        print("\nAverage acc: {:.4f}\n".format(average_overall), file=log_txt)

    # Save performance
    test_dir = './results/test/'
    check_path(test_dir)
    task_dir = test_dir + args.task + '/'
    check_path(task_dir)
    model_dir = task_dir + args.model_name + '/m' + str(args.m_holdout) + '/' + 'run' + args.run + '/'
    check_path(model_dir)

    # saving test results
    test_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
    test_f = open(test_fname, 'w')
    test_f.write('acc\n')
    test_f.write('{:.2f}'.format(average_overall))
    test_f.close()

    return average_overall


def main():
    lo_trn_los = 10
    lo_val_los = 10
    hi_val_acc = 10
    hi_tst_acc = 10

    prog_start = time.time()
    for epoch in range(0, args.epochs):
        tl = train(epoch)
        lo_trn_los = tl if tl < lo_trn_los else lo_trn_los
        if not epoch % args.test_every and epoch != 0:
            ta = test(epoch)
            if ta > hi_tst_acc:
                hi_tst_acc = ta
                model.save_model('./saved_models/', epoch, hi_tst_acc, loss=0)
            else:
                hi_tst_acc = hi_tst_acc

    prog_end = time.time()
    print("Training completed in {:.2f} minutes.".format((prog_end - prog_start) / 60), file=log_txt)
    log.info("Training completed in {:.2f} minutes.".format((prog_end - prog_start) / 60))
    print("\nlo_trn_los: {:.4f}\nhi_tst_acc: {:.4f}\n".format(lo_trn_los, hi_tst_acc), file=log_txt)
    log.info("\nlo_trn_los: {:.4f}\nhi_tst_acc: {:.4f}\n".format(lo_trn_los, hi_tst_acc))


if __name__ == '__main__':
    main()
