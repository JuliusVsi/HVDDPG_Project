import torch
import numpy as np
import time


def timestr(form=None):
    if form is None:
        return time.strftime('<%Y-%m-%d  %H:%M:%S>', time.localtime())
    if form == 'mdhm':
        return time.strftime('<%m-%d  %H:%M>', time.localtime())


def batch_loader(x_set, y, batch_size, shuffle=True, drop_last=False):
    if shuffle:
        shuffled_index = np.random.permutation(len(x_set))
    else:
        shuffled_index = np.arange(len(x_set))

    mini_batch_index = 0
    num_remain = len(x_set) - batch_size
    while num_remain >= 0:
        index = shuffled_index[mini_batch_index: (mini_batch_index + batch_size)]
        mini_batch_index += batch_size
        num_remain -= batch_size
        yield x_set[index], y[index]

    if not drop_last:
        if mini_batch_index < x_set.shape[0]:
            index = shuffled_index[mini_batch_index:]
            yield x_set[index], y[index]


class BNNTrainer(object):
    def __init__(self, bnn_model, train_set, batch_size, optimizer, max_epoch,
                 log_cfg, extra_w_kl=1.0, test_set=None, extra_arg=dict()):
        self.model = bnn_model
        self.train_set = train_set
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.test_set = test_set
        self.device = 'cpu'
        self.length = len(train_set[0])
        self.max_epoch = max_epoch
        self.log_cfg = log_cfg
        self.weight_kl = batch_size / self.length * extra_w_kl

        self.external_criterion = extra_arg.get('external_criterion', None)
        self.external_criterion_val = extra_arg.get('external_criterion_val', None)

        self.start_epoch = 1
        self.model.to(self.device)

    def train(self):
        loss_all = []
        metric_train, kl_train = 0, 0

        for epoch in range(1, self.max_epoch + 1):
            loss_all.append(self.train_epoch())

            if epoch % self.log_cfg['display_interval'] == 0 or epoch == self.start_epoch:
                display_num = self.log_cfg['display_interval']
                loss_avg = np.array(loss_all[-display_num:]).mean()
                print('%s Epoch = %d : loss = %.7f, learning_rate = %.7e'
                      % (timestr(), epoch, loss_avg, self._get_lr()))

            if epoch % self.log_cfg['val_interval'] == 0 or epoch == self.start_epoch:
                metric_train, kl_train = self.validation(self.train_set, is_prob=False)
                print('%s Epoch = %d : metric_train = %.7f, kl_train = %.7f'
                      % (timestr(), epoch, metric_train, kl_train))
                if self.test_set:
                    metric_val, kl_val = self.validation(self.test_set, is_prob=False)
                    print('%s Epoch = %d : metric_train = %.7f, kl_train = %.7f'
                          % (timestr(), epoch, metric_val, kl_val))
        loss_mean = np.mean(np.array(loss_all))
        return loss_mean, metric_train, kl_train

    def train_epoch(self):
        self.model.train()      # ??? model has no train function
        loss_arr = []
        for inputs, targets in batch_loader(self.train_set[0], self.train_set[1], self.batch_size):
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets,  dtype=torch.float32)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            predictions = self.model.forward(inputs)      # forward???

            self.optimizer.zero_grad()
            if self.external_criterion:
                loss = self.external_criterion(predictions, targets) + self.weight_kl * self.model.kl_new_prior()
            else:
                loss = self.model.loss(predictions, targets, self.weight_kl)
            loss.backward()
            self.optimizer.step()
            loss_arr.append(loss.item())

        return np.array(loss_arr).mean()

    def validation(self, dataset, is_prob=True):
        self.model.eval()      # ??????
        if dataset is None:
            raise RuntimeError('Data set is empty.')

        metric_arr = []
        kl_arr = []

        with torch.no_grad():
            for inputs, targets in batch_loader(dataset[0], dataset[1], self.batch_size, shuffle=False):
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    targets = torch.tensor(targets, dtype=torch.float32)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model.forward(inputs, is_prob)  # forward???

                if self.external_criterion_val:
                    metric = \
                        self.external_criterion_val(predictions, targets)  # + self.weight_kl*self.model.kl_new_prior()
                elif self.external_criterion:
                    metric = \
                        self.external_criterion(predictions, targets)  # + self.weight_kl * self.model.kl_new_prior()
                else:
                    metric = self.model.loss(predictions, targets, 0.0)

                kl = self.model.kl_new_prior()
                metric_arr.append(metric.item())
                kl_arr.append(kl.item())

        metric_mean = np.mean(np.array(metric_arr))
        kl_mean = np.mean(np.array(kl_arr))

        return metric_mean, kl_mean

    def inference(self, inputs, is_prob=True):
        with torch.no_grad():
            predictions = self.model.forward(inputs, is_prob)
        return predictions

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
