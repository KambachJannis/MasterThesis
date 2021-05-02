import os

class Manager:
    def __init__(self, exp_id, save_path):
        self.id = exp_id
        self.save_path = os.path.join(save_path, exp_id)
        
        self.dataset_type = None
        self.batch_size_train = None
        self.batch_size_val = None
        self.n_classes = None
        self.transform = None
        
        self.net_name = None
        self.opt_name = None
        self.loss_name = None

        self.adam_learning_rate = None
        self.adam_betas = None
        self.adam_decay = None
        
        self.epochs = None
        self.current_epoch = None
        self.loss_list = None
        self.best_loss = None
        
class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum 
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts