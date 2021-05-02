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
        
    def loadState(self, manager):
        self.epochs = manager.epochs + 1
        self.current_epoch = manager.current_epoch
        self.loss_list = manager.loss_list
        
    def begin(self):
        self.current_epoch = 0
        self.loss_list = []