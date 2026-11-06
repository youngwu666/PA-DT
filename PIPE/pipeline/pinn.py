import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn import metrics

from config_gpu import config_gpu
device = config_gpu(1)

def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))
    R2 = metrics.r2_score(true_label,pred_label)
    return MAE, MAPE, MSE, RMSE, R2

def get_logger(log_name='log.txt'):
    logger = logging.getLogger('mylogger')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - function:%(funcName)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name is not None:
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))
    
class Sin(nn.Module):

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):

    def __init__(self,input_dim=2,
                 output_dim=1,
                 layers_num=4,
                 hidden_dim=50,
                 droupout=0.2,
                 act = 'sigmoid'):
        super(MLP, self).__init__()

        assert layers_num >= 1
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.act = act

        self.layers = []   
        for i in range(layers_num):
            if i == 0:   
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(self.get_activation())
            elif i == layers_num-1:  
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:       
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(self.get_activation())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)   
        self._init()       

    def _init(self):      
        for layer in self.net:
            if isinstance(layer,nn.Linear): 
                nn.init.xavier_normal_(layer.weight)   

    def forward(self, x):      
        x = self.net(x)        
        return x
    
    def get_activation(self): 
        if self.act == 'sigmoid':
            return nn.Sigmoid()
        elif self.act == 'relu':
            return nn.ReLU()
        elif self.act == 'tanh':
            return nn.Tanh()
        elif self.act == 'Sin':
            return Sin()
        else:
            raise ValueError(f"Unsupported activation: {self.act}")

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act='relu', dropout=0.0):
        super(ResidualBlock, self).__init__()

        if act == 'sigmoid':
            activation = nn.Sigmoid()
        elif act == 'relu':
            activation = nn.ReLU()
        elif act == 'tanh':
            activation = nn.Tanh()
        elif act == 'Sin':
            activation = Sin()
        else:
            raise ValueError(f"Unsupported activation: {act}")
        
        self.block = nn.Sequential(
            
            nn.Linear(in_dim, out_dim),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
            
        )
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)  
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        return self.shortcut(x) + self.block(x)  
    
class MLPRes(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims,
                 droupout=0.2,
                 act='relu',
                ):
        super(MLPRes, self).__init__()
        
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        
        for out_dim in hidden_dims:
            self.blocks.append(
                ResidualBlock(in_dim, out_dim, act=act, dropout=droupout)
            )
            in_dim = out_dim 
        
        self.fc_out = nn.Linear(in_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.fc_out(x)

class Predictor(nn.Module):

    def __init__(self, input_dim = 40):
        
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 16),
            Sin(),
            nn.Linear(16,1)
        )
        self.input_dim = input_dim

    def forward(self,x):
        return self.net(x)

class Solution_u(nn.Module):

    def __init__(self, input_dim, output_dim, layers_num, hidden_dim, droupout, act):
        super(Solution_u, self).__init__()
        
        self.encoder = MLP(input_dim = input_dim,
                           output_dim = output_dim,
                           layers_num = layers_num,
                           hidden_dim = hidden_dim,
                           droupout = droupout,
                           act = act)
              
        self.predictor = Predictor(input_dim = output_dim)
        self._init_()

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight,mode='fan_in',nonlinearity='relu')
                
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

class AverageMeter(object):
    

    def __init__(self):
        self.reset() 

    
    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 

    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LR_Scheduler(object):
    def __init__(self, optimizer, 
                 warmup_epochs, warmup_lr, 
                 num_epochs, base_lr, 
                 final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

class PINN(nn.Module):
    
    def __init__(self, args):

        super(PINN, self).__init__()
        self.args = args
        
        
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()
        
        self.dat_typ = torch.float32
        
        self.MLP_in    = args.MLP_in
        self.MLP_out   = args.MLP_out
        self.MLP_hidden_dim   = args.MLP_hidden_dim    
        self.MLP_layers_num   = args.MLP_layers_num   
        self.MLP_droupout = args.MLP_droupout
        
        self.F_in    = args.F_in
        self.F_out   = args.F_out   
        self.F_hidden_dim = args.F_hidden_dim      
        self.F_layers_num = args.F_layers_num      
        self.F_droupout = args.F_droupout
        self.act = args.act
        self.act_F = args.act_F
        self.opt = args.opt
        self.r_seed  = args.r_seed
        self.random_seed(self.r_seed)

        self.train_data = None   
        
        self.solution_u = Solution_u(self.MLP_in, 
                                     self.MLP_out, 
                                     self.MLP_layers_num, 
                                     self.MLP_hidden_dim, 
                                     self.MLP_droupout, 
                                     self.act).to(device)
        
        
        
        
        
        
        

        hidden_dims = [32, 64, 128, 128, 128, 8]
        self.dynamical_F = MLPRes(self.F_in, 
                          self.F_out, 
                          hidden_dims,
                          self.F_droupout, 
                          self.act_F).to(device)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        
        self.alpha = self.args.alpha
        self.beta = self.args.beta  
        self.datap = self.args.datap      
        
        
        self.structure = [self.MLP_in] + (self.MLP_layers_num-1) * [self.MLP_hidden_dim] + [self.MLP_out]

        
        
        self.optimizer1 = self.opt_(self.solution_u, args.warmup_lr, self.opt)
        self.optimizer2 = self.opt_(self.dynamical_F, args.lr_F, self.opt)

        self.scheduler = LR_Scheduler(optimizer = self.optimizer1,
                                      warmup_epochs = args.warmup_epochs,
                                      warmup_lr = args.warmup_lr,
                                      num_epochs = args.epochs,
                                      base_lr = args.lr,
                                      final_lr = args.final_lr)
        self.iter = 0
        self.fold = 0
        print("************************************************************")
        print(">>>>> :", datetime.datetime.now())
        print(">>>>> ;")
        print("           :", self.r_seed)
        print("             :", self.dat_typ)
        print("            :", self.act)
        print("             :", self.opt)
        print("                 :", self.MLP_hidden_dim)
        print("                 :", self.MLP_layers_num)
        print("             :", self.structure)

    def random_seed(self, seed=1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def opt_(self, Network, lr, opt):
        if opt == "SGD":
            optimizer = optim.SGD(
                Network.parameters(),lr=lr
            )
        elif opt == "RMSprop":
            optimizer = optim.RMSprop(
                Network.parameters(),lr=lr
            )
        elif opt == "Adam":
            optimizer = optim.Adam(
                Network.parameters(),lr=lr
            )
        elif opt == "Adamax":
            optimizer = optim.Adamax(
                Network.parameters(),lr=lr
            )
        elif opt == "Nadam":
            optimizer = optim.NAdam(
                Network.parameters(),lr=lr
            )
        elif opt == "LBFGS":
            optimizer = optim.LBFGS(
                Network.parameters(), 
                lr=lr, 
                max_iter=500, 
                max_eval=500, 
                history_size=100,
                tolerance_grad=1e-7, 
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe"       
            )
        else:
            raise NotImplementedError(">>>>> opt_")
        return optimizer
    
    def forward(self, xt):
        
        xt = xt.to(device)
        xt.requires_grad = True

        x = xt[:,0:-1] 
        t = xt[:,-1:] 

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
            )[0]
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
            )[0]        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
            )[0]
        
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t],dim=1))
        f = u_t - F

        return u, f, u_t, u_tt


        
    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x,y) in enumerate(validloader):
                x = x.to(device)
                u = self.predict(x)
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
                
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        
        return mse.item()
    
    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x,y) in enumerate(testloader):
                x = x.to(device)
                u = self.predict(x)
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()  

    def _save_args(self):
        if self.args.log_dir is not None:
            
            
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")  
          
    def predict(self, xt):
        return self.solution_u(xt)   


    def adjust_alpha(self, epoch, total_epochs, alpha_min= 0.1, alpha_max= 0.5):
        progress = epoch / total_epochs
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * progress))

    def train_one_epoch(self, epoch, dataloader):
        
        self.train()

        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        
        for iter, (x, y) in enumerate(dataloader):
            
            x, y  = x.to(device), y.to(device)
            
            u, f, u_t, u_tt = self.forward(x)
            
            
            loss1 = self.datap * self.loss_func(u, y)

            
            f_target = torch.zeros_like(f)
            
            alpha = self.adjust_alpha(epoch, self.args.epochs)
            loss2 = alpha * self.loss_func(f, f_target)

            
            
            
            
            loss_mono = torch.relu(-u_t).sum()
            loss3 = self.beta * (loss_mono)

            
            loss = loss1 + loss2 + loss3
        
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())


        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg       

    def Train(self, fold, trainloader, validloader,testloader):
        
        self.fold = fold
        self.iter = 0
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        loss_history = []
        loss_1 = []
        loss_2 = []
        loss_3 = []
        valid = []
        
        
        for e in range(1, self.args.epochs + 1):
            
            early_stop += 1
            loss1, loss2, loss3 = self.train_one_epoch(e,trainloader)
            
            total_loss = loss1 + loss2 + loss3
 
            loss_history.append(total_loss)
            loss_1.append(loss1)
            loss_2.append(loss2)
            loss_3.append(loss3)
            
            current_lr = self.scheduler.step()
            
            info = '[Adam] Fold {}, epoch:{}, lr:{:.6f}, total loss:{:.6f}'.format(
                self.fold, e, current_lr, total_loss)     
            self.logger.info(info)
            
            if e % 1 == 0:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                valid.append(valid_mse)
                self.logger.info(info)            
            

            
            if valid_mse < min_valid_mse :    
                min_valid_mse = valid_mse
                early_stop = 0
                
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(true_label,pred_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                
                valid_true_label,valid_pred_label = self.Test(validloader)
                [_, _, _, _, valid_R2] = eval_metrix(true_label,pred_label)
                
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, f'true_label_fold{self.fold}.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, f'pred_label_fold{self.fold}.npy'), pred_label)
            

            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = ' epoch {}  (Fold {})'.format(e, self.fold)
                self.logger.info(info)
                break
  
        self.clear_logger()
        
        return min_valid_mse, loss_history, loss_1, loss_2, loss_3, valid, true_label, pred_label, valid_true_label,valid_pred_label,valid_R2
    