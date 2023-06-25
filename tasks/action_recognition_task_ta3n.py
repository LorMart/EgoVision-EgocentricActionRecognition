from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger

from typing import Dict, Tuple


class ActionRecognition(tasks.Task, ABC):
    """Action recognition model."""
    
    def __init__(self, name: str, task_models: Dict[str, torch.nn.Module], batch_size: int, 
                 total_batch: int, models_dir: str, num_classes: int,
                 num_clips: int, model_args: Dict[str, float], args, **kwargs) -> None:
        """Create an instance of the action recognition model.

        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        num_classes : int
            number of labels in the classification task
        num_clips : int
            number of clips
        model_args : Dict[str, float]
            model-specific arguments
        """
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args

        # self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.gsd_loss = utils.AverageMeter()
        self.gtd_loss = utils.AverageMeter()
        self.grd_loss = utils.AverageMeter()
        self.lae = utils.AverageMeter()
        self.ly = utils.AverageMeter()
        
        self.num_clips = num_clips
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use the cross entropy loss as the default criterion for the classification task
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        
        # Initializeq the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,weight_decay=model_args[m].weight_decay,momentum=model_args[m].sgd_momentum)

    def forward(self, source_data: Dict[str, torch.Tensor], target_data: Dict[str, torch.Tensor], training,  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ### aggiungere se stiamo trainando o no
        
        """Forward step of the task

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality 

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        logits_source = {}
        features = {}
        for i_m, m in enumerate(self.modalities):
            logits_source[m], feat = self.task_models[m](source_data[m], target_data[m], self.model_args[m].train_clips, training, **kwargs)

            if i_m == 0:
                for k in feat.keys():
                    features[k] = {}
            for k in feat.keys():
                features[k][m] = feat[k]

        return logits_source, features

    def compute_loss(self, logits: Dict[str, torch.Tensor], label: torch.Tensor, predictions: Dict[str,torch.Tensor], loss_weight: float=1.0) :
        """Fuse the logits from different modalities and compute the classification loss.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        if 'gsd' in self.model_args['RGB'].modules:
            pred_gsd_source = predictions['pred_gsd_source']
            domain_label_source=torch.zeros(pred_gsd_source.shape[0], dtype=torch.int64)    
            
            pred_gsd_target = predictions['pred_gsd_target']
            domain_label_target=torch.ones(pred_gsd_target.shape[0], dtype=torch.int64)

            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_gsd_all=torch.cat((pred_gsd_source, pred_gsd_target),0)

            gsd_loss = self.criterion(pred_gsd_all, domain_label_all)
            self.gsd_loss.update(torch.mean(gsd_loss) / (self.total_batch / self.batch_size), self.batch_size) 
        
        if 'gtd' in self.model_args['RGB'].modules:
            pred_gtd_source = predictions['pred_gtd_source']
            domain_label_source=torch.zeros(pred_gtd_source.shape[0], dtype=torch.int64)
        
            pred_gtd_target = predictions['pred_gtd_target']
            domain_label_target=torch.ones(pred_gtd_target.shape[0], dtype=torch.int64)
            
            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_gtd_all=torch.cat((pred_gtd_source,pred_gtd_target),0)

            gtd_loss = self.criterion(pred_gtd_all, domain_label_all)
            self.gtd_loss.update(torch.mean(gtd_loss) / (self.total_batch / self.batch_size), self.batch_size)
        
            if self.model_args['RGB'].use_attention == True:
                pred_clf_all = torch.cat((logits, predictions['pred_clf_target']))
                lae = self.attentive_entropy(pred_clf_all, pred_gtd_all)
                self.lae.update(lae/(self.total_batch / self.batch_size), self.batch_size)
       
        if 'grd' in self.model_args['RGB'].modules and self.model_args['RGB'].aggregation_strategy == 'TemporalRelation':
            grd_loss = []
            for pred_grd_source_single_scale, pred_grd_target_single_scale in zip(predictions['pred_grd_source'], predictions['pred_grd_target']):
                domain_label_source = torch.zeros(pred_grd_source_single_scale.shape[0], dtype=torch.int64)
                domain_label_target = torch.ones(pred_grd_target_single_scale.shape[0], dtype=torch.int64)

                domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
                pred_grd_all_single_scale = torch.cat((pred_grd_source_single_scale, pred_grd_target_single_scale))

                grd_loss_single_scale = self.criterion(pred_grd_all_single_scale, domain_label_all)
                grd_loss.append(grd_loss_single_scale)
            grd_loss = sum(grd_loss)/(len(grd_loss))
            self.grd_loss.update(torch.mean(grd_loss) / (self.total_batch / self.batch_size), self.batch_size)
       
        self.ly.update(torch.mean(self.criterion(logits, label)) / (self.total_batch / self.batch_size), self.batch_size)
        
    def attentive_entropy(self, pred, pred_domain):
        softmax = torch.nn.Softmax(dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)

        # attention weight
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 + entropy

        # attentive entropy
        loss = torch.mean(weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
        
        return loss

    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        
        self.accuracy.update(logits, label)

    def wandb_log(self):
        """Log the current loss and top1/top5 accuracies to wandb."""
        logs = {
            'loss verb': self.loss.val, 
            'top1-accuracy': self.accuracy.avg[1],
            'top5-accuracy': self.accuracy.avg[5]
        }

        # Log the learning rate, separately for each modality.
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr

            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        self.gsd_loss.reset()
        self.gtd_loss.reset()
        self.lae.reset()
        self.grd_loss.reset()
        self.ly.reset()

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()

    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        
        self.gsd_loss 
        self.gtd_loss 
        self.grd_loss
        self.lae 
        self.ly 
        """
        final_loss = 0
        if 'gsd' in self.model_args['RGB'].modules:
            final_loss += self.gsd_loss.val * model_args['RGB'].lambda_s
        if 'gtd' in self.model_args['RGB'].modules:
            final_loss += self.gtd_loss.val * model_args['RGB'].lambda_t
			if model_args['RGB'].use_attention == True:
				final_loss += self.lae.val * model_args['RGB'].gamma
        if 'grd' in self.model_args['RGB'].modules and self.model_args['RGB'].aggregation_strategy == 'TemRelation':
            final_loss += self.grd_loss.val * model_args['RGB'].lambda_r
        final_loss += self.ly.val   
        

        final_loss.backward(retain_graph=retain_graph) 
