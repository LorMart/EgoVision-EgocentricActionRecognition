import torch.nn as nn
import torch
import models
from torch.autograd import Function
from models import TRNmodule

class TA3N(nn.Module):
    "Model architecture explained in the paper Temporal Attentive Alignment for Large-Scale Video Domain Adaptation"

    def __init__(self,  input_feature_dim=1024, num_classes=8, model_config=None, name='ta3n'):
        """Iniziales TA3N model instance
        Args:
              num_classes ---> output in the logit layer of the predictor
              input_feature_dim ---> input dimension
              name ---> name of the model
              model_config ---> configuration from the configs file
              train_clips ---> number of vectors of features for each video to classify
              aggregation_strategy ---> TemporalPooling/TemporalRelation
              n_relations ---> number of relations of TRN """

        super(TA3N, self).__init__()
        self.num_classes = num_classes
        self.model_config = model_config
        self.input_feature_dim = input_feature_dim
        self.train_clips = model_config.train_clips
        self.aggregation_strategy = model_config.aggregation_strategy
        if model_config.aggregation_strategy == 'TemporalRelation':
            self.n_relations = self.train_clips-1

        """Spacial Module ---> Gsf"""
        self.gsf = self.FCL(input_feature_dim=self.input_feature_dim, output_feature_dim=self.input_feature_dim, dropout=model_config.dropout)

        """Domain Classifier ---> Gsd"""
        self.gsd = self.DomainClassifier(input_feature_dim=self.input_feature_dim, beta = self.model_config.beta0 , dropout=model_config.dropout)

        """Temporal Module ---> TemporalPooling/TemporalRelation"""
        self.temporal_module = self.TemporalModule(self.input_feature_dim,self.train_clips, self.aggregation_strategy, self.model_config)
        self.input_feature_dim = self.temporal_module.output_feature_dim  #input_feature_dim diventa num_bottlenecks(TRN) oppure output_feature_dim(TemporalPooling)
        print(self.temporal_module.output_feature_dim)
        """Domain Classifiers ---> Gtd & Grd"""
        self.grd = []
        if model_config.aggregation_strategy == 'TemporalRelation':
            for i in range(self.n_relations):
                self.grd.append(self.DomainClassifier(self.model_config.num_bottleneck, beta = self.model_config.beta1, dropout = model_config.dropout)) #one for each relation

        self.gtd = self.DomainClassifier(input_feature_dim=self.input_feature_dim, beta = self.model_config.beta0, dropout=model_config.dropout)

        """Final Classifier"""
        self.gy = self.FCL(input_feature_dim=self.input_feature_dim, output_feature_dim=self.num_classes, dropout=model_config.dropout)

    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy
        return weights

    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        weights_attn = self.get_trans_attn(pred_domain)

        weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
        # raise UserWarning(f'Dimensions are weights_attn={weights_attn.shape}, feat_fc={feat_fc.shape}')
        feat_fc_attn = (weights_attn+1) * feat_fc

        return feat_fc_attn, weights_attn[:,:,0]

    def forward(self, source_data, target_data, train_clips):

        if self.model_config.training == True:
            training = True
        """Spacial Module ---> Gsf"""
        source_data = self.gsf(source_data)
        target_data = self.gsf(target_data) if training else None

        """Domain Classifier ---> Gsd"""
        pred_gsd_source = self.gsd(source_data.view((-1,1024)))
        pred_gsd_target = self.gsd(target_data.view((-1,1024))) if training else None

        """Temporal Module ---> TemporalPooling/TemporalRelation"""
        source_data, dict_feat_trn_source = self.temporal_module(source_data, train_clips)
        target_data, dict_feat_trn_target = self.temporal_module(target_data, train_clips) if training else None

        """Domain Classifiers ---> Grd"""
        if self.model_config.aggregation_strategy == 'TemporalRelation':
            for i in range(self.n_relations):
                pred_grd_source = self.grd[i](dict_feat_trn_source[i])
                pred_grd_target = self.grd[i](dict_feat_trn_target[i]) if training else None

        """??? implement attention ???""" ##manca implementazione
        if self.model_config.use_attention:



            pass

        if self.model_config.aggregation_strategy == 'TemporalRelation':
            source_data = torch.sum(source_data, 1)
            target_data = torch.sum(target_data, 1) if training else None

        """Domain Classifiers ---> Gtd"""
        pred_gtd_source = self.gtd(source_data)
        pred_gtd_target = self.gtd(target_data) if training else None

        """Final Prediction"""
        final_logits_source = self.gy(source_data)
        final_logits_target = self.gy(target_data)

        return final_logits_source , { "pred_gsd_source": pred_gsd_source,"pred_gsd_target": pred_gsd_target, \
                                                                    "pred_gtd_source": pred_gtd_source,"pred_gtd_target": pred_gtd_target, \
                                                                    "pred_grd_source": pred_grd_source,"pred_grd_target": pred_grd_target,
                                                                    "pred_clf_source": final_logits_source,"pred_clf_target": final_logits_target}
    
    class TemporalModule(nn.Module):
        """Implementation of 2 different strategies to aggregate the frame features"""
        def __init__(self, input_feature_dim, train_clips, strategy = 'TemporalPooling',  model_config=None):
            super(TA3N.TemporalModule, self).__init__()
            self.input_feature_dim = input_feature_dim
            self.strategy = strategy
            self.model_config = model_config
            self.num_bottleneck = 512
            self.train_segments = 5
            if strategy == 'TemporalPooling':
               self.output_feature_dim = self.input_feature_dim
            else:
              """TemporalRelation"""
              self.output_feature_dim = self.num_bottleneck
              self.temporalRelation = TRNmodule.RelationModuleMultiScale(self.input_feature_dim, self.num_bottleneck, self.train_segments)



        def forward(self, x, clips):
            x = x[:,None,:,:]
            if self.strategy == 'TemporalPooling':
                x = nn.AvgPool2d([clips, 1])(x)
                x = x.squeeze(1).squeeze(1)
                return x, None
            else:
              """TemporalRelation"""
              x, features = self.temporalRelation(x.squeeze(1))
              return x, features





    class FCL(nn.Module):
        """Implementation of all the MLPs in the architecture"""

        def __init__(self, input_feature_dim, output_feature_dim, dropout=0.5):
            super(TA3N.FCL, self).__init__()
            self.input_feature_dim = input_feature_dim
            self.output_feature_dim = output_feature_dim
            self.dropout = nn.Dropout(p=dropout)
            self.linear = nn.Linear(self.input_feature_dim, self.output_feature_dim)
            self.relu = nn.ReLU()
            

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x


    class GradReverse(Function):
            @staticmethod
            def forward(ctx, x, beta):
                ctx.beta = beta
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                grad_input = grad_output.neg() * ctx.beta
                return grad_input, None

    class DomainClassifier(nn.Module):
        """Implementation of Gd"""

        def __init__(self, input_feature_dim, dropout, beta):
            super(TA3N.DomainClassifier, self).__init__()
            self.input_feature_dim = input_feature_dim
            self.beta = beta
            self.output_feature_dim = 2
            self.linear = nn.Linear(self.input_feature_dim, self.input_feature_dim)
            self.hidden = nn.Linear(self.input_feature_dim, self.output_feature_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = TA3N.GradReverse.apply(x, self.beta)
            x = self.linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.hidden(x)
            return x