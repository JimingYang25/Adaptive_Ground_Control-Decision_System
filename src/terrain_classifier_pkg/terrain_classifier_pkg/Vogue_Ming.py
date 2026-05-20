# Classifier model with Architecture --Maintainer: Jiming Yang

import torch
import torch.nn as nn

class Vogue_Ming(nn.Module):
    """Multitask MLP:Shared labelencoder + Classification header + Regression header(slope,roughness,height variation)"""
    def __init__(self, input_dim=40, hidden_dims=[240,40], num_classes=9, dropout=0.2):
        super().__init__()
        self.frame_branch=nn.Sequential(nn.Linear(10,80),nn.BatchNorm1d(80),nn.ReLU(),nn.Dropout(dropout),)
        self.window_branch=nn.Sequential(nn.Linear(30,240),nn.BatchNorm1d(240),nn.ReLU(),nn.Dropout(dropout))
        # shared layers
        layers = []
        prev_dim = 320
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.shared = nn.Sequential(*layers)
        
        # classification header
        self.class_head = nn.Linear(prev_dim, num_classes)
        # regression header
        self.slope_head = nn.Linear(prev_dim, 1)
        self.rough_head = nn.Linear(prev_dim, 1)
        self.elev_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        extract=torch.cat((self.frame_branch(x[:,:10]),self.window_branch(x[:,10:])),dim=1)
        feat = self.shared(extract)
        cls_logits = self.class_head(feat)
        slope = self.slope_head(feat)
        rough = self.rough_head(feat)
        elev = self.elev_head(feat)
        return cls_logits, slope, rough, elev
    
# Classifier model with Architecture --Maintainer: Jiming Yang

