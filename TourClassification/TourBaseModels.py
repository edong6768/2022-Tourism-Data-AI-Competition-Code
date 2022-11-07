import torch
from torch import nn
from torchvision import transforms

class TourTextBase(nn.Module):
    def __init__(self, max_len=200, detach=True, device=torch.device('cpu')):
        super(TourTextBase, self).__init__()
        self.max_len = max_len
        self.detach = detach
        self.device = device
        self._init_text_model()
        
    def _init_text_model(self):
        raise NotImplementedError
    
    @property
    def last_layer_size(self):
        raise NotImplementedError
    
    def __call__(self, text):
        raise NotImplementedError
    

class TourVisionBase(nn.Module):
    def __init__(self, detach=True, device=torch.device('cpu')):
        super(TourVisionBase, self).__init__()
        self.detach = detach
        self.device = device
        self._init_vision_model()
        
    def _init_vision_model(self):
        raise NotImplementedError
    
    @property
    def last_layer_size(self):
        raise NotImplementedError
    
    @property
    def imsize(self): 
        raise NotImplementedError
    
    def __call__(self, image):
        raise NotImplementedError


class TourVisionBaseResize(TourVisionBase):
    def __init__(self, device=torch.device('cpu')):
        super(TourVisionBase, self).__init__(device=device)
        self.resize = transforms.Resize((self.imsize, self.imsize))
    
    
    
class TourClassificationBase(nn.Module):
    def __init__(self, num_cls = 128, device=torch.device('cpu')):
        super(TourClassificationBase, self).__init__()
        
        self.num_cls = num_cls
        self.device = device
    
    @classmethod
    def get_dense(cls, in_len, out_len, h_width=2000, h_height=2):
        dense = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_len, h_width),
                nn.BatchNorm1d(h_width),
                nn.ReLU()
            ),
            *(nn.Sequential(
                nn.Linear(h_width, h_width),
                nn.BatchNorm1d(h_width),
                nn.ReLU()
            ) for _ in range(h_height)),
            nn.Sequential(
                nn.Linear(h_width, out_len),
                nn.BatchNorm1d(out_len)
            )
        )
        for layer in dense:
            nn.init.kaiming_normal_(layer[0].weight.data)

        return dense

    @classmethod
    def loss_calc(cls, loss, pred, label):
        return loss(pred, *label),

    @classmethod
    def pred_calc(cls, x):
        if isinstance(x, tuple) or isinstance(x, list):
            return x[0]
        return x
            
    @classmethod
    def get_transformer_encoder(cls, feature_size, nhead=8, num_layers=2):
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return encoder

    def _embed_pipe(self, *data):
        raise NotImplementedError
    
    def _training_pipe(self, embed):
        return self.trainingModel(embed)

    def forward(self, *data):
        embed = self._embed_pipe(*data)
        x = self._training_pipe(embed)
        return x