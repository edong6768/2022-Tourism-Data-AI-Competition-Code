from .TourBaseModels import *

from transformers import CLIPVisionModel, AutoTokenizer, AutoModel, ViTModel
from transformers.feature_extraction_utils import BatchFeature
from sentence_transformers import SentenceTransformer

import torch
from itertools import starmap

class SentenceTransformerTextModel(TourTextBase):
    def __init__(self, max_len=200, detach=True, device=torch.device('cpu')):
        super(SentenceTransformerTextModel, self).__init__(max_len, detach, device)
        
    def _init_text_model(self):
        self.text_model = SentenceTransformer("Huffon/sentence-klue-roberta-base", device=self.device)
        self.text_model.max_seq_length = self.max_len
    
    @property
    def last_layer_size(self): return 768
    
    def __call__(self, text):
        return self.text_model.encode(text, convert_to_tensor=True)
    
    
class KlueRobertaLargeTextModel(TourTextBase):
    def __init__(self, max_len=200, flatten=True, detach=True, device=torch.device('cpu')):
        super(KlueRobertaLargeTextModel, self).__init__(max_len, detach, device)
        self.flatten = flatten
        
    def _init_text_model(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.text_model = AutoModel.from_pretrained("klue/roberta-large")
        self.text_model.gradient_checkpointing_enable()  
        
    @property
    def last_layer_size(self): return 1024*self.max_len
    
    def _text_tokenize(self, text):
        text_tokens = [self.text_tokenizer.encode_plus(
            t,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt',
        ) for t in text]
        input_ids = torch.stack([t['input_ids'].flatten() for t in text_tokens]).to(self.device)
        attention_mask = torch.stack([t['attention_mask'].flatten() for t in text_tokens]).to(self.device)
    
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    def __call__(self, text):
        text_embed = self.text_model(**self._text_tokenize(text)).last_hidden_state
        
        if self.detach:
            text_embed = text_embed.detach()
        if self.flatten:
            text_embed = torch.flatten(text_embed, 1)
        return text_embed


class CLIPVITVisionModel(TourVisionBase):
    def __init__(self, pretrain_id="openai/clip-vit-base-patch32", detach=True, device=torch.device('cpu')):
        self.pretrain_id = pretrain_id
        super(CLIPVITVisionModel, self).__init__(detach, device=device)
        
    def _init_vision_model(self):
        self.vision_model = CLIPVisionModel.from_pretrained(self.pretrain_id)
        
    @property
    def last_layer_size(self): return 768
    
    @property
    def imsize(self): return 224
    
    def __call__(self, image):
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).pooler_output  # pooled CLS states
        if self.detach:
            image_embed = image_embed.detach()
        return image_embed


class GoogleVITVisionModel(TourVisionBase):
    def __init__(self, flatten=True, pretrain_id='google/vit-large-patch32-384', detach=True, device=torch.device('cpu')):
        self.flatten=flatten
        self.pretrain_id = pretrain_id
        super(GoogleVITVisionModel, self).__init__(detach, device=device)
        
    def _init_vision_model(self):
        self.vision_model = ViTModel.from_pretrained(self.pretrain_id).to(self.device)
        
    @property
    def last_layer_size(self): return None
    
    @property
    def imsize(self): return 384
    
    def __call__(self, image):
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).last_hidden_state
        
        if self.detach:
            image_embed = image_embed.detach()
        if self.flatten:
            image_embed = torch.flatten(image_embed, 1)
        return image_embed
    
class CLIPVITVisionModelResize(TourVisionBaseResize):
    def __init__(self, pretrain_id="openai/clip-vit-base-patch32", detach=True, device=torch.device('cpu')):
        self.pretrain_id = pretrain_id
        super(CLIPVITVisionModelResize, self).__init__(detach, device=device)
        
    def _init_vision_model(self):
        self.vision_model = CLIPVisionModel.from_pretrained(self.pretrain_id)
        
    @property
    def last_layer_size(self): return 768
    
    @property
    def imsize(self): return 224
    
    def __call__(self, image):
        image = self.resize(image)
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).pooler_output  # pooled CLS states
        
        if self.detach:
            image_embed = image_embed.detach()
        return image_embed


class GoogleVITVisionModelResize(TourVisionBaseResize):
    def __init__(self, flatten=True, pretrain_id='google/vit-large-patch32-384', detach=True, device=torch.device('cpu')):
        self.flatten=flatten
        self.pretrain_id = pretrain_id
        super(GoogleVITVisionModelResize, self).__init__(detach, device=device)
        
    def _init_vision_model(self):
        self.vision_model = ViTModel.from_pretrained(self.pretrain_id).to(self.device)
        
    @property
    def last_layer_size(self): return None
    
    @property
    def imsize(self): return 384
    
    def __call__(self, image):
        image = self.resize(image)
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).last_hidden_state
        
        if self.detach:
            image_embed = image_embed.detach()
        if self.flatten:
            image_embed = torch.flatten(image_embed, 1)
        return image_embed
    

class TourClassification(TourClassificationBase):
    def __init__(self, 
                 text_model:TourTextBase, 
                 vision_model:TourVisionBase, 
                 num_cls = 128,
                 max_len=200,
                 detach=True,
                 device=torch.device('cpu')):
        super(TourClassification, self).__init__(num_cls, device)
        
        self.max_len = max_len
        self.text_model = text_model(max_len=max_len, detach=detach, device=device)
        self.vision_model = vision_model(detach=detach, device=device)
        
        self.in_len = self.text_model.last_layer_size+self.vision_model.last_layer_size
        self.trainingModel = self.get_dense(self.in_len, out_len = self.num_cls)
    
    def _embed_pipe(self, *data):
        image, text = data
        return torch.cat((
            self.text_model(text),
            self.vision_model(image)
            ), dim=1)


class TourClassificationTransformer(TourClassificationBase):
    def __init__(self, 
                 num_cls = 128,
                 max_len=200,
                 detach=True,
                 device=torch.device('cpu')):
        super(TourClassificationTransformer, self).__init__(num_cls, device)
        
        self.max_len = max_len
        self.text_model = KlueRobertaLargeTextModel(max_len=max_len, flatten=False, detach=detach, device=device)
        self.vision_model = GoogleVITVisionModel(flatten=False, detach=detach, device=device)
        
        self.in_len = self.text_model.text_model.config.hidden_size
        
        self.trainingEncoder = self.get_transformer_encoder(self.in_len)
        self.trainingDense = self.get_dense(self.in_len, self.num_cls, h_height=0)
    
    def _embed_pipe(self, *data):
        image, text = data
        return torch.cat((
            self.text_model(text),
            self.vision_model(image)
            ), dim=1)
    
    def _training_pipe(self, embed):
        embed = self.trainingEncoder(embed)[:,0] # cls token
        embed = self.trainingDense(embed)
        return embed
    
class TourClassificationTransformerAux(TourClassificationTransformer):
    def __init__(self, num_cls, num_cls_aux1, num_cls_aux2, max_len=200, detach=True, device=torch.device('cpu')):
        super(TourClassificationTransformerAux, self).__init__(
            num_cls = num_cls,
            max_len=max_len,
            detach=detach,
            device=device)
        
        self.num_cls_aux1 = num_cls_aux1
        self.num_cls_aux2 = num_cls_aux2
        
        self.tda1, self.tda2 = self._init_training_dense_aux()
    
    @classmethod
    def loss_calc(cls, loss_fn, pred, label):
        L, Lax1, Lax2 = starmap(loss_fn, zip(pred, label))
        L_tot = L*0.85 + Lax1*0.05 + Lax2*0.1
        return L_tot,
        
    def _init_training_dense_aux(self):
        linearaux1 = self.get_dense(self.in_len, self.num_cls_aux1, h_height=0)
        linearaux2 = self.get_dense(self.in_len, self.num_cls_aux2, h_height=0)
        return linearaux1, linearaux2
    
    def _training_pipe(self, embed):
        embed = self.trainingEncoder(embed)[:,0] # cls token
        return [m(embed) for m in (self.trainingDense, self.tda1, self.tda2)]
        
       
class TourClassification1(TourClassification):
    def __init__(self, num_cls = 128, max_len=200, detach=True, device=torch.device('cpu')):
        super(TourClassification1, self).__init__(
            text_model=SentenceTransformerTextModel, 
            vision_model=CLIPVITVisionModel, 
            num_cls = num_cls,
            max_len=max_len,
            detach=detach,
            device=device)

class TourClassification1Aux(TourClassification1):
    def __init__(self, num_cls, num_cls_aux1, num_cls_aux2, max_len=200, detach=True, device=torch.device('cpu')):
        super(TourClassification1Aux, self).__init__(
            num_cls = num_cls,
            max_len=max_len,
            detach=detach,
            device=device)
        
        self.num_cls_aux1 = num_cls_aux1
        self.num_cls_aux2 = num_cls_aux2
        
        self.tm, self.tma1, self.tma2 = self._init_training_model()
    
    @classmethod
    def loss_calc(cls, loss_fn, pred, label):
        L, Lax1, Lax2 = starmap(loss_fn, zip(pred, label))
        L_tot = L*0.85 + Lax1*0.05 + Lax2*0.1
        return L_tot,
        
    def _init_training_model(self):
        linear1 = self.get_dense(self.in_len, self.num_cls)
        linearaux1 = self.get_dense(self.in_len, self.num_cls_aux1)
        linearaux2 = self.get_dense(self.in_len, self.num_cls_aux2)
        return linear1, linearaux1, linearaux2
    
    def _training_pipe(self, embed):
        return [m(embed) for m in (self.tm, self.tma1, self.tma2)]
    
    
class TourClassification2(TourClassification):
    def __init__(self, num_cls = 128, max_len=200, detach=True, device=torch.device('cpu')):
        super(TourClassification2, self).__init__(
            text_model=KlueRobertaLargeTextModel, 
            vision_model=CLIPVITVisionModel, 
            num_cls = num_cls,
            max_len=max_len,
            detach=detach,
            device=device)
    
class TourClassification1EmbedEnsemble(TourClassification1):
    def __init__(self, num_cls=128, max_len=200, detach=True, device=torch.device('cpu')):
        super(TourClassification1EmbedEnsemble, self).__init__(
            num_cls = num_cls,
            max_len=max_len,
            detach=detach,
            device=device)
        
        self.tm, self.tmt, self.tmi = self._init_training_model()
    
    @classmethod
    def loss_calc(cls, loss_fn, pred, label):
        return starmap(loss_fn, zip(pred, [*label]*3))

    @classmethod
    def pred_calc(cls, pred):
        return sum(w*p for w, p in zip((0.5, 0.25, 0.25), pred))
        
    def _init_training_model(self):
        linear1 = self.get_dense(self.in_len, self.num_cls)
        lineartext = self.get_dense(self.text_model.last_layer_size, self.num_cls)
        linearimg = self.get_dense(self.vision_model.last_layer_size, self.num_cls)
        return linear1, lineartext, linearimg

    def _embed_pipe(self, *data):
        image, text = data
        return self.text_model(text), self.vision_model(image)
    
    def _training_pipe(self, embed):
        text, image = embed
        embed = torch.cat((text, image), dim=1)
        return [m(x) for m, x in zip((self.tm, self.tmt, self.tmi), (embed, text, image))]
    

    