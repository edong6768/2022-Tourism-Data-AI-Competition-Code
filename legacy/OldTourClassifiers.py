from transformers import CLIPVisionModel, AutoTokenizer, AutoModel
from transformers.feature_extraction_utils import BatchFeature
from sentence_transformers import SentenceTransformer

import torch
from torch import nn


class TourClassification1(nn.Module):
    def __init__(self, num_cls = 128, device=torch.device('cpu')):
        super(TourClassification1, self).__init__()
        self.device = device
        
        self.text_model = SentenceTransformer("Huffon/sentence-klue-roberta-base", device=device)
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_model.max_seq_length = 200
        
        self.linear1 = nn.Sequential(
            nn.Linear(1536, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(2000, num_cls),
            nn.BatchNorm1d(num_cls)
        )
        
        torch.nn.init.kaiming_normal_(self.linear1[0].weight.data)
        torch.nn.init.kaiming_normal_(self.linear2[0].weight.data)
        torch.nn.init.kaiming_normal_(self.linear3[0].weight.data)
        torch.nn.init.kaiming_normal_(self.linear4[0].weight.data)
        
        
    def forward(self, image, text):
        text_embed = self.text_model.encode(text, convert_to_tensor=True)
        
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).pooler_output.detach()  # pooled CLS states
        
        embed = torch.cat((text_embed, image_embed), dim=1)
        
        x = self.linear1(embed)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x
    
class TourClassification2(nn.Module):
    def __init__(self, num_cls = 128, device=torch.device('cpu')):
        super(TourClassification2, self).__init__()
        self.device = device
        
        self.text_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.text_model = AutoModel.from_pretrained("klue/roberta-large")
        self.max_len = 200
        
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        self.linear1 = nn.Sequential(
            nn.Linear(1024*self.max_len+768, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(2000, num_cls),
            nn.BatchNorm1d(num_cls)
        )
        
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
    
    def forward(self, image, text):
        
        text_embed = self.text_model(**self._text_tokenize(text)).last_hidden_state.detach()
        text_embed = torch.flatten(text_embed, 1)
        
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).pooler_output.detach()  # pooled CLS states
        
        embed = torch.cat((text_embed, image_embed), dim=1)
        
        x = self.linear1(embed)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x
    
class TourClassificationText(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(TourClassificationText, self).__init__()
        self.device = device
        
        self.text_model = SentenceTransformer("Huffon/sentence-klue-roberta-base", device=device)
        
        self.text_model.max_seq_length = 200
        
        self.linear1 = nn.Sequential(
            nn.Linear(768, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(2000, 128),
            nn.BatchNorm1d(128)
        )
        
        
    def forward(self, text):
        text_embed = self.text_model.encode(text, convert_to_tensor=True)
        x = self.linear1(text_embed)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x
    
class TourClassificationImage(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(TourClassificationImage, self).__init__()
        self.device = device
        
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.linear1 = nn.Sequential(
            nn.Linear(768, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(2000, 128),
            nn.BatchNorm1d(128)
        )
        
        
    def forward(self, image):
        
        image_f = BatchFeature(data={"pixel_values": image}, tensor_type="pt")
        image_embed = self.vision_model(**image_f).pooler_output.detach()  # pooled CLS states
        
        x = self.linear1(image_embed)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x