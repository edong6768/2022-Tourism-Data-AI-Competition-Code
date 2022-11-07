# :cityscape: 2022 Tourism Data AI Competition Submission Code
Submissions for 2022 Tourism Data AI Competition(https://dacon.io/competitions/official/235978/overview/description) 

## Task Description
Classify `cat3` label of datapoint given image and text pair.

## Data example
|id|img_path|overview|cat1|cat2|cat3|
|-|-|-|-|-|-|
TRAIN_00000|./image/train/TRAIN_00000.jpg|"소안항은 조용한 섬으로 인근해안이 청정해역으로 일찍이 김 양식을 해서 높은 소득을 올리고 있으며 바다낚시터로도 유명하다. 항 주변에 설치된 양식장들은 섬사람들의 부지런한 생활상을 고스 란히 담고 있으며 일몰 때 섬의 정경은 바다의 아름다움을 그대로 품고 있는 듯하다. 또한, 섬에는 각시여 전설, 도둑바위 등의 설화가 전해 내려오고 있으며, 매년 정월 풍어제 풍속이 이어지고 있다.<br>"|자연|자연관광지|항구/포구
TRAIN_00001|./image/train/TRAIN_00001.jpg|"경기도 이천시 모가면에 있는 골프장으로 대중제 18홀이다. 회원제로 개장을 했다가 2016년 대중제로 전환하여 재개장했다. 총 부지 약 980,,000㎡에 전장 6,607m에 18홀 파 72이다. Lake 코스와 Mountain 코스가 있다. 미국 100대 골프 코스 설계자인 짐 파지오가 아마추어에게는 쉽고 프로골퍼에게는 어렵게 설계했다고 한다. 가까이에 뉴스프링빌CC, 써닝포인트CC, 비에이비스타CC, 덕평CC 등의 골프장이 있다."|레포츠|육상 레포츠|골프
TRAIN_00002|./image/train/TRAIN_00002.jpg|금오산성숯불갈비는 한우고기만을 전문적으로 취급하고 사용하는 부식 자재 또한 유기농법으로 재배한 청정야채만을 취급하고 있다고 한다. 음식을 담은 그릇도 모두 전통 놋그릇으로 통일하였고 수저 또한 깨끗하고 예쁜 수젓집에 넣어서 나오는 등 작은 곳에서부터 정성을 다하는 모습을 느낄 수 있다.|음식|음식점|한식
|||...|||

# Approach
Use pre-trained text/image models to extract features, the use transformer block/Dense layer to classify the label.
Incorporated augmentation techniques in both text and images and used Focal loss to handle data imbalance problem.
Attempt to implement ensemble but unsuccessful due to limited computational resource.
### Best Test Accuracy : `74.538%`

## Train Log
|Config|Description|Augmentation|Ensemble|Text Model|Vision Model|Loss Func|Optimizer|LR|
|-|-|-|-|-|-|-|-|-|
1|Basic model|X|X|Huffon/sentence-klue-roberta-base|openai/clip-vit-base-patch32|CrossEnt.|Adam|1.00E-03
2|Ensemble(3)|X|O|Huffon/sentence-klue-roberta-base|openai/clip-vit-base-patch32	|CrossEnt.|	Adam|	1.00E-03
3|Augmentation(Data Oversample)|O	|X|Huffon/sentence-klue-roberta-base	|openai/clip-vit-base-patch32	|CrossEnt.	|Adam	|1.00E-03
4|Change text model|X|X|klue/roberta-large|openai/clip-vit-base-patch32	|CrossEnt.	|Adam	|1.00E-02
5|Auxilary Classifier(use cat1, 2)|X|X|Huffon/sentence-klue-roberta-base|openai/clip-vit-base-patch32	|CrossEnt.	|Adam	|1.00E-02
6|Auxilary Classifier(use cat1, 2)|X|X|Huffon/sentence-klue-roberta-base|openai/clip-vit-base-patch32	|CrossEnt.	|Adam	|3.00E-05
7|Transformer Encoder + Focal Loss|X|X|klue/roberta-large|	google/vit-large-patch32-384	|Focal Loss	|Adam	|3.00E-05
8|Auxilary Classifier with Focal Loss|X|X|klue/roberta-large|	google/vit-large-patch32-384	|Focal Loss|	Adam	|3.00E-05
