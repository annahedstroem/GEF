"""This module contains the settings for the comparison run. Runs only on GPU."""

from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
import numpy as np
import torch
import torchvision

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import DataLoader

from typing import Optional, Union
from dataclasses import dataclass
import torch
import os
import cv2
import numpy as np
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
    ResNetForImageClassification,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
)
import torchvision
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from torch.utils.data import Dataset

from OpenXAI.openxai.dataloader import ReturnLoaders
from OpenXAI.openxai import LoadModel


OPENXAI_TABULAR_DATASETS = [
    "adult",
    "compas",
    "gaussian",
    "german",
    "gmsc",
    "heart",
    "heloc",
    "pima",
]


@dataclass
class TaskConfig:
    task: str
    num_classes: int
    base_model_name: str
    batch_size: int
    full_size: int
    target_col_name: Optional[str] = None
    split: Optional[str] = None
    token_max_length: Optional[int] = None
    feature_col_name: Union[list, str] = None
    xai_layer_name: Optional[str] = None
    load_name: Optional[str] = None
    add_channel: Optional[bool] = None
    class_labels: Optional[dict] = None
    img_size: Optional[int] = None
    local_path: Optional[str] = None
    nr_channels: Optional[int] = None


# Define configurations using the `TaskConfig` dataclass.
AVAILABLE_EXPERIMENTS = {
    "imdb": TaskConfig(
        task="text",
        feature_col_name="text",
        target_col_name="label",
        load_name="plain_text",
        split="test",
        base_model_name="AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",  # 7M params
        batch_size=25,
        full_size=250,
        token_max_length=512,
        num_classes=2,
        class_labels={0: "negative", 1: "positive"},
    ),
    "emotion": TaskConfig(
        task="text",
        feature_col_name="text",
        target_col_name="label",
        load_name="split",
        split="test",
        base_model_name="j-hartmann/emotion-english-distilroberta-base",  # ~81M params
        batch_size=25,
        full_size=250,
        token_max_length=512,
        num_classes=6,
        class_labels={
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        },
    ),
    "sst2": TaskConfig(
        task="text",
        feature_col_name="sentence",
        target_col_name="label",
        load_name="default",
        split="validation",
        base_model_name="VityaVitalich/bert-tiny-sst2",  # 13 MB
        batch_size=125,
        full_size=250,
        token_max_length=59,
        num_classes=2,
        class_labels={0: "negative", 1: "positive"},
    ),
    "sms_spam": TaskConfig(
        task="text",
        feature_col_name="sms",
        target_col_name="label",
        load_name="plain_text",
        split="train",
        base_model_name="mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection",  # 7 M
        batch_size=50,
        full_size=250,
        token_max_length=128,
        num_classes=2,
        class_labels={0: "not spam", 1: "spam"},
    ),
    "imagenet-1k": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        target_col_name="label",
        split="validation",
        base_model_name="torchvision.models.resnet18",
        batch_size=250,
        full_size=250,
        num_classes=1000,
        img_size=224,
        xai_layer_name="list(model.named_modules())[61][1]",  # if VGG16 then: "model.features[-2]"
    ),
    "mnist": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="lenet",
        xai_layer_name="list(model.named_modules())[3][1]",
        batch_size=500,
        full_size=500,
        num_classes=10,
        nr_channels=1,
        img_size=28,
        class_labels={i + 1: f"{i+1}" for i in range(10)},
    ),
    # sehee-lim/fashion_classification_model, "SE6446/VitMix-v1"
    "fashion_mnist": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="lenet",
        xai_layer_name="list(model.named_modules())[3][1]",
        batch_size=500,
        full_size=500,
        num_classes=10,
        nr_channels=1,
        img_size=28,
        class_labels={
            "0": "T - shirt / top",
            "1": "Trouser",
            "2": "Pullover",
            "3": "Dress",
            "4": "Coat",
            "5": "Sandal",
            "6": "Shirt",
            "7": "Sneaker",
            "8": "Bag",
            "9": "Ankle boot",
        },
    ),
    "retina": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="cnn",
        xai_layer_name="list(model.named_modules())[-11][1]",
        batch_size=250,
        full_size=250,  # 400 in total
        num_classes=7,
        nr_channels=3,
        img_size=28,
        class_labels={
            "0": "Severity Level 1",
            "1": "Severity Level 2",
            "2": "Severity Level 3",
            "3": "Severity Level 4",
            "4": "Severity Level 5",
        },
    ),
    "derma": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="cnn",
        xai_layer_name="list(model.named_modules())[-11][1]",
        batch_size=500,
        full_size=500,  # 2000 in total
        num_classes=7,
        nr_channels=3,
        img_size=28,
        class_labels={
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions",
        },
    ),
    "path": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="cnn",
        xai_layer_name="list(model.named_modules())[-11][1]",
        batch_size=500,
        full_size=500,  # 2000 in total
        num_classes=9,
        nr_channels=3,
        img_size=28,
        class_labels={
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium",
        },
    ),
    "blood": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="cnn",
        xai_layer_name="list(model.named_modules())[-11][1]",
        batch_size=500,
        full_size=500,  # 2000 in total
        num_classes=8,
        nr_channels=3,
        img_size=28,
        class_labels={
            "0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet",
        },
    ),
    "chest": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="cnn",
        xai_layer_name="list(model.named_modules())[-11][1]",
        batch_size=500,
        full_size=500,  # 2000 in total
        num_classes=14,
        nr_channels=1,
        img_size=28,
        class_labels={
            "0": "atelectasis",
            "1": "cardiomegaly",
            "2": "effusion",
            "3": "infiltration",
            "4": "mass",
            "5": "nodule",
            "6": "pneumonia",
            "7": "p neumothorax",
            "8": "consolidation",
            "9": "edema",
            "10": "emphysema",
            "11": "fibrosis",
            "12": "pleural",
            "13": "hernia",
        },
    ),
    "pneumonia": TaskConfig(
        task="vision",
        load_name="local",
        local_path="/mnt/beegfs/home/anonymous/assets/",
        base_model_name="cnn",
        xai_layer_name="list(model.named_modules())[-11][1]",
        batch_size=250,
        full_size=250,  # 600 in total
        num_classes=2,
        nr_channels=1,
        img_size=28,
        class_labels={"0": "normal", "1": "pneumonia"},
    ),
    "avila": TaskConfig(
        task="tabular",
        load_name="local",
        # local_path="/mnt/beegfs/home/anonymous/assets/",
        local_path="../../assets/",
        base_model_name="mlp",
        batch_size=500,
        full_size=500,
        num_classes=12,
        class_labels={i + 1: f"{i+1}" for i in range(12)},
    ),
    "adult": TaskConfig(
        task="tabular",
        base_model_name="ann",
        batch_size=250,
        full_size=250,
        num_classes=2,
        class_labels={0: "under threshold", 1: "over threshold"},
    ),
    "compas": TaskConfig(
        task="tabular",
        base_model_name="ann",
        batch_size=250,
        full_size=250,
        num_classes=2,
        class_labels={0: "not rearrested", 1: "rearrested"},
    ),
}


IMAGENET_MODELS_SPECIAL_LOADING = {
    "google/vit-base-patch16-224": {
        "classifier": ViTForImageClassification,  # 346 MB
        "processor": ViTImageProcessor,
    },
    "microsoft/resnet-50": {
        "classifier": AutoImageProcessor,  # 346 MB
        "processor": ResNetForImageClassification,
    },
    "facebook/convnext-tiny-224": {
        "classifier": ConvNextForImageClassification,  # 114 MB
        "processor": ConvNextImageProcessor,
    },
}


class LeNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True
        )
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
        )
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(torch.nn.Module):
    def __init__(
        self, num_features: int = 10, num_labels: int = 12, n_neurons: int = 150
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, n_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_neurons, num_labels)
        # self.ac2 = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        logits = self.fc2(x)
        # x = self.ac2(logits)
        return logits


class RetinaCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        super(RetinaCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DermaCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        super(DermaCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PathCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 9):
        super(PathCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BloodCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 8):
        super(BloodCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ChestCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 14):
        super(ChestCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PneumoniaCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(PneumoniaCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CustomTabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CustomImageNetDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.folder_path = folder_path
        self.transforms = transforms
        self.input_names = [self.folder_path + x for x in os.listdir(self.folder_path)]
        self.input_names.sort()

        if self.transforms is None:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(224),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, index: int):
        # TODO. Add y labels.
        y = torch.tensor(0, dtype=torch.float)
        image = cv2.imread(self.input_names[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image)
        return image, y


class OpenXAIModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.predict = self.model.predict_with_logits

    def forward(self, x):
        # Assumes the output from predict is a numpy array.
        np_output = self.model.predict(x)  # .float()
        return np_output


class Experiment:

    def __init__(
        self,
        dataset_name: str,
        device: torch.device,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        full_size: Optional[int] = None,
    ):
        """Initialise the experiment."""
        self.dataset_name = dataset_name
        self.device = device
        self.config = AVAILABLE_EXPERIMENTS[dataset_name]

        self.full_size = full_size if full_size is not None else self.config.full_size
        self.batch_size = min(
            batch_size if batch_size is not None else self.config.batch_size,
            self.full_size,
        )
        self.config.batch_size = self.batch_size
        self.config.full_size = self.full_size

        self.model_name = (
            model_name if model_name is not None else self.config.base_model_name
        )
        self.model = self.load_model()
        self.total_params = sum(p.numel() for p in self.model.parameters())

        print(
            f"Loading experiment for '{self.dataset_name}' with '{self.model_name}' (#{self.total_params//1000000}M params) ..."
        )

        self.tokenizer = self.load_tokenizer()
        self.processor = self.load_processor()

        self.test_dataloader = self.load_test_dataset()
        self.xai_layer_name = (
            self.get_embedding_layer()
            if self.config.xai_layer_name is None and self.config.task == "text"
            else None
        )  # self.get_second_last_layer()

    def load_model(self):
        """Load the model."""

        if "torchvision" in self.model_name:
            if "resnet" in self.model_name:
                return eval(self.model_name + "(weights=ResNet18_Weights.DEFAULT)").to(
                    self.device
                )
            if "vit" in self.model_name:
                return eval(self.model_name + "(weights=ViT_B_16_Weights.DEFAULT)").to(
                    self.device
                )

        # Load local model.
        if self.config.local_path is not None:
            model_path = f"{self.config.local_path}models/{self.dataset_name.lower()}_{self.model_name.lower()}"
            model = None
            if "mnist" in self.dataset_name:
                model = LeNet()
            elif self.dataset_name == "avila":
                model = MLP()
            if self.dataset_name == "derma":
                model = DermaCNN()
            if self.dataset_name == "retina":
                model = RetinaCNN()
            if self.dataset_name == "blood":
                model = BloodCNN()
            if self.dataset_name == "pneumonia":
                model = PneumoniaCNN()
            if self.dataset_name == "path":
                model = PathCNN()
            if self.dataset_name == "chest":
                model = ChestCNN()

            model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            assert (
                model is not None
            ), f"Local model not found for {self.dataset_name} via path {model_path}."

            return model

        if self.dataset_name in OPENXAI_TABULAR_DATASETS:

            model_openxai = LoadModel(
                data_name=self.dataset_name, ml_model=self.model_name, pretrained=True
            )
            model = OpenXAIModelWrapper(model_openxai)
            model.to(self.device)
            return model

        # Load HuggingFace model.
        classifier = (
            IMAGENET_MODELS_SPECIAL_LOADING[self.model_name]["classifier"]
            if self.model_name in IMAGENET_MODELS_SPECIAL_LOADING
            else None
        )
        if classifier is not None:
            return classifier.from_pretrained(self.model_name).to(self.device)
        try:
            if self.config.task == "text":
                return AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
            if self.config.task == "vision":
                return AutoModelForImageClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
            if self.config.task == "tabular":
                return AutoModel.from_pretrained(self.model_name).to(self.device)
            if self.config.task == "audio":
                return AutoModelForAudioClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
        except Exception as e:
            raise KeyError(
                f"Couldn't load model for '{self.dataset_name}', with {self.model_name}.\nError message: {e}"
            )

    def load_test_dataset(self):
        """Load the dataset."""

        if self.dataset_name in OPENXAI_TABULAR_DATASETS:

            _, self.test_dataloader = ReturnLoaders(
                data_name=self.dataset_name, download=False, batch_size=self.batch_size
            )
            return self.test_dataloader

        elif self.config.local_path is not None:
            if "mnist" in self.dataset_name or self.dataset_name in [
                "avila",
                "retina",
                "derma",
                "path",
                "blood",
                "chest",
                "pneumonia",
                "imagenet-1k",
            ]:
                if self.dataset_name == "imagenet-1k":
                    path = f"{self.config.local_path}test_sets/imagenet_test_set_random.npy"
                else:
                    path = f"{self.config.local_path}test_sets/{self.dataset_name.lower()}_test_set.npy"
                assets = np.load(
                    path,
                    allow_pickle=True,
                ).item()
                x_batch = assets["x_batch"]
                y_batch = assets["y_batch"]
                self.config.full_size = len(x_batch)
                processed_dataset = TensorDataset(
                    torch.tensor(x_batch, dtype=torch.float),
                    torch.tensor(y_batch, dtype=torch.int),
                )

            elif self.dataset_name == "imagenet-1k":
                processed_dataset = CustomImageNetDataset(
                    folder_path=self.config.local_path
                )
        else:

            try:
                dataset = load_dataset(
                    self.dataset_name,
                    self.config.load_name,
                    split=self.config.split,
                    cache_dir="/mnt/beegfs/home/anonymous/datasets/",
                )
            except Exception as e:
                raise KeyError(
                    f"Couldn't load {self.dataset_name}.\nError message: {e}"
                )

            try:

                if self.config.task == "text":

                    def preprocess_function(dataset):
                        # Tokenize the text.
                        return self.tokenizer(
                            dataset[self.config.feature_col_name],
                            padding="max_length",
                            truncation=True,
                            max_length=self.config.token_max_length,
                            return_tensors="pt",
                        )

                    # Apply the tokenization to the entire dataset and convert format to PyTorch tensors.
                    processed_dataset = dataset.map(preprocess_function, batched=True)
                    processed_dataset.set_format(
                        type="torch",
                        columns=[
                            "input_ids",
                            "attention_mask",
                            self.config.target_col_name,
                        ],
                    )

                elif self.config.task == "vision":

                    def preprocess_function(dataset):

                        if self.config.add_channel:
                            transformations = torchvision.transforms.Compose(
                                [
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Lambda(
                                        lambda x: x.repeat(3, 1, 1)
                                    ),
                                ]
                            )
                            dataset["image"] = torch.stack(
                                [transformations(image) for image in dataset["image"]]
                            )
                            # print(dataset["image"][0].shape)

                        processed = self.processor(
                            images=dataset["image"],
                            return_tensors="pt",
                        )
                        processed[self.config.target_col_name] = torch.tensor(
                            dataset[self.config.target_col_name]
                        )
                        return processed

                    processed_dataset = dataset.map(preprocess_function, batched=True)
                    processed_dataset.set_format(
                        type="torch",
                        columns=["pixel_values", self.config.target_col_name],
                    )

                elif self.config.task == "tabular":

                    # Preprocess the dataset deoending on dtype.
                    for column in dataset.column_names:
                        if dataset.features[column].dtype == "string":
                            le = LabelEncoder()
                            dataset = dataset.map(
                                lambda examples: {
                                    column: le.fit_transform(examples[column])
                                },
                                batched=True,
                            )

                    # Convert to PyTorch tensors.
                    features = torch.tensor(
                        np.array(dataset[:][self.config.feature_col_name]),
                        dtype=torch.float32,
                    )
                    labels = torch.tensor(
                        np.array(dataset[:][self.config.target_col_name]),
                        dtype=torch.long,
                    )
                    processed_dataset = CustomTabularDataset(features, labels)

                    # decision_function = lambda x: self.model(x)

                elif self.config.task == "audio":

                    def preprocess_function(batch):
                        processed_features = []

                        for audio_sample in batch["audio"]:
                            audio_array = audio_sample["array"]
                            sampling_rate = audio_sample["sampling_rate"]

                            features = self.processor(
                                audio_array,
                                sampling_rate=sampling_rate,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                            )

                            processed_features.append(features)

                        return {"input_values": processed_features}

                    # Apply preprocessing and set format for PyTorch.
                    processed_dataset = dataset.map(preprocess_function, batched=True)
                    processed_dataset.set_format(
                        type="torch",
                        columns=["input_values", self.config.target_col_name],
                    )

            except Exception as e:
                raise KeyError(
                    f"Couldn't load data for '{self.dataset_name}', with task {self.config.task}. Check implementation.\nError message: {e}"
                )

        self.test_dataloader = DataLoader(
            processed_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )

        return self.test_dataloader

    def load_processor(self):
        """Load the processor."""
        if self.config.local_path is not None:
            return None
        try:
            processor = (
                IMAGENET_MODELS_SPECIAL_LOADING[self.model_name]["processor"]
                if self.model_name in IMAGENET_MODELS_SPECIAL_LOADING
                else None
            )
            if processor is not None:
                return processor.from_pretrained(self.model_name).to(self.device)
            if self.config.task == "vision":
                if "torchvision" in self.model_name:
                    return None
                if self.config.add_channel:
                    return AutoImageProcessor.from_pretrained(
                        self.model_name, do_rescale=False
                    )
                return AutoImageProcessor.from_pretrained(self.model_name)
            elif self.config.task == "audio":
                return AutoFeatureExtractor.from_pretrained(self.model_name)
            else:
                return None
        except Exception as e:
            raise KeyError(
                f"Couldn't load processor for '{self.dataset_name}', with {self.model_name}.\nError message: {e}"
            )

    def load_tokenizer(self):
        """Load the tokenizer."""
        try:
            if self.config.task == "text":
                return AutoTokenizer.from_pretrained(self.model_name)
            else:
                return None
        except Exception as e:
            raise KeyError(
                f"Couldn't load tokenizer for '{self.dataset_name}', with {self.model_name}.\nError message: {e}"
            )

    def get_embedding_layer(self):
        """Get the name of the embedding layer."""
        try:
            # By layer name pattern (assuming naming convention is consistent) or later type.
            for name, module in self.model.named_modules():  # [::-1]:
                if "embeddings" in name or isinstance(module, torch.torch.nn.Embedding):
                    print(f"{name} is used as the embedding layer.")
                    return name
        except:
            # Fallback if no embedding layer is found.
            raise ValueError("Embedding layer not found")

    def get_second_last_layer(self):
        """Get the name of the second last layer."""
        try:
            # Convert the model layers to a list.
            layers = list(self.model.named_modules())
            if len(layers) < 3:  # Check if there are enough layers
                raise ValueError(
                    "The model does not have enough layers to extract the second last layer."
                )

            # Get the name of the second last layer.
            second_last_layer_name = layers[-2][0]
            print(f"{second_last_layer_name} is used as the second last layer.")
            return second_last_layer_name
        except:
            # Fallback if no embedding layer is found.
            raise ValueError("Second last layer not found")

    def generate_batch(self):
        """Generate batches of data."""
        nr_samples = 0
        for batch in self.test_dataloader:
            if nr_samples >= self.full_size:
                break
            if self.config.local_path is not None:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                am_batch = None
            elif self.config.task == "text":
                x_batch = batch["input_ids"].to(self.device)
                y_batch = batch[self.config.target_col_name].to(self.device)
                am_batch = batch["attention_mask"].to(self.device)
            elif self.config.task == "vision":
                x_batch = batch["pixel_values"].to(self.device)
                y_batch = batch[self.config.target_col_name].to(self.device)
                am_batch = None
            elif self.config.task == "tabular":
                if self.dataset_name in OPENXAI_TABULAR_DATASETS:
                    x_batch = batch[0].to(self.device)
                    x_batch = x_batch.float()
                    x_batch.requires_grad = True
                    y_batch = batch[1].to(self.device)
                else:
                    x_batch = batch["features"].to(self.device)
                    y_batch = batch[self.config.target_col_name].to(self.device)
                am_batch = None
            elif self.config.task == "audio":
                x_batch = batch["input_values"].to(self.device)
                y_batch = batch[self.config.target_col_name].to(self.device)
                am_batch = None
            nr_samples += len(x_batch)
            yield x_batch, y_batch, am_batch


if __name__ == "__main__":
    pass
