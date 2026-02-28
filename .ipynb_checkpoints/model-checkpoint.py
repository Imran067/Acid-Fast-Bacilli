import torchvision
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models import ResNet50_Weights

def get_model(num_classes=2):
    model = fcos_resnet50_fpn(
        weights=None,                      
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=num_classes
    )
    return model
