import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
#--------------------------------------------#
class Classification(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : 'model_data/vit-patch_16.pth',
        "model_path": 'logs/best_epoch_weights.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        #--------------------------------------------------------------------#
        #   输入的图片大小
        #--------------------------------------------------------------------#
        "input_shape"       : [224, 224],
        #--------------------------------------------------------------------#
        #   所用模型种类：
        #   mobilenetv2、
        #   resnet18、resnet34、resnet50、resnet101、resnet152
        #   vgg11、vgg13、vgg16、vgg11_bn、vgg13_bn、vgg16_bn、
        #   vit_b_16、
        #   swin_transformer_tiny、swin_transformer_small、swin_transformer_base
        #--------------------------------------------------------------------#
        "backbone"          : 'vit_b_16',
        #--------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化classification
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #---------------------------------------------------#
        #   获得种类
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   载入模型与权值
        #---------------------------------------------------#
        if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    # def generate(self):
    #     # 载入模型（不包括预训练的分类层）
    #     if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small',
    #                              'swin_transformer_base']:
    #         # 假设这些backbone不需要input_shape参数
    #         self.model = get_model_from_name[self.backbone](num_classes=1000, pretrained=False)  # 临时使用1000个类别来匹配预训练模型
    #     else:
    #         self.model = get_model_from_name[self.backbone](input_shape=self.input_shape, num_classes=1000,
    #                                                         pretrained=False)  # 临时使用1000个类别
    #
    #     # 假设self.model的最后一层是名为'classifier'或'head'的层，需要找到它并替换它
    #     classifier_name = 'classifier' if hasattr(self.model, 'classifier') else 'head'  # 根据您的模型架构调整
    #     pretrained_classifier = getattr(self.model, classifier_name)
    #
    #     # 创建一个新的全连接层，其输入特征数为预训练分类层的输入特征数
    #     num_ftrs = pretrained_classifier.in_features
    #     new_classifier = nn.Linear(num_ftrs, self.num_classes)  # 假设self.num_classes是2
    #
    #     # 替换模型中的原始分类层
    #     setattr(self.model, classifier_name, new_classifier)
    #
    #     # 加载预训练权重（除了最后一层）
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     pretrained_dict = torch.load(self.model_path, map_location=device)
    #     model_dict = self.model.state_dict()
    #
    #     # 过滤出预训练模型中除了最后一层以外的权重
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                        not k.endswith(classifier_name + '.weight') and not k.endswith(classifier_name + '.bias')}
    #
    #     # 加载预训练权重到模型中
    #     model_dict.update(pretrained_dict)
    #     self.model.load_state_dict(model_dict)
    #
    #     # 初始化新添加的全连接层的权重（通常这一步是可选的，因为PyTorch会自动初始化）
    #     # 但如果你想要特定的初始化方式，可以在这里添加
    #
    #     # 将模型设置为评估模式
    #     self.model.eval()
    #     print('{} model, and custom classes loaded.'.format(self.model_path))
    #
    #     # 如果在GPU上运行
    #     if self.cuda:
    #         self.model = nn.DataParallel(self.model)
    #         self.model = self.model.to(device)



    # def generate(self):
    #     # 载入模型（不包括预训练的分类层）
    #     self.model = get_model_from_name[self.backbone](input_shape=self.input_shape, num_classes=self.num_classes,
    #                                                     pretrained=False)
    #
    #     # 加载预训练权重（除了分类层）
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     pretrained_dict = torch.load(self.model_path, map_location=device)
    #
    #     # 假设model的最后一层是名为'head'或'classifier'的全连接层
    #     # 这里需要根据您的模型架构来确定正确的层名
    #     model_dict = self.model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                        not k.endswith('head.weight') and not k.endswith('head.bias')}
    #
    #     # 加载预训练权重到模型中
    #     model_dict.update(pretrained_dict)
    #     self.model.load_state_dict(model_dict)
    #
    #     # 如果需要，可以将模型设置为评估模式
    #     self.model.eval()
    #     print('{} model, and custom classes loaded.'.format(self.model_path))
    #
    #     # 如果在GPU上运行
    #     if self.cuda:
    #         self.model = nn.DataParallel(self.model)
    #         self.model = self.model.to(device)  # 使用.to(device)而不是.cuda()
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        #---------------------------------------------------#
        #   获得所属种类
        #---------------------------------------------------#
        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        #---------------------------------------------------#
        #   绘图并写字
        #---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        plt.show()
        return class_name
