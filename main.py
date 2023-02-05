import torchvision
from torchvision.io import read_image
from torchvision.models import efficientnet_b4
import torch
from torchvision.transforms import functional as F, InterpolationMode
import pandas as pd
import os

########## THIS CODE WILL ONLY RUN ON A COMPUTER WITH CUDA DEVICE AVAILABLE (i.e. GPU) ##########
# If you want to run on a CPU only device, change "cuda" to "cpu". Performance of the model may be poor on CPU.
device = torch.device("cuda:0")

# Build a map of the image data on persistent storage for use by the model.
directory = '/media/michael/SecondDisk/GDELT_intl/'
ind_files = []

for (root, dirs, files) in os.walk(directory, topdown=True):
    ind_files = files

# Map of ground truth images for comparison.
gt_images = ["fseries_truck.jpeg", "container_truck.jpeg", "dump_truck1.jpeg", "intl_truck.jpg",
             "dump_truck2.jpeg", "dump_truck3.jpeg", "dump_truck4.jpeg", "dump_truck5.jpeg", "himars.jpeg",
             "himars2.jpeg"]


def calc_eucledian_distance_batch(tensor1, tensor2):

    # This function calculates the Eucledian distances between two batches of image tensors.

    output = torch.zeros([tensor1.shape[0], tensor2.shape[0]])

    for i in range(0, tensor1.shape[0]):
        for k in range(0, tensor2.shape[0]):
            tmp = torch.sub(tensor1[i], tensor2[k])
            tmp = torch.pow(tmp, 2)
            output[i, k] = torch.sum(torch.sqrt(tmp))

    return output


class ImageClassification(torch.nn.Module):

    # This class was copied from current version torchvision file and modified to allow backwards compatability
    # for an earlier version of torchvision. It performs the data preprocessing step for an imagenet trained
    # efficientnet

    def __init__(self, crop_size, resize_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size, resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = InterpolationMode.BICUBIC

    def forward(self, img):

        img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        # Changed from original ImageNet trained class. Don't do center crop again,
        # as object detector has already set the bounding boxes for us.
        if self.crop_size[0] != 0:
            img = F.center_crop(img, self.crop_size)
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img


class EfficientNetModel(torch.nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return x


class EmbeddingExtractor(torch.nn.Module):

    def __init__(self, detector, embedding_net, embedding_transformer, coco_classes, threshold):

        super().__init__()
        self.detector = detector
        self.embedding_net = embedding_net
        self.embedding_transformer = embedding_transformer
        self.coco_classes = coco_classes
        self.threshold = threshold

    def forward(self, x):

        preds = self.detector([x])
        boxes = preds[0]['boxes']
        scores = preds[0]['scores']
        classes = preds[0]['labels']
        is_done = False
        indx = 0

        tensor_batch = []

        while not is_done:
            if scores[indx] > self.threshold:
                if classes[indx] in self.coco_classes:
                    x1 = int(boxes[indx, 0].item())
                    y1 = int(boxes[indx, 1].item())
                    x2 = int(boxes[indx, 2].item())
                    y2 = int(boxes[indx, 3].item())
                    # Just a minimum area check. If detection is too small it may not contain enough information
                    # to allow a useful embedding comparison
                    if (x2 - x1) < 50 and (y2 - y1) < 50:
                        indx = indx + 1
                        continue
                    # Ok, so we have a detection. Let's crop it and add it to the tensor batch.
                    t1 = x[:, y1:y2, x1:x2]
                    t1 = self.embedding_transformer(t1)
                    tensor_batch.append(t1)
                indx = indx + 1
            else:
                is_done = True

        # Stack all the detection tensors to pass to EfficientNet to generate the embeddings.
        tb_len = len(tensor_batch)
        if tb_len != 0:
            input_ready = torch.stack(tensor_batch)
            emb_final = self.embedding_net(input_ready)
            return emb_final

#In the example - we just want to use trucks to get a detailed test evaluation. This may not exist in production code.
coco_class_list = [8]

# Step 1: Initialize the detection and localization models with the best available weights.
# Trained on Imagenet and MS COCO respectively.
transforms = ImageClassification(crop_size=0, resize_size=384)
backbone = efficientnet_b4(pretrained=True)
model = EfficientNetModel(backbone)
model.eval()
model_detect = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model_detect.eval()

# Step 2: Use the detection and localization models to create the Embedding Extractor and move it to the GPU
ModelEmbeddingExtractor = EmbeddingExtractor(model_detect, model, transforms, coco_class_list, 0.25)
ModelEmbeddingExtractor.to(device)

tensor_batch = None
list_vals = []

# Step 3: Generate the embeddings of the image data linked in GDELT using the model.
for file in ind_files:
    try:
        full_path = directory + file
        img = read_image(full_path)
        img = F.convert_image_dtype(img, torch.float)
        img = img.to(device)
        out1 = ModelEmbeddingExtractor(img)

        for i in range(0, out1.shape[0]):
            list_vals.append(file)

        if tensor_batch is None:
            tensor_batch = out1.detach()
        else:
            tensor_batch = torch.concat([tensor_batch, out1.detach()])
    #@TODO: Refine exception handling
    except:
        continue

gt_embeddings = None
embedding_labels = []

# Step 4: Generate the embeddings of the ground truth image data
for file in gt_images:
    try:
        full_path = file
        img = read_image(full_path)
        img = F.convert_image_dtype(img, torch.float)
        img = img.to(device)
        out1 = ModelEmbeddingExtractor(img)

        for i in range(0, out1.shape[0]):
            embedding_labels.append(file)

        if gt_embeddings is None:
            gt_embeddings = out1.detach()
        else:
            gt_embeddings = torch.concat([gt_embeddings, out1.detach()])
    #@TODO: Refine exception handling
    except:
        continue

# Step 5: Compare the embeddings of the ground truth image data and GDELT imaged data to get Eucledian Distances.
distances = calc_eucledian_distance_batch(gt_embeddings, tensor_batch)

# Step 6: Get the nearest neighbour and save results.
t_np = distances.numpy() #convert to Numpy array
df = pd.DataFrame(t_np, columns=list_vals) #convert to a dataframe
# Get index of nearest neighbour
bz = df.idxmin()
# Write to output files
for i in range(0, len(list_vals)):
    print("Detection:" + list_vals[i] + " " + "NN:" + embedding_labels[bz.iloc[i]])
df.insert(0, "comp_val", embedding_labels, allow_duplicates=True)
df.to_csv("testfile", index=False)
