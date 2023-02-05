# MS-COCO-Detections-Embedding-CNN
Uses Efficientnet to create embeddings from Retinanet detections which can then be used to refine MS COCO detection classes.

This model allows you to refine classifications made by an MS COCO trained detector (in this case Retinanet) by generating detection embeddings which can then be compared to ground truth images. This allows you to do things like split the MS COCO class 'car' into 'red car', 'blue car' etc without having to retrain/train any models.

Model details:

![image](https://user-images.githubusercontent.com/35029869/216811627-47d49937-d887-49be-951f-81cef3ae6650.png)


After a set of detection embeddings are generated for an image, they are compared to the ‘ground truth embeddings’
Each detection embedding is then classified as it’s nearest neighbour in the ground truth embeddings.

Each detection embedding is then classified as it’s nearest neighbour in the ground truth embeddings.

It is currently configured to search the Retinanet 'truck' detections (MS COCO Class 8) for military equipment (missile launchers, self propelled artillery etc).

To use this model for other purposes you must make 3 changes:

1. Update 'directory' on Line 14 to the directory containing the images you want to classify.
2. Change 'gt_images' on Line 21 to contain the images you want to use as the ground truths.
3. Change coco_class_list to the relevant MS COCO classes you want to refine.

Thanks for reading/considering using this model.


