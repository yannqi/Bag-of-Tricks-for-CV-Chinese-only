# 训练调参技巧总结

## 通用技巧

1. 使用混合精度 Automatic Mixed Precision。与单精度 (FP32) 相比，一些运算在不损失准确率的情况下，使用半精度 (FP16)速度更快。AMP能够自动决定应该以哪种精度执行哪种运算，这样既可以加快训练速度，又减少了内存占用。

2. 梯度裁剪，当神经网络深度逐渐增加，网络参数量增多的时候，反向传播过程中链式法则里的梯度连乘项数便会增多，更易引起梯度消失和梯度爆炸。对于梯度爆炸问题，解决方法之一便是进行梯度剪裁，即设置一个梯度大小的上限。

    梯度处理的过程：计算梯度$\rightarrow$裁剪梯度$\rightarrow$更新网络参数
    ```torch.nn.utils.clip_grad_norm_()``` 的使用应该在```loss.backward()``` 之后，```optimizer.step()```之前：
    ```...
    loss = criterion(...)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm) #
    optimizer.step()
    ...
    ```
clip_grad_norm和Amp的代码综合在`SpeedUp/amp_clip_grad_norm.py`内。
### 数据预处理技巧
1. Data Aug的处理。可以参考他人模型的Data Aug的处理方式，但也要有一个度。例如，在行人重识别检测中，你把图像水平翻转可以理解，垂直翻转就不得行了(没人头倒着走)。

### 学习率调整技巧
1. 学习率随Batch_size变化篇，learning_rate = init_learning_rate * number_gpu * batch_size / base_batch_size

2. 余弦退火策略。learning rate scheduler的使用：推荐使用cosine annealing strategy ，相比于step decay，cosine annealing strategy，其Accuracy涨幅更加均匀，且最终在ResNet分类上Top-1能提点0.5左右。[COSINEANNEALINGLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)

3. warmup，热处理启动。一开始采用较低的学习率，等训练一段时间之后再采用较高的学习率并且随着学习率衰减策略衰减。能够有效的防止模型在较大学习率的时候由于步长过大很快陷入一个局部最小值区域的问题。具体实现方式见代码`Learning Rate/warmup_cosine_annealing.py`。(针对warmup可以写一个规范的代码加Blog)



## 分类篇

### 损失函数技巧
在使用交叉熵损失函数时，可以使用smooth label，有效防止过拟合。` criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)`

## 目标检测篇

## 语义分割篇

各种涨点小技巧汇总（均有浮动，非绝对）：

| 技巧 |有效涨点|
|-|-|
|在test期间使用Multi_scale input|~1%|
|在test期间使用Flip进行翻转|~0.4%|
|在VOC数据集使用COCO进行预训练|~3%|
|大的Crop_size可以更加有效的涨点|例如从321x321到513x513，可以有效涨点 ~10%！ （但也有上限，根据图像的尺寸上限）|
|在train阶段，将图像还原回原图像尺寸之后，再计算损失，可以有效涨点|~1.3%|
|冻结BN层，再去fine-tuning其它参数，可有效涨点|~1.3%|


# 提速篇
1. 如果想提高训练速度，可以使用`torch.optim.lr_scheduler.OneCycleLR`或者`torch.optim.lr_scheduler.CyclicLR`，同等精度下，可以提高10x训练速度。
2. 设置DataLoader的num_workers>0和pin_memory=True，可以约提高2x训练速度。  num_workers的数目，通常可以设置成GPU数目的4倍，过大过小都不好。
3. 想提速，有限GPU内存内，最大化batch_size就行了。
4. 设置AMP(Automatic Mixed Precision)，自动混合精度。混合FP16和FP32精度，在不损失模型性能的前提下，提高2x~5x训练速度。
    ```python
    import torch
    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()

    for data, label in data_iter:
    optimizer.zero_grad()
    # Casts operations to mixed precision
    with torch.cuda.amp.autocast():
        loss = model(data)

    # Scales the loss, and calls backward()
    # to create scaled gradients
    scaler.scale(loss).backward()

    # Unscales gradients and calls
    # or skips optimizer.step()
    scaler.step(optimizer)

    # Updates the scale for next iteration
    scaler.update()
    ```
5. 打开cudNN benchmarking，如果模型的输入尺寸是固定的，则可以打开`torch.backends.cudnn.benchmark = True`，带来明显的速度提升。

6.

# 参考
[An overview of some of the lowest-effort, highest-impact ways of accelerating the training of deep learning models in PyTorch.](https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/)


[Pytorch example code](https://github.com/pytorch/vision/blob/1d0786b0a35661408388ed4268e382f56bcde627/references/classification/train.py)