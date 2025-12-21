- - -

- [x] 将每个样本的一帧转化为一个张量，用于输入给CNN
- [x] 编写处理RGB模态的CNN，用于获取每一帧的表示
- [x] 按时序处理每一帧表示的GRU/LSTM，输出预测结果
- [x] 尝试编写训练脚本
- [x] 训练并提交
- [x] 查看LSTM是否是欠拟合问题所在：实验结果如下
    ```text
    不采用LSTM：严重的欠拟合

    Using device: cuda
    Epoch 1/20 | Time 538.0s | Train loss 2.9752 acc 0.0975 | Val loss 2.8514 acc 0.2188
    Epoch 2/20 | Time 800.6s | Train loss 2.8727 acc 0.1356 | Val loss 2.7066 acc 0.2891
    Epoch 3/20 | Time 768.3s | Train loss 2.7569 acc 0.1738 | Val loss 2.5268 acc 0.4297
    Epoch 4/20 | Time 745.3s | Train loss 2.6405 acc 0.2535 | Val loss 2.3624 acc 0.5625
    Epoch 5/20 | Time 727.6s | Train loss 2.5098 acc 0.3227 | Val loss 2.1527 acc 0.6719
    Epoch 6/20 | Time 744.7s | Train loss 2.3776 acc 0.3582 | Val loss 1.9907 acc 0.7188
    Epoch 7/20 | Time 747.8s | Train loss 2.2416 acc 0.4362 | Val loss 1.8405 acc 0.6875
    Epoch 8/20 | Time 747.3s | Train loss 2.1590 acc 0.4637 | Val loss 1.7509 acc 0.6562
    Epoch 9/20 | Time 749.0s | Train loss 2.0860 acc 0.5053 | Val loss 1.6859 acc 0.6953
    Epoch 10/20 | Time 733.2s | Train loss 2.0768 acc 0.4814 | Val loss 1.6275 acc 0.7109
    Epoch 11/20 | Time 731.4s | Train loss 2.0293 acc 0.4876 | Val loss 1.5546 acc 0.6953

    采用1层隐藏层的LSTM：
    Using device: cuda                                                                                       
    Epoch 1/20 | Time 774.9s | Train loss 2.9123 acc 0.1525 | Val loss 2.7700 acc 0.2812
    Epoch 2/20 | Time 781.6s | Train loss 2.6135 acc 0.2739 | Val loss 2.3737 acc 0.3594
    Epoch 3/20 | Time 776.9s | Train loss 2.2911 acc 0.3910 | Val loss 2.0750 acc 0.4688
    Epoch 4/20 | Time 786.8s | Train loss 2.0050 acc 0.4902 | Val loss 1.8288 acc 0.4922
    Epoch 5/20 | Time 771.2s | Train loss 1.7995 acc 0.5479 | Val loss 1.5997 acc 0.5312
    Epoch 6/20 | Time 785.1s | Train loss 1.6274 acc 0.5683 | Val loss 1.4435 acc 0.6016
    Epoch 7/20 | Time 776.7s | Train loss 1.4038 acc 0.6480 | Val loss 1.2592 acc 0.6719
    Epoch 8/20 | Time 783.0s | Train loss 1.2667 acc 0.6693 | Val loss 1.2155 acc 0.6328
    Epoch 9/20 | Time 800.0s | Train loss 1.1496 acc 0.7074 | Val loss 1.1135 acc 0.6797
    Epoch 10/20 | Time 761.1s | Train loss 1.1048 acc 0.7287 | Val loss 1.0523 acc 0.6953
    Epoch 11/20 | Time 441.9s | Train loss 1.0235 acc 0.7447 | Val loss 1.0173 acc 0.6719

    ---

    采用1层隐藏层的LSTM：
    Epoch 1/20 | Time 777.3s | Train loss 2.9469 acc 0.0931 | Val loss 2.8081 acc 0.3125
    Epoch 2/20 | Time 783.1s | Train loss 2.7245 acc 0.2855 | Val loss 2.3766 acc 0.5625
    Epoch 3/20 | Time 780.4s | Train loss 2.3400 acc 0.4158 | Val loss 1.9116 acc 0.6328
    Epoch 4/20 | Time 781.6s | Train loss 1.9680 acc 0.5044 | Val loss 1.5631 acc 0.6172
    Epoch 5/20 | Time 742.7s | Train loss 1.7429 acc 0.5470 | Val loss 1.3973 acc 0.6406
    Epoch 6/20 | Time 761.7s | Train loss 1.5558 acc 0.5931 | Val loss 1.2292 acc 0.6562
    Epoch 7/20 | Time 772.5s | Train loss 1.3961 acc 0.6170 | Val loss 1.0651 acc 0.6328
    Epoch 8/20 | Time 777.4s | Train loss 1.2236 acc 0.6543 | Val loss 1.0135 acc 0.6719
    Epoch 9/20 | Time 788.4s | Train loss 1.1423 acc 0.7012 | Val loss 0.9069 acc 0.7578
    Epoch 10/20 | Time 786.8s | Train loss 1.0725 acc 0.7181 | Val loss 0.8653 acc 0.7422
    Epoch 11/20 | Time 772.1s | Train loss 1.0214 acc 0.7234 | Val loss 0.7894 acc 0.8203

    采用2层隐藏层的LSTM：
    Using device: cuda
    Epoch 1/20 | Time 773.4s | Train loss 2.9386 acc 0.1179 | Val loss 2.7843 acc 0.2969
    Epoch 2/20 | Time 758.8s | Train loss 2.5304 acc 0.2686 | Val loss 2.1087 acc 0.3281
    Epoch 3/20 | Time 745.5s | Train loss 2.0406 acc 0.4264 | Val loss 1.6800 acc 0.5547
    Epoch 4/20 | Time 755.9s | Train loss 1.7019 acc 0.4929 | Val loss 1.3596 acc 0.5391
    Epoch 5/20 | Time 758.6s | Train loss 1.5398 acc 0.5612 | Val loss 1.1880 acc 0.5547
    Epoch 6/20 | Time 761.8s | Train loss 1.3700 acc 0.5762 | Val loss 1.0844 acc 0.6250
    Epoch 7/20 | Time 742.9s | Train loss 1.2022 acc 0.6383 | Val loss 1.0012 acc 0.6328
    Epoch 8/20 | Time 750.4s | Train loss 1.0619 acc 0.6383 | Val loss 0.8886 acc 0.6406
    Epoch 9/20 | Time 738.8s | Train loss 0.9914 acc 0.6915 | Val loss 0.8716 acc 0.6719
    Epoch 10/20 | Time 723.7s | Train loss 0.9637 acc 0.6897 | Val loss 0.7905 acc 0.7188
    Epoch 11/20 | Time 736.7s | Train loss 0.9003 acc 0.7394 | Val loss 0.7376 acc 0.7344
    ```
- [x] 调整训练超参数以解决欠拟合问题：增加批量大小、删去dropout层、取消权重衰减
    ```text
    Using device: cuda
    Epoch 1/20 | Time 364.0s | Train loss 2.9338 acc 0.1915 | Val loss 2.8730 acc 0.3526
    Epoch 2/20 | Time 360.3s | Train loss 2.7681 acc 0.4846 | Val loss 2.6774 acc 0.5302
    Epoch 3/20 | Time 360.6s | Train loss 2.5700 acc 0.6195 | Val loss 2.4314 acc 0.6576
    Epoch 4/20 | Time 355.6s | Train loss 2.2806 acc 0.7032 | Val loss 2.1351 acc 0.6835
    Epoch 5/20 | Time 358.7s | Train loss 1.9729 acc 0.7403 | Val loss 1.8287 acc 0.6905
    Epoch 6/20 | Time 359.8s | Train loss 1.6466 acc 0.7579 | Val loss 1.5226 acc 0.7139
    Epoch 7/20 | Time 358.6s | Train loss 1.3535 acc 0.8173 | Val loss 1.2716 acc 0.8015
    Epoch 8/20 | Time 359.2s | Train loss 1.1431 acc 0.8631 | Val loss 1.1487 acc 0.8335
    Epoch 9/20 | Time 357.1s | Train loss 1.0166 acc 0.8902 | Val loss 1.0432 acc 0.8726
    Epoch 10/20 | Time 359.0s | Train loss 0.9243 acc 0.9042 | Val loss 0.9518 acc 0.9038
    Epoch 11/20 | Time 358.2s | Train loss 0.8571 acc 0.9032 | Val loss 0.8780 acc 0.9038
    Epoch 12/20 | Time 359.9s | Train loss 0.7749 acc 0.9292 | Val loss 0.8144 acc 0.9038
    Epoch 13/20 | Time 358.0s | Train loss 0.6994 acc 0.9266 | Val loss 0.7453 acc 0.9046
    Epoch 14/20 | Time 360.5s | Train loss 0.6362 acc 0.9417 | Val loss 0.6902 acc 0.9124
    Epoch 15/20 | Time 358.5s | Train loss 0.6052 acc 0.9495 | Val loss 0.6605 acc 0.9124
    Epoch 16/20 | Time 358.6s | Train loss 0.5689 acc 0.9479 | Val loss 0.6335 acc 0.9124
    Epoch 17/20 | Time 358.4s | Train loss 0.5581 acc 0.9485 | Val loss 0.6111 acc 0.9046
    Epoch 18/20 | Time 359.9s | Train loss 0.5180 acc 0.9453 | Val loss 0.5852 acc 0.9046
    Epoch 19/20 | Time 359.2s | Train loss 0.5287 acc 0.9407 | Val loss 0.5614 acc 0.9133
    Epoch 20/20 | Time 358.2s | Train loss 0.4854 acc 0.9511 | Val loss 0.5512 acc 0.9289
    ```
- [x] 继续训练上述模型，直至最终收敛以查看效果
    ```bash
    python Resnet18/src/train.py --resume checkpoints_batchsize32_and_no_reg/best.pth --epochs 50 --only_rgb --ckpt_dir checkpoints_batchsize32_and_no_reg >> log.txt
    ```
    结果：
    ```text
    Using device: cuda
    Loaded checkpoint from epoch 21
    Epoch 22/50 | Time 361.7s | Train loss 0.4382 acc 0.9635 | Val loss 0.5184 acc 0.9367
    Epoch 23/50 | Time 358.4s | Train loss 0.4417 acc 0.9573 | Val loss 0.5105 acc 0.9445
    Epoch 24/50 | Time 359.8s | Train loss 0.4248 acc 0.9563 | Val loss 0.4991 acc 0.9367
    Epoch 25/50 | Time 358.4s | Train loss 0.4117 acc 0.9661 | Val loss 0.4893 acc 0.9445
    Epoch 26/50 | Time 358.8s | Train loss 0.4024 acc 0.9661 | Val loss 0.4797 acc 0.9445
    Epoch 27/50 | Time 356.5s | Train loss 0.3985 acc 0.9661 | Val loss 0.4701 acc 0.9445
    Epoch 28/50 | Time 355.6s | Train loss 0.3824 acc 0.9599 | Val loss 0.4638 acc 0.9445
    Epoch 29/50 | Time 358.5s | Train loss 0.3779 acc 0.9667 | Val loss 0.4611 acc 0.9445
    Epoch 30/50 | Time 357.5s | Train loss 0.3877 acc 0.9661 | Val loss 0.4588 acc 0.9445
    Epoch 31/50 | Time 360.8s | Train loss 0.3786 acc 0.9599 | Val loss 0.4548 acc 0.9445
    Epoch 32/50 | Time 364.7s | Train loss 0.3793 acc 0.9651 | Val loss 0.4531 acc 0.9445
    Epoch 33/50 | Time 367.5s | Train loss 0.3622 acc 0.9625 | Val loss 0.4481 acc 0.9445
    Epoch 34/50 | Time 366.3s | Train loss 0.3806 acc 0.9635 | Val loss 0.4447 acc 0.9445
    Epoch 35/50 | Time 361.0s | Train loss 0.3739 acc 0.9615 | Val loss 0.4419 acc 0.9445
    Epoch 36/50 | Time 359.9s | Train loss 0.3535 acc 0.9693 | Val loss 0.4400 acc 0.9445
    Epoch 37/50 | Time 359.7s | Train loss 0.3727 acc 0.9667 | Val loss 0.4354 acc 0.9445
    Epoch 38/50 | Time 364.2s | Train loss 0.3464 acc 0.9714 | Val loss 0.4356 acc 0.9445
    Epoch 39/50 | Time 366.6s | Train loss 0.3538 acc 0.9740 | Val loss 0.4327 acc 0.9445
    Epoch 40/50 | Time 366.3s | Train loss 0.3461 acc 0.9688 | Val loss 0.4310 acc 0.9445
    Epoch 41/50 | Time 360.7s | Train loss 0.3517 acc 0.9651 | Val loss 0.4298 acc 0.9445
    Epoch 42/50 | Time 362.8s | Train loss 0.3653 acc 0.9641 | Val loss 0.4260 acc 0.9445
    Epoch 43/50 | Time 361.1s | Train loss 0.3494 acc 0.9693 | Val loss 0.4270 acc 0.9445
    Epoch 44/50 | Time 365.3s | Train loss 0.3378 acc 0.9714 | Val loss 0.4241 acc 0.9445
    Epoch 45/50 | Time 362.8s | Train loss 0.3683 acc 0.9657 | Val loss 0.4245 acc 0.9445
    Epoch 46/50 | Time 366.4s | Train loss 0.3430 acc 0.9693 | Val loss 0.4237 acc 0.9445
    Epoch 47/50 | Time 365.3s | Train loss 0.3569 acc 0.9641 | Val loss 0.4260 acc 0.9445
    Epoch 48/50 | Time 368.5s | Train loss 0.3640 acc 0.9615 | Val loss 0.4217 acc 0.9445
    Epoch 49/50 | Time 368.5s | Train loss 0.3440 acc 0.9729 | Val loss 0.4222 acc 0.9445
    Epoch 50/50 | Time 364.3s | Train loss 0.3400 acc 0.9677 | Val loss 0.4229 acc 0.9445
    Resume training finished. Best val acc: 0.9445043057203293
    ```

- [x] 使用`torch.optim.lr_scheduler.ReduceLROnPlateau`动态调整学习率
    ```bash
    python Resnet18/src/train.py --resume checkpoints_batchsize32_and_no_reg/ckpt_epoch20.pth --epochs 30 --only_rgb --ckpt_dir checkpoints_batchsize32_and_no_reg/plateau_scheduler --scheduler_type plateau --patience 5 --lr_factor 0.1 --min_lr 1e-6 >> log.txt
    ```
    结果：
    ```text
    Loaded checkpoint from epoch 20
    Epoch 21/30 | Time 363.1s | Train loss 0.4621 acc 0.9583 | Val loss 0.5339 acc 0.9445
    Epoch 22/30 | Time 361.1s | Train loss 0.4321 acc 0.9661 | Val loss 0.5119 acc 0.9367
    Epoch 23/30 | Time 360.6s | Train loss 0.4320 acc 0.9599 | Val loss 0.4888 acc 0.9445
    Epoch 24/30 | Time 364.6s | Train loss 0.4146 acc 0.9667 | Val loss 0.4753 acc 0.9445
    Epoch 25/30 | Time 361.1s | Train loss 0.3971 acc 0.9661 | Val loss 0.4617 acc 0.9367
    Epoch 26/30 | Time 369.6s | Train loss 0.3752 acc 0.9714 | Val loss 0.4440 acc 0.9445
    Epoch 27/30 | Time 360.7s | Train loss 0.3796 acc 0.9625 | Val loss 0.4274 acc 0.9445
    Epoch 28/30 | Time 360.8s | Train loss 0.3701 acc 0.9625 | Val loss 0.4273 acc 0.9445
    Epoch 29/30 | Time 360.8s | Train loss 0.3506 acc 0.9703 | Val loss 0.4255 acc 0.9445
    Epoch 30/30 | Time 359.5s | Train loss 0.3385 acc 0.9688 | Val loss 0.4220 acc 0.9445
    Resume training finished. Best val acc: 0.9445043057203293
    ```

- [x] 支线：解冻Resnet18主干进行微调10个epoch
    ```bash
    python Resnet18/src/train.py --resume checkpoints_batchsize32_and_no_reg/ckpt_epoch20.pth --epochs 30 --only_rgb --ckpt_dir checkpoints_batchsize32_and_no_reg/resnet18_finetune >> log.txt
    ```
    结果：
    ```text
    Loaded checkpoint from epoch 20
    Warning: Optimizer state has different number of parameter groups (likely due to unfreezing backbone). Skipping optimizer state loading.
    Epoch 21/30 | Time 195.2s | Train loss 0.5043 acc 0.9407 | Val loss 0.4707 acc 0.9523
    Epoch 22/30 | Time 184.7s | Train loss 0.3825 acc 0.9635 | Val loss 0.4163 acc 0.9445
    Epoch 23/30 | Time 183.7s | Train loss 0.3338 acc 0.9755 | Val loss 0.3581 acc 0.9523
    Epoch 24/30 | Time 182.2s | Train loss 0.2857 acc 0.9792 | Val loss 0.3173 acc 0.9601
    Epoch 25/30 | Time 181.2s | Train loss 0.2391 acc 0.9834 | Val loss 0.2910 acc 0.9601
    Epoch 26/30 | Time 181.9s | Train loss 0.2226 acc 0.9844 | Val loss 0.2668 acc 0.9523
    Epoch 27/30 | Time 182.3s | Train loss 0.1972 acc 0.9870 | Val loss 0.2462 acc 0.9445
    Epoch 28/30 | Time 181.2s | Train loss 0.1717 acc 0.9870 | Val loss 0.2260 acc 0.9523
    Epoch 29/30 | Time 182.6s | Train loss 0.1644 acc 0.9912 | Val loss 0.2139 acc 0.9601
    Epoch 30/30 | Time 182.8s | Train loss 0.1575 acc 0.9922 | Val loss 0.2022 acc 0.9601
    Resume training finished. Best val acc: 0.9601293057203293
    ```

- [x] 支线：验证此模型是否过拟合；采用全量微调后在测试集上再次尝试——成功！
    ```bash
    python -u Resnet18/src/train.py --resume checkpoints_batchsize32_and_no_reg/resnet18_finetune/best.pth --epochs 40 --train_full --only_rgb --ckpt_dir checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full --frames_per_clip 32 >> log.txt
    ```
    结果：
    ```text
    Using device: cuda
    Loaded checkpoint from epoch 24
    Optimizer state loaded successfully.
    Epoch 25/40 | Time 188.2s | Train loss 0.2699 acc 0.9746
    Epoch 26/40 | Time 186.9s | Train loss 0.2392 acc 0.9766
    Epoch 27/40 | Time 185.8s | Train loss 0.2053 acc 0.9852
    Epoch 28/40 | Time 185.3s | Train loss 0.1620 acc 0.9863
    Epoch 29/40 | Time 184.5s | Train loss 0.1460 acc 0.9902
    Epoch 30/40 | Time 184.5s | Train loss 0.1417 acc 0.9879
    Epoch 31/40 | Time 184.9s | Train loss 0.1261 acc 0.9902
    Epoch 32/40 | Time 184.2s | Train loss 0.1179 acc 0.9922
    Epoch 33/40 | Time 182.4s | Train loss 0.1128 acc 0.9961
    Epoch 34/40 | Time 182.3s | Train loss 0.1069 acc 0.9898
    Epoch 35/40 | Time 183.3s | Train loss 0.0978 acc 0.9961
    Epoch 36/40 | Time 183.1s | Train loss 0.0961 acc 0.9949
    Epoch 37/40 | Time 182.9s | Train loss 0.0887 acc 1.0000
    Epoch 38/40 | Time 183.1s | Train loss 0.0849 acc 0.9980
    Epoch 39/40 | Time 183.2s | Train loss 0.0862 acc 0.9949
    Epoch 40/40 | Time 184.6s | Train loss 0.0816 acc 1.0000
    Resume training finished. Best train loss: 0.08161398093216121
    ```

- [x] 回归主线：采用支线的预训练Resnet18构建其他模态的独立模型
    ```bash
    # infrared
    python -u Resnet18/src/train.py --freeze_backbone --pretrained_weights_path checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full/best.pth --epochs 40 --modalities infrared --ckpt_dir checkpoints_batchsize32_and_no_reg/infrared_finetune --frames_per_clip 32 >> log.txt
    ```
    红外训练结果：
    ```text
    Using device: cuda
    Loading ResNet weights from checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full/best.pth
    Custom ResNet weights loaded successfully.
    Epoch 1/40 | Time 184.8s | Train loss 2.9329 acc 0.1051 | Val loss 2.9631 acc 0.1767
    Epoch 2/40 | Time 181.8s | Train loss 2.7899 acc 0.4321 | Val loss 2.9047 acc 0.2002
    Epoch 3/40 | Time 179.9s | Train loss 2.5914 acc 0.5508 | Val loss 2.7892 acc 0.2939
    Epoch 4/40 | Time 180.6s | Train loss 2.3104 acc 0.6846 | Val loss 2.6068 acc 0.4582
    Epoch 5/40 | Time 180.4s | Train loss 1.9912 acc 0.7090 | Val loss 2.3065 acc 0.5294
    Epoch 6/40 | Time 180.7s | Train loss 1.6336 acc 0.7585 | Val loss 1.9311 acc 0.5450
    Epoch 7/40 | Time 179.9s | Train loss 1.3386 acc 0.8236 | Val loss 1.5493 acc 0.6880
    Epoch 8/40 | Time 180.4s | Train loss 1.1064 acc 0.8850 | Val loss 1.3224 acc 0.7772
    Epoch 9/40 | Time 180.3s | Train loss 0.9773 acc 0.9048 | Val loss 1.1123 acc 0.8405
    Epoch 10/40 | Time 181.4s | Train loss 0.8891 acc 0.9136 | Val loss 0.9198 acc 0.8882
    Epoch 11/40 | Time 180.2s | Train loss 0.7856 acc 0.9365 | Val loss 0.8127 acc 0.8882
    Epoch 12/40 | Time 181.0s | Train loss 0.7065 acc 0.9443 | Val loss 0.7483 acc 0.9116
    Epoch 13/40 | Time 181.9s | Train loss 0.6280 acc 0.9531 | Val loss 0.6815 acc 0.9203
    Epoch 14/40 | Time 182.1s | Train loss 0.5592 acc 0.9677 | Val loss 0.6215 acc 0.9359
    Epoch 15/40 | Time 181.3s | Train loss 0.5195 acc 0.9688 | Val loss 0.5948 acc 0.9281
    Epoch 16/40 | Time 181.9s | Train loss 0.4762 acc 0.9818 | Val loss 0.5685 acc 0.9515
    Epoch 17/40 | Time 180.7s | Train loss 0.4509 acc 0.9870 | Val loss 0.5409 acc 0.9515
    Epoch 18/40 | Time 181.0s | Train loss 0.4261 acc 0.9844 | Val loss 0.5210 acc 0.9437
    Epoch 19/40 | Time 181.9s | Train loss 0.4039 acc 0.9896 | Val loss 0.5055 acc 0.9515
    Epoch 20/40 | Time 182.1s | Train loss 0.4083 acc 0.9834 | Val loss 0.4836 acc 0.9593
    Epoch 21/40 | Time 180.9s | Train loss 0.4118 acc 0.9777 | Val loss 0.4684 acc 0.9515
    Epoch 22/40 | Time 182.0s | Train loss 0.3542 acc 0.9912 | Val loss 0.4593 acc 0.9437
    Epoch 23/40 | Time 180.9s | Train loss 0.3618 acc 0.9886 | Val loss 0.4536 acc 0.9593
    Epoch 24/40 | Time 179.8s | Train loss 0.3442 acc 0.9886 | Val loss 0.4446 acc 0.9515
    Epoch 25/40 | Time 180.2s | Train loss 0.3284 acc 0.9974 | Val loss 0.4364 acc 0.9593
    Epoch 26/40 | Time 178.7s | Train loss 0.3237 acc 0.9922 | Val loss 0.4284 acc 0.9593
    Epoch 27/40 | Time 179.1s | Train loss 0.3129 acc 0.9922 | Val loss 0.4211 acc 0.9515
    Epoch 28/40 | Time 178.5s | Train loss 0.3017 acc 0.9974 | Val loss 0.4131 acc 0.9593
    Epoch 29/40 | Time 179.6s | Train loss 0.3079 acc 0.9948 | Val loss 0.4065 acc 0.9593
    Epoch 30/40 | Time 179.1s | Train loss 0.2897 acc 0.9948 | Val loss 0.4034 acc 0.9593
    Epoch 31/40 | Time 181.2s | Train loss 0.2840 acc 0.9974 | Val loss 0.3993 acc 0.9593
    Epoch 32/40 | Time 179.0s | Train loss 0.2829 acc 0.9948 | Val loss 0.3994 acc 0.9593
    Epoch 33/40 | Time 179.5s | Train loss 0.2815 acc 0.9948 | Val loss 0.3967 acc 0.9593
    Epoch 34/40 | Time 180.7s | Train loss 0.2800 acc 0.9964 | Val loss 0.3982 acc 0.9593
    Epoch 35/40 | Time 179.7s | Train loss 0.2726 acc 0.9974 | Val loss 0.3925 acc 0.9593
    Epoch 36/40 | Time 180.8s | Train loss 0.2711 acc 0.9974 | Val loss 0.3901 acc 0.9593
    Epoch 37/40 | Time 181.0s | Train loss 0.2810 acc 0.9948 | Val loss 0.3906 acc 0.9593
    Epoch 38/40 | Time 180.3s | Train loss 0.2649 acc 0.9964 | Val loss 0.3855 acc 0.9593
    Epoch 39/40 | Time 180.9s | Train loss 0.2659 acc 0.9948 | Val loss 0.3831 acc 0.9593
    Epoch 40/40 | Time 179.9s | Train loss 0.2737 acc 0.9948 | Val loss 0.3830 acc 0.9593
    Training finished. Best val acc: 0.9593211263418198
    ```
    ```bash
    # depth
    python -u Resnet18/src/train.py --freeze_backbone --pretrained_weights_path checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full/best.pth --epochs 40 --modalities depth --ckpt_dir checkpoints_batchsize32_and_no_reg/depth_finetune --frames_per_clip 32 >> log.txt
    ```
    深度训练结果：
    ```text
    Loading ResNet weights from checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full/best.pth
    Custom ResNet weights loaded successfully.
    Epoch 1/40 | Time 184.0s | Train loss 2.9647 acc 0.1166 | Val loss 2.9770 acc 0.0555
    Epoch 2/40 | Time 180.0s | Train loss 2.8474 acc 0.2582 | Val loss 2.9310 acc 0.1837
    Epoch 3/40 | Time 178.1s | Train loss 2.6931 acc 0.4748 | Val loss 2.8584 acc 0.1915
    Epoch 4/40 | Time 180.0s | Train loss 2.4814 acc 0.6341 | Val loss 2.7279 acc 0.2400
    Epoch 5/40 | Time 181.5s | Train loss 2.2097 acc 0.6475 | Val loss 2.5791 acc 0.3112
    Epoch 6/40 | Time 178.7s | Train loss 1.9263 acc 0.7086 | Val loss 2.3953 acc 0.3925
    Epoch 7/40 | Time 178.1s | Train loss 1.6443 acc 0.7403 | Val loss 2.0492 acc 0.5294
    Epoch 8/40 | Time 178.4s | Train loss 1.4331 acc 0.7839 | Val loss 1.6393 acc 0.6670
    Epoch 9/40 | Time 177.1s | Train loss 1.3155 acc 0.7861 | Val loss 1.3843 acc 0.7069
    Epoch 10/40 | Time 178.0s | Train loss 1.2153 acc 0.8090 | Val loss 1.2533 acc 0.7530
    Epoch 11/40 | Time 178.5s | Train loss 1.1125 acc 0.8423 | Val loss 1.1793 acc 0.7452
    Epoch 12/40 | Time 177.6s | Train loss 1.0496 acc 0.8308 | Val loss 1.1173 acc 0.7702
    Epoch 13/40 | Time 178.6s | Train loss 0.9245 acc 0.8625 | Val loss 1.0332 acc 0.7936
    Epoch 14/40 | Time 177.6s | Train loss 0.8673 acc 0.8599 | Val loss 0.9812 acc 0.7608
    Epoch 15/40 | Time 177.7s | Train loss 0.7962 acc 0.8636 | Val loss 0.9390 acc 0.8015
    Epoch 16/40 | Time 177.3s | Train loss 0.7564 acc 0.9052 | Val loss 0.9129 acc 0.7936
    Epoch 17/40 | Time 178.3s | Train loss 0.7317 acc 0.8896 | Val loss 0.8866 acc 0.8015
    Epoch 18/40 | Time 179.6s | Train loss 0.7059 acc 0.9079 | Val loss 0.8573 acc 0.8171
    Epoch 19/40 | Time 180.0s | Train loss 0.6853 acc 0.9042 | Val loss 0.8326 acc 0.8015
    Epoch 20/40 | Time 180.8s | Train loss 0.6571 acc 0.9016 | Val loss 0.8036 acc 0.8093
    Epoch 21/40 | Time 180.5s | Train loss 0.6369 acc 0.9120 | Val loss 0.7850 acc 0.8093
    Epoch 22/40 | Time 182.1s | Train loss 0.6034 acc 0.9235 | Val loss 0.7730 acc 0.8093
    Epoch 23/40 | Time 182.5s | Train loss 0.5770 acc 0.9344 | Val loss 0.7561 acc 0.8171
    Epoch 24/40 | Time 178.1s | Train loss 0.5866 acc 0.9329 | Val loss 0.7441 acc 0.8015
    Epoch 25/40 | Time 177.8s | Train loss 0.5788 acc 0.9329 | Val loss 0.7402 acc 0.8171
    Epoch 26/40 | Time 178.7s | Train loss 0.5622 acc 0.9375 | Val loss 0.7308 acc 0.8171
    Epoch 27/40 | Time 179.2s | Train loss 0.5572 acc 0.9391 | Val loss 0.7222 acc 0.8257
    Epoch 28/40 | Time 179.0s | Train loss 0.5487 acc 0.9355 | Val loss 0.7099 acc 0.8499
    Epoch 29/40 | Time 181.3s | Train loss 0.5232 acc 0.9433 | Val loss 0.7082 acc 0.8249
    Epoch 30/40 | Time 178.2s | Train loss 0.5277 acc 0.9339 | Val loss 0.7014 acc 0.8570
    Epoch 31/40 | Time 178.2s | Train loss 0.5151 acc 0.9479 | Val loss 0.6913 acc 0.8499
    Epoch 32/40 | Time 177.7s | Train loss 0.5238 acc 0.9292 | Val loss 0.6890 acc 0.8656
    Epoch 33/40 | Time 177.9s | Train loss 0.5100 acc 0.9433 | Val loss 0.6848 acc 0.8734
    Epoch 34/40 | Time 179.3s | Train loss 0.5013 acc 0.9433 | Val loss 0.6827 acc 0.8499
    Epoch 35/40 | Time 178.9s | Train loss 0.4992 acc 0.9443 | Val loss 0.6777 acc 0.8499
    Epoch 36/40 | Time 177.9s | Train loss 0.5014 acc 0.9443 | Val loss 0.6773 acc 0.8578
    Epoch 37/40 | Time 178.1s | Train loss 0.4976 acc 0.9485 | Val loss 0.6739 acc 0.8578
    Epoch 38/40 | Time 179.4s | Train loss 0.4823 acc 0.9443 | Val loss 0.6728 acc 0.8656
    Epoch 39/40 | Time 179.9s | Train loss 0.5031 acc 0.9386 | Val loss 0.6733 acc 0.8656
    Epoch 40/40 | Time 179.4s | Train loss 0.4933 acc 0.9547 | Val loss 0.6689 acc 0.8578
    Training finished. Best val acc: 0.8733836263418198
    ```

- [x] 针对红外图像允许ResNet18微调
    ```bash
    python -u Resnet18/src/train.py --resume /root/pattern-recognition-course-assignments/checkpoints_batchsize32_and_no_reg/infrared_finetune/ckpt_epoch40.pth --epochs 60 --modalities infrared --ckpt_dir checkpoints_batchsize32_and_no_reg/infrared_finetune/train_full_and_resnet18_finetune --train_full --frames_per_clip 32 >> log.txt
    ```
    结果：
    ```text
    Using device: cuda
    Loaded checkpoint from epoch 40
    Warning: Optimizer state has different number of parameter groups (likely due to unfreezing backbone). Skipping optimizer state loading.
    Epoch 41/60 | Time 191.6s | Train loss 0.3305 acc 0.9723
    Epoch 42/60 | Time 190.9s | Train loss 0.2168 acc 0.9930
    Epoch 43/60 | Time 183.7s | Train loss 0.1352 acc 0.9980
    Epoch 44/60 | Time 183.7s | Train loss 0.0964 acc 0.9969
    Epoch 45/60 | Time 186.3s | Train loss 0.0764 acc 0.9980
    Epoch 46/60 | Time 188.3s | Train loss 0.0635 acc 0.9980
    Epoch 47/60 | Time 186.7s | Train loss 0.0573 acc 1.0000
    Epoch 48/60 | Time 187.4s | Train loss 0.0441 acc 0.9980
    Epoch 49/60 | Time 187.7s | Train loss 0.0356 acc 1.0000
    Epoch 50/60 | Time 186.2s | Train loss 0.0331 acc 1.0000
    Epoch 51/60 | Time 185.9s | Train loss 0.0308 acc 1.0000
    Epoch 52/60 | Time 186.8s | Train loss 0.0266 acc 1.0000
    Epoch 53/60 | Time 183.1s | Train loss 0.0260 acc 1.0000
    Epoch 54/60 | Time 190.1s | Train loss 0.0230 acc 1.0000
    Epoch 55/60 | Time 190.9s | Train loss 0.0211 acc 1.0000
    Epoch 56/60 | Time 188.4s | Train loss 0.0198 acc 1.0000
    Epoch 57/60 | Time 189.1s | Train loss 0.0188 acc 1.0000
    Epoch 58/60 | Time 188.7s | Train loss 0.0189 acc 1.0000
    Epoch 59/60 | Time 186.6s | Train loss 0.0180 acc 1.0000
    Epoch 60/60 | Time 186.8s | Train loss 0.0165 acc 1.0000
    Resume training finished. Best train loss: 0.01647493604104966
    ```

- [x] 针对RGB图像继续微调ResNet18，直至Epoch 60
    ```bash
    python -u Resnet18/src/train.py --resume /root/pattern-recognition-course-assignments/checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full/best.pth --epochs 60 --train_full --modalities rgb --ckpt_dir checkpoints_batchsize32_and_no_reg/resnet18_finetune/train_full --frames_per_clip 32 >> log.txt
    ```
    结果：
    ```text
    Using device: cuda
    Loaded checkpoint from epoch 40
    Optimizer state loaded successfully.
    Epoch 41/60 | Time 190.2s | Train loss 0.0801 acc 1.0000
    Epoch 42/60 | Time 185.8s | Train loss 0.0772 acc 0.9969
    Epoch 43/60 | Time 184.5s | Train loss 0.0720 acc 1.0000
    Epoch 44/60 | Time 185.0s | Train loss 0.0731 acc 1.0000
    Epoch 45/60 | Time 190.1s | Train loss 0.0667 acc 1.0000
    Epoch 46/60 | Time 189.2s | Train loss 0.0704 acc 1.0000
    Epoch 47/60 | Time 191.4s | Train loss 0.0718 acc 1.0000
    Epoch 48/60 | Time 188.7s | Train loss 0.0656 acc 1.0000
    Epoch 49/60 | Time 187.0s | Train loss 0.0661 acc 1.0000
    Epoch 50/60 | Time 190.4s | Train loss 0.0676 acc 1.0000
    Epoch 51/60 | Time 187.4s | Train loss 0.0654 acc 1.0000
    Epoch 52/60 | Time 186.9s | Train loss 0.0610 acc 1.0000
    Epoch 53/60 | Time 184.7s | Train loss 0.0615 acc 1.0000
    Epoch 54/60 | Time 183.7s | Train loss 0.0633 acc 1.0000
    Epoch 55/60 | Time 184.5s | Train loss 0.0605 acc 1.0000
    Epoch 56/60 | Time 184.3s | Train loss 0.0607 acc 1.0000
    Epoch 57/60 | Time 184.9s | Train loss 0.0641 acc 1.0000
    Epoch 58/60 | Time 183.3s | Train loss 0.0601 acc 1.0000
    Epoch 59/60 | Time 183.9s | Train loss 0.0627 acc 1.0000
    Epoch 60/60 | Time 182.8s | Train loss 0.0575 acc 1.0000
    Resume training finished. Best train loss: 0.057495471090078354
    ```

    
- - -