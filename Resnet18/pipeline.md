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

- [ ] 支线：验证此模型是否过拟合；采用全量微调后在测试集上再次尝试
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
- - -