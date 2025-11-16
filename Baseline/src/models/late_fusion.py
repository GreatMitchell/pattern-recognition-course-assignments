import numpy as np
from collections import Counter

class LateFusion:
    def __init__(self, n_classes=20):
        self.n_classes = n_classes

    def majority_vote(self, predictions_list):
        """
        绝对多数投票。

        Parameters:
        ----------
        predictions_list (list of ndarray): 每个模态的预测结果列表，即 [rgb_preds, depth_preds, infrared_preds]，其中每一个都是形状为 (n_samples,) 的预测标签ndarray。

        Returns:
        -------
        fused_predictions (ndarray): 融合后的预测结果，形状为 (n_samples,)。
        """
        n_samples = len(predictions_list[0])
        fused_predictions = []
        
        for i in range(n_samples):
            # 收集每个模态对当前样本的预测
            votes = [pred[i] for pred in predictions_list]
            # 统计票数
            vote_count = Counter(votes)
            # 选择得票最多的类别
            winner = vote_count.most_common(1)[0][0]
            fused_predictions.append(winner)
        
        return np.array(fused_predictions)
    
    def weighted_vote(self, predictions_list, accuracies):
        """
        基于准确率的加权投票。

        Parameters:
        ----------
        predictions_list (list of ndarray): 每个模态的预测结果列表，即 [rgb_preds, depth_preds, infrared_preds]，其中每一个都是形状为 (n_samples,) 的预测标签ndarray。
        accuracies (list) : 每个模态的SVM在训练集上的预测准确率构成的Python list，即[rgb_accuracy, depth_accuracy, infrared_accuracy]。

        Returns:
        -------
        fused_predictions (ndarray): 融合后的预测结果，形状为 (n_samples,)。
        """
        # 归一化准确率为权重
        weights = np.array(accuracies) / sum(accuracies)
        # print(f"归一化权重: RGB={weights[0]:.3f}, Depth={weights[1]:.3f}, Infrared={weights[2]:.3f}")
        
        n_samples = len(predictions_list[0])
        fused_predictions = []
        
        for i in range(n_samples):
            # 初始化每个类别的加权得分
            class_scores = np.zeros(self.n_classes)
            
            for modality_idx, pred in enumerate(predictions_list):
                predicted_class = pred[i]
                class_scores[int(predicted_class)] += weights[modality_idx]
            
            # 选择得分最高的类别
            winner = int(np.argmax(class_scores))
            fused_predictions.append(winner)
        
        return np.array(fused_predictions)
    
    def average_probability(self, probabilities_list):
        """
        平均概率融合。

        Parameters:
        ----------
        probabilities_list (list of ndarray): 每个模态的预测概率列表，即 [rgb_probs, depth_probs, infrared_probs]，其中每一个都是形状为 (n_samples, n_classes) 的预测概率ndarray。

        Returns:
        -------
        fused_predictions (ndarray): 融合后的预测结果，形状为 (n_samples,)。
        """
        n_samples = probabilities_list[0].shape[0]
        fused_predictions = []
        
        for i in range(n_samples):
            # 收集每个模态对当前样本的预测概率
            probs = [probs[i] for probs in probabilities_list]
            # 计算平均概率
            avg_probs = np.mean(probs, axis=0)
            # 选择概率最高的类别
            winner = int(np.argmax(avg_probs))
            fused_predictions.append(winner)
        
        return np.array(fused_predictions)
    
    def weighted_probability(self, probabilities_list, accuracies):
        """
        基于准确率的加权概率融合。

        Parameters:
        ----------
        probabilities_list (list of ndarray): 每个模态的预测概率列表，即 [rgb_probs, depth_probs, infrared_probs]，其中每一个都是形状为 (n_samples, n_classes) 的预测概率ndarray。
        accuracies (list) : 每个模态的SVM在训练集上的预测准确率构成的Python list，即[rgb_accuracy, depth_accuracy, infrared_accuracy]。

        Returns:
        -------
        fused_predictions (ndarray): 融合后的预测结果，形状为 (n_samples,)。
        """
        # 归一化准确率为权重
        weights = np.array(accuracies) / sum(accuracies)
        # print(f"归一化权重: RGB={weights[0]:.3f}, Depth={weights[1]:.3f}, Infrared={weights[2]:.3f}")
        
        n_samples = probabilities_list[0].shape[0]
        fused_predictions = []
        
        for i in range(n_samples):
            # 初始化每个类别的加权概率
            class_probs = np.zeros(probabilities_list[0].shape[1])
            
            for modality_idx, probs in enumerate(probabilities_list):
                class_probs += probs[i] * weights[modality_idx]
            
            # 选择加权概率最高的类别
            winner = int(np.argmax(class_probs))
            fused_predictions.append(winner)
        
        return np.array(fused_predictions)
    
    def prior_weighted_probability(self, probabilities_list, weights):
        """
        先验权重的加权概率融合。

        Parameters:
        ----------
        probabilities_list (list of ndarray): 每个模态的预测概率列表，即 [rgb_probs, depth_probs, infrared_probs]，其中每一个都是形状为 (n_samples, n_classes) 的预测概率ndarray。
        weights (list) : 每个模态的权重构成的Python list，即[rgb_weight, depth_weight, infrared_weight]。

        Returns:
        -------
        fused_predictions (ndarray): 融合后的预测结果，形状为 (n_samples,)。
        """
        # 归一化权重
        weights = np.array(weights) / sum(weights)

        n_samples = probabilities_list[0].shape[0]
        fused_predictions = []
        
        for i in range(n_samples):
            # 初始化每个类别的加权概率
            class_probs = np.zeros(probabilities_list[0].shape[1])
            
            for modality_idx, probs in enumerate(probabilities_list):
                class_probs += probs[i] * weights[modality_idx]
            
            # 选择加权概率最高的类别
            winner = int(np.argmax(class_probs))
            fused_predictions.append(winner)
        
        return np.array(fused_predictions)