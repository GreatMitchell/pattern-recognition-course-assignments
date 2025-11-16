# used for data loading
from utils.data_loader import DataLoader

# used for feature extraction
import features.extractors as extractors

# used for model building
from models.svm import SVMModel

# used for late fusion
from models.late_fusion import LateFusion

# other imports
import numpy as np

class TrainingPipeline:
    """
    采用基于准确率的加权概率晚期融合，完成SVM预测模型训练的全套流水线。
    """

    def __init__(self, aggregation_method='mean', n_classes=20):
        """
        Pipeline初始化。

        Parameters:
        ----------
        aggregation_method (str): 帧特征聚合方法，支持 'mean'、 'max'、 'stat_concat'。
        n_classes (int): 分类任务中的类别数量。
        """
        self.data_loader = DataLoader()
        self.feature_extractor = extractors.FeatureExtractor(self.data_loader, aggregation_method=aggregation_method)
        self.svm_model_rgb = SVMModel()
        self.svm_model_depth = SVMModel()
        self.svm_model_infrared = SVMModel()
        self.late_fusion = LateFusion(n_classes=n_classes)

    def run_validation(self):
        # step1: extract features
        rgb_train_features, depth_train_features, infrared_train_features = self.feature_extractor.separate_fusion(set_type='train')
        rgb_val_features, depth_val_features, infrared_val_features = self.feature_extractor.separate_fusion(set_type='val')
        print("Feature extraction completed.")

        # step2: get labels
        train_labels = self.data_loader.get_train_labels()
        val_labels = self.data_loader.get_val_labels()
        print("Labels loaded.")

        # step3: train SVMs for each modality
        self.svm_model_rgb.train(rgb_train_features, train_labels)
        self.svm_model_depth.train(depth_train_features, train_labels)
        self.svm_model_infrared.train(infrared_train_features, train_labels)
        print("SVM training completed.")

        # step4: evaluate SVMs on validation set
        rgb_val_accuracy = self.svm_model_rgb.evaluate(rgb_val_features, val_labels)
        depth_val_accuracy = self.svm_model_depth.evaluate(depth_val_features, val_labels)
        infrared_val_accuracy = self.svm_model_infrared.evaluate(infrared_val_features, val_labels)
        print(f"Validation Accuracies - RGB: {rgb_val_accuracy:.4f}, Depth: {depth_val_accuracy:.4f}, Infrared: {infrared_val_accuracy:.4f}")

        # step5: get accuracies for weighting
        rgb_train_accuracy = self.svm_model_rgb.evaluate(rgb_train_features, train_labels)
        depth_train_accuracy = self.svm_model_depth.evaluate(depth_train_features, train_labels)
        infrared_train_accuracy = self.svm_model_infrared.evaluate(infrared_train_features, train_labels)

        # step6: get probabilities for late fusion
        rgb_val_probs = self.svm_model_rgb.probability(rgb_val_features)
        depth_val_probs = self.svm_model_depth.probability(depth_val_features)
        infrared_val_probs = self.svm_model_infrared.probability(infrared_val_features)

        # step7: accuracy-weighted probability late fusion
        probabilities_list = [rgb_val_probs, depth_val_probs, infrared_val_probs]
        accuracies = [rgb_train_accuracy, depth_train_accuracy, infrared_train_accuracy]
        fused_val_preds = self.late_fusion.weighted_probability(probabilities_list, accuracies)
        print("Late fusion completed.")

        # step8: evaluate fused predictions
        fused_accuracy = np.mean(fused_val_preds == val_labels)
        print(f"Fused Validation Accuracy: {fused_accuracy:.4f}")

    def run(self, save=True):
        """
        Full pipeline for training on all data and testing.

        Parameters:
        ----------
        save (bool): 是否保存训练好的模型。

        Returns:
        -------
        rgb_test_preds (np.ndarray): RGB模态的测试集预测结果。
        fused_test_preds (np.ndarray): 融合后的测试集预测结果。
        """
        # step1: extract features
        rgb_features, depth_features, infrared_features = self.feature_extractor.separate_fusion(set_type='all')
        print("Feature extraction completed.")

        # step2: get labels
        all_labels = self.data_loader.get_all_labels()
        print("Labels loaded.")

        # step3: train SVMs for each modality
        self.svm_model_rgb.train(rgb_features, all_labels)
        self.svm_model_depth.train(depth_features, all_labels)
        self.svm_model_infrared.train(infrared_features, all_labels)
        print("SVM training completed.")

        # step4: extract test features
        rgb_test_features, depth_test_features, infrared_test_features = self.feature_extractor.separate_fusion(set_type='test')
        rgb_test_preds = self.svm_model_rgb.predict(rgb_test_features)
        print("Test feature extraction completed.")

        # step5: get accuracies for weighting
        rgb_train_accuracy = self.svm_model_rgb.evaluate(rgb_features, all_labels)
        depth_train_accuracy = self.svm_model_depth.evaluate(depth_features, all_labels)
        infrared_train_accuracy = self.svm_model_infrared.evaluate(infrared_features, all_labels)

        # step6: get probabilities for late fusion
        rgb_test_probs = self.svm_model_rgb.probability(rgb_test_features)
        depth_test_probs = self.svm_model_depth.probability(depth_test_features)
        infrared_test_probs = self.svm_model_infrared.probability(infrared_test_features)

        # step7: accuracy-weighted probability late fusion
        probabilities_list = [rgb_test_probs, depth_test_probs, infrared_test_probs]
        accuracies = [rgb_train_accuracy, depth_train_accuracy, infrared_train_accuracy]
        fused_test_preds = self.late_fusion.weighted_probability(probabilities_list, accuracies)
        print("Late fusion on test set completed.")

        # step8: save models if required
        if save:
            self.svm_model_rgb.save_model('svm_model_rgb.joblib')
            self.svm_model_depth.save_model('svm_model_depth.joblib')
            self.svm_model_infrared.save_model('svm_model_infrared.joblib')
            print("Models saved.")

        return rgb_test_preds, fused_test_preds