from utils.data_loader import DataLoader
from models.lstm import VideoRecognitionModel
from utils.device import try_gpu
import torch

device = try_gpu()
dl = DataLoader()
test_loader = dl.load_multi_modal_dataiter(set='test', batch_size=1, shuffle=False, num_workers=0)

def load_single_modality_model(checkpoint_path, modality):
    """
    加载单模态模型
    
    参数:
    checkpoint_path (str): 模型检查点的路径。
    modality (str): 模态类型 ('rgb' 或 'infrared')。
    
    返回:
    model: 加载的模型。
    """
    model = VideoRecognitionModel(num_classes=20, modalities=[modality], use_lstm=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint  # 可能直接是 state_dict

    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式
    return model

def load_multi_modality_model(checkpoint_path, modalities, learn_weights=False, pretrained_weights_paths=None, use_attention=False):
    """
    加载多模态模型（中期融合）
    
    参数:
    checkpoint_path (str): 模型检查点的路径。
    modalities (list of str): 要使用的模态列表，如 ['rgb', 'infrared']。
    pretrained_weights_paths (dict): 各模态的预训练权重路径字典。
    
    返回:
    model: 加载的模型。
    """
    model = VideoRecognitionModel(num_classes=20, modalities=modalities, use_lstm=True, learn_weights=learn_weights, pretrained_weights_paths=pretrained_weights_paths, use_attention=use_attention).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint  # 可能直接是 state_dict

    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式
    return model

def mid_fusion_predict(checkpoint_path, modalities=['rgb', 'infrared'], learn_weights=False, pretrained_weights_paths=None, use_attention=False):
    """
    使用中期融合策略进行预测（特征级别拼接模态）。
    
    参数:
    checkpoint_path (str): 多模态模型检查点的路径。
    modalities (list of str): 要使用的模态列表。支持 'rgb', 'depth', 'infrared'。
    pretrained_weights_paths (dict): 各模态的预训练权重路径字典，如 {'rgb': 'path/to/rgb.pth', 'infrared': 'path/to/ir.pth'}。
    
    返回:
    无，直接将结果写入 submission.csv。
    """
    if modalities is None or len(modalities) == 0:
        raise ValueError("Modalities must be specified for mid-fusion prediction.")
    
    # 验证模态
    supported_modalities = {'rgb', 'depth', 'infrared'}
    for mod in modalities:
        if mod not in supported_modalities:
            raise ValueError(f"Unsupported modality: {mod}. Supported modalities are: {supported_modalities}")
    
    # 加载多模态模型
    model = load_multi_modality_model(checkpoint_path, modalities, learn_weights, pretrained_weights_paths, use_attention)
    
    # 清空submission.csv文件
    with open('submission.csv', 'w') as f:
        f.write("sample_id,prediction\n")  # 写入表头

    for i, batch in enumerate(test_loader):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device) if 'depth' in modalities else None
        infrared = batch['infrared'].to(device) if 'infrared' in modalities else None
        lengths = batch['lengths']
        sample_id = batch['ids'][0]
        
        with torch.no_grad():
            logits = model(rgb, depth, infrared, lengths=lengths)
            pred = logits.argmax(dim=1).item()
        
        # 保存 (sample_id, pred) 到结果文件
        with open('submission.csv', 'a') as f:
            f.write(f"{sample_id},{pred}\n")

def predict(checkpoint_path, modalities=['rgb']):
    """
    使用指定的模型检查点和模态对测试数据进行预测。

    参数:
    checkpoint_path (str): 模型检查点的路径。
    modalities (list of str): 要使用的模态列表。支持 'rgb', 'depth', 'infrared'。

    返回:
    无，直接将结果写入 submission.csv。
    """
    if modalities is None or len(modalities) == 0:
        # 如果没有指定模态，默认使用RGB
        modalities = ['rgb']
    else:
        # 指定的模态必须在支持的模态中
        supported_modalities = {'rgb', 'depth', 'infrared'}
        for mod in modalities:
            if mod not in supported_modalities:
                raise ValueError(f"Unsupported modality: {mod}. Supported modalities are: {supported_modalities}")
            
    model = load_single_modality_model(checkpoint_path, modalities)

    for i, batch in enumerate(test_loader):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device) if 'depth' in modalities else None
        infrared = batch['infrared'].to(device) if 'infrared' in modalities else None
        lengths = batch['lengths']
        sample_id = batch['ids'][0]
        logits = model(rgb, depth, infrared, lengths=lengths)
        with torch.no_grad():
            pred = logits.argmax(dim=1).item()
        
        # 保存 (sample_id, pred) 到结果文件
        with open('submission.csv', 'a') as f:
            f.write(f"{sample_id},{pred}\n")

def late_fusion_predict(rgb_checkpoint, ir_checkpoint, weights=[0.5, 0.5]):
    """
    使用RGB和红外模态进行晚期融合预测，通过加权平均预测向量。

    参数:
    rgb_checkpoint (str): RGB模态模型检查点的路径。
    ir_checkpoint (str): 红外模态模型检查点的路径。
    weights (list of float): RGB和红外模态的权重，默认为 [0.5, 0.5]。

    返回:
    无，直接将结果写入 submission.csv。
    """
    if len(weights) != 2:
        raise ValueError("Weights must be a list of two floats.")
    
    # 加载单模态模型
    rgb_model = load_single_modality_model(rgb_checkpoint, 'rgb')
    ir_model = load_single_modality_model(ir_checkpoint, 'infrared')
    
    # 清空submission.csv文件
    # with open('submission.csv', 'w') as f:
    #     f.write("sample_id,prediction\n")  # 写入表头

    for i, batch in enumerate(test_loader):
        rgb = batch['rgb'].to(device)
        infrared = batch['infrared'].to(device)
        lengths = batch['lengths']
        sample_id = batch['ids'][0]
        
        with torch.no_grad():
            # RGB模态预测
            rgb_logits = rgb_model(rgb, None, None, lengths=lengths)
            # 红外模态预测
            ir_logits = ir_model(rgb, None, infrared, lengths=lengths)
            # 晚期融合：加权平均
            final_logits = weights[0] * rgb_logits + weights[1] * ir_logits
            pred = final_logits.argmax(dim=1).item()
        
        # 保存 (sample_id, pred) 到结果文件
        with open('submission.csv', 'a') as f:
            f.write(f"{sample_id},{pred}\n")