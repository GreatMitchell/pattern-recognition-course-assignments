from utils.data_loader import DataLoader
from models.lstm import VideoRecognitionModel
from utils.device import try_gpu
import torch

device = try_gpu()
model = VideoRecognitionModel(num_classes=20, use_lstm=True).to(device)

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint  # 可能直接是 state_dict

    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式

def predict(modalities=['rgb']):
    # 如果没有指定模态，默认使用RGB
    if modalities is None or len(modalities) == 0:
        modalities = ['rgb']

    dl = DataLoader()
    test_loader = dl.load_multi_modal_dataiter(set='test', batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(test_loader):
        rgb = batch['rgb'].to(device) if 'rgb' in modalities else None
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