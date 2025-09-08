import torch

data = torch.load("/nfs/ml_lab/projects/ml_lab/xlian/input_data/filtered_missing_tokenized.pt", weights_only=False)

# 看看这个文件里有什么
print(type(data))
if isinstance(data, dict):
    print(data.keys())   # 如果是字典，打印所有键
elif isinstance(data, list):
    print(f"List length: {len(data)}")
    print(type(data[0]) if len(data) > 0 else "empty list")

torch.save(data[:100] if isinstance(data, list) else dict(list(data.items())[:10]),
           "/nfs/ml_lab/projects/ml_lab/xlian/input_data/smi_prot_test.pt")