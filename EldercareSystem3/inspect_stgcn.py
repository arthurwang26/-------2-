import torch
ckpt = torch.load(r'C:\Users\arthu\Desktop\新增資料夾 (2)\EldercareSystem2\weights\stgcn_ntu60_xsub_coco17.pth', map_location='cpu', weights_only=True)
sd = ckpt['state_dict']
keys = list(sd.keys())
print(f'Num keys: {len(keys)}')
print('First 10:', keys[:10])
print('Last 5:', keys[-5:])
for k in keys[-5:]:
    print(f'  {k}: {sd[k].shape}')
print('\nMeta:', ckpt.get('meta', {}))
