import torch
import os
from models.efficient_attention import EfficientAttnSASRecModel
from tqdm import tqdm

BATCH_SIZE = 1
SEQLEN = 65536
EMB_DIM = 1024
attn = EfficientAttnSASRecModel(
    n_items=1000,
    emb_dim=EMB_DIM,
    num_codebooks=8,
    num_codewords=16
)

os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

src = torch.randint(low=0, high=100, size=(BATCH_SIZE, SEQLEN), dtype=torch.long).to(device)

attn.to(device)
attn.eval()
attn.manual_ip_table_update()
trails = 100
total_time = 0.
attn(src)
attn(src)
for i in tqdm(range(trails)):
    with torch.no_grad():
        total_time += attn(src)
print("time for efficient attention: {:.2f}ms".format(total_time / trails))