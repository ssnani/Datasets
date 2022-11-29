import torch
from torch.utils.data import Dataset, DataLoader


class RandomGPUDataset(Dataset):
	def __init__(self) -> None:
		super().__init__()

	def __getitem__(self, idx):
		x = torch.rand((100,300,300)).to(device='cuda:0')
		return x
		
	def __len__(self):
		return 100


if __name__=="__main__":
	train_dataset = RandomGPUDataset()
	train_loader = DataLoader(train_dataset, batch_size = 1, num_workers=1)
	for _batch_idx, val in enumerate(train_loader):
		print(f"doa: {val[0].shape}, {val[0].dtype}, {val[0].device} \n")

