from random import shuffle as randomshuffle
from pathlib import Path
from typing import Optional, Callable, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


class Hdf5Dataset(Dataset):
	
	def __init__(
		self,
		path: Union[Path,str],
		shuffle: bool = True,
		transform: Optional[Callable] = None,
		event_limit: Optional[int] = None
	):
		r"""
		Args:
			path (Path or str): Path to directory containing HDF5 files, with .h5 suffix 
			shuffle (bool): Randomize order of events. Always enable for training, 
				otherwise all batches will contain a single class only. Optionally enable
				shuffling in DataLoader
			transform (Callable, optional): Function for data transforms
			event_limit (int, optional): Limit number of events in dataset 
		"""
		super().__init__()
		
		self._transform = transform		

		# Get file list
		if not isinstance(path, Path):
			path = Path(path)
		
		filenames = path.glob('*.h5')
		
		# Open file descriptors and get event keys
		# Store file name along with key, to keep track
		self._files = {}
		self._event_keys = []
		for full_file_path in filenames:
			fd = h5py.File(full_file_path, 'r')
			filename = full_file_path.name.__str__()
			self._files[filename] = fd
			event_keys = list(fd.keys())
			event_keys = [(filename, key) for key in event_keys]
			self._event_keys += event_keys
		
		assert len(self._files) > 0, f'No files found in {path}'
		
		# Shuffle keys
		randomshuffle(self._event_keys)
		
		# Limit number of events
		if event_limit:
			self._event_keys = self._event_keys[:event_limit]


	def __len__(self):
		return len(self._event_keys)

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		filename, key = self._event_keys[idx]
		group = self._files[filename].get(key)
		data = group.get('data')[()]
		label = group.get('label')[()].decode()
		
		if self._transform is not None:
			data = self._transform(data)

		# GTX cards are single precision only
		data = data.astype(np.float32)

		return (data, label)



if __name__ == '__main__':

	trainpath = '/disk/atlas3/data_MC/2dhistograms/sph_BH/training/100'
	dataset = Hdf5Dataset(trainpath, event_limit=16)
	
	dataloader = DataLoader(dataset, batch_size=8)

	for data_batch, label_batch in dataloader:

		print('data_batch.shape:', data_batch.shape)
		
		event0, label0 = data_batch[0], label_batch[0]
		print('event0.shape:', event0.shape)
		print('event0.label:', label0)




	

