import zipfile
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
	"""
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)  # Updates progress
		self.last_block = block_num


print('Downloading zip file ...')
with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
	urlretrieve('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip', 'data_road.zip', pbar.hook)

print('Extracting zip file...')
zip_ref = zipfile.ZipFile('data_road.zip', 'r')
zip_ref.extractall()
zip_ref.close()
