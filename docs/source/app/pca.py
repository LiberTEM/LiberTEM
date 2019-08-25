import numpy as np 

from libertem import api
from libertem.udf.pca import run_pca


def RunPca(lt_ctx, path_to_dataset, n_component):
	"""
	Run Principal Component Analysis on the provided dataset
	using n_component number of components. This function returns
	the dataset with reduced dimension after PCA is performed.
	"""
	with api.Context() as ctx:
		path = path_to_dataset
		dataset = ctx.load(
		'empad',
		path=path,
		scan_size=(256, 256),
		)

	res = run_pca(lt_ctx, dataset, n_component)

	left_singular = res['left_singular'].data
	singular_vals = res['singular_vals'].data
	components = res['components'].data

	loading = left_singular @ np.diag(singular_vals)

	dataset = dataset.reshape((dataset.shape[0]*dataset.shape[1], dataset.shape[2]*dataset.shape[3]))
	projected_data = dataset @ components

	return projected_data
