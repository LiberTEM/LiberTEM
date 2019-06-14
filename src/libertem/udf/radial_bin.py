import collections
import functools

import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.masks import _make_circular_mask, radial_bins

def make_result_buffers():
	return {
		'intensity': BufferWrapper(
			kind='sig', dtype='float32')
	}


def compute_batch(frame, radial_bin, intensity):
	binned_frame = frame * radial_bin
	intensity[:] = binned_frame


def run_radial_bin(ctx, dataset,center_x, center_y):
	"""
	"""
	job = ctx.create_mask_job(
		factories=lambda: radial_bins(centerX=center_x, 
							centerY=center_y, 
							imageSizeX=dataset.shape.sig[1],
							imageSizeY=dataset.shape.sig[0]
							),
		dataset=dataset
		)

	results = ctx.run(job)

	# results = ctx.run_udf(
	# 	dataset=dataset,
	# 	make_buffers=make_result_buffers,
	# 	init=functools.partial(
	# 		radial_bins,
	# 		centerX=center_x,
	# 		centerY=center_y,
	# 		imageSizeX=dataset.shape.sig[0],
	# 		imageSizeY=dataset.shape.sig[1]
	# 		),
	# 	fn=compute_batch
	# 	)

	return results