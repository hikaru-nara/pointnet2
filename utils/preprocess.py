import numpy as np
import tensorflow as tf
"""
this file defines preprocessing function
input: batch point clouds
output: 
"""
batch_point_sampling = None

class Preprocessor(object):
	def __init__(self, config=None):
		super(Preprocessor, self).__init__()
		self.config = config
		self.batch_size = config.batch_size
		self.num_gpus = config.num_gpus
		print('batch_size',self.batch_size)
		# self.batch_size = config.batch_size//self.num_gpus # batch size per gpu
		self.npoints = [512,128]
		self.nsamples = [32,64]
		self.radius = [0.2,0.4]
		self.results = {'512': {'new_xyz': tf.zeros((self.batch_size, self.npoints[0], 3)),
								'idx': tf.zeros((self.batch_size, self.npoints[0], self.nsamples[0]))},
					 	'128': {'new_xyz': tf.zeros((self.batch_size, self.npoints[1], 3)),
					 			'idx': tf.zeros((self.batch_size, self.npoints[1], self.nsamples[1]))}}
		# 2-step downsampling scale
		# in each scale the result contain a dict of 'new_xyz' and 'idx'
		# 'new_xyz': sampled points, TF tensor of shape (batch_size, npoint, 3)
		# 'idx': index in the previous layer of point cloud, 
		# 		each index is a point in group with a sampled point, shape (batch_size, npoint, nsample)

	def batch_preprocess_grouping_and_sampling(self, batch_point_clouds):
		print('-------------preprocessing---------------')
		new_xyz_512 = self.sampling(batch_point_clouds, self.npoints[0])
		idx_512 = self.grouping(batch_point_clouds, new_xyz_512, self.nsamples[0], self.radius[0])
		self.results['512']['new_xyz'] = new_xyz_512
		self.results['512']['idx'] = idx_512
		print('512',new_xyz_512.shape,idx_512.shape)
		new_xyz_128 = self.sampling(new_xyz_512, self.npoints[1])
		idx_128 = self.grouping(new_xyz_512, new_xyz_128, self.nsamples[1], self.radius[1])
		self.results['128']['new_xyz'] = new_xyz_128
		self.results['128']['idx'] = idx_128
		print('128',new_xyz_128.shape,idx_128.shape)

		return None

	def sampling(self, xyz, npoint):
		return xyz[:, :npoint]

	def grouping(self, xyz, new_xyz, nsample, radius, knn=False):
		npoint = int(new_xyz.shape[1])
		return tf.cast(
				tf.tile(
					tf.constant(
						np.arange(nsample).reshape((1,1,nsample))
						),
					[self.batch_size,npoint,1]
					),
				tf.int32
				)



# def preprocess_grouping_and_sampling(batch_point_clouds):
	