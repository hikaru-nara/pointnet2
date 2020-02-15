import numpy as np
import tensorflow as tf
import faiss # facebook ai similarity search, for KNN and Ball query
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
		# self.num_gpus = config.num_gpus
		# print('batch_size',self.batch_size)
		# self.batch_size = config.batch_size//self.num_gpus # batch size per gpu
		self.npoints = [512,128]
		self.nsamples = [32,64]
		self.radius = [0.2,0.4]
		self.knn = config.knn
		self.K = [32,64] # KNN k, currently not used, should be adjustable with cmdline param
		self.results = {self.npoints[0]: {'new_xyz': tf.zeros((self.batch_size, self.npoints[0], 3)),
								'idx': tf.zeros((self.batch_size, self.npoints[0], self.nsamples[0]))},
					 	self.npoints[1]: {'new_xyz': tf.zeros((self.batch_size, self.npoints[1], 3)),
					 			'idx': tf.zeros((self.batch_size, self.npoints[1], self.nsamples[1]))}}
		# 2-step downsampling scale
		# in each scale the result contain a dict of 'new_xyz' and 'idx'
		# 'new_xyz': sampled points, TF tensor of shape (batch_size, npoint, 3)
		# 'idx': index in the previous layer of point cloud, 
		# 		each index is a point in group with a sampled point, shape (batch_size, npoint, nsample)

	def multi_gpu_init(self):
		self.batch_size = self.batch_size//self.config.num_gpus
		self.results = {self.npoints[0]: {'new_xyz': tf.zeros((self.batch_size, self.npoints[0], 3)),
								'idx': tf.zeros((self.batch_size, self.npoints[0], self.nsamples[0]))},
					 	self.npoints[1]: {'new_xyz': tf.zeros((self.batch_size, self.npoints[1], 3)),
					 			'idx': tf.zeros((self.batch_size, self.npoints[1], self.nsamples[1]))}}

	def batch_preprocess_grouping_and_sampling(self, batch_point_clouds, knn=False):
		print('-------------preprocessing---------------')
		new_xyz_512 = self.sampling(batch_point_clouds, self.npoints[0])
		idx_512 = self.grouping(batch_point_clouds, new_xyz_512, self.nsamples[0], self.radius[0])
		self.results[self.npoints[0]]['new_xyz'] = new_xyz_512
		self.results['512']['idx'] = idx_512
		print('512',new_xyz_512.shape,idx_512.shape)
		new_xyz_128 = self.sampling(new_xyz_512, self.npoints[1])
		idx_128 = self.grouping(new_xyz_512, new_xyz_128, self.nsamples[1], self.radius[1])
		self.results['128']['new_xyz'] = new_xyz_128
		self.results['128']['idx'] = idx_128
		# print('128',new_xyz_128.shape,idx_128.shape)

		return None

	def sampling(self, xyz, npoint):
		batch_size = xyz.shape[0]
		ndataset = xyz.shape[1]
		new_xyz_list = []
		for i in range(batch_size):
			idx = np.arange(ndataset)
			sample_idx = np.random.choice(idx,npoint,replace=False).reshape(npoint,1)
			# batch_idx = np.repeat([i],npoint).astype(sample_idx.dtype)
			# gathernd_idx = tf.stack([batch_idx,sample_idx],axis=-1)
			new_xyz = tf.gather_nd(xyz[i],sample_idx)
			new_xyz_list.append(new_xyz)
		return tf.stack(new_xyz_list,axis=0)

	def grouping(self, xyz, new_xyz, nsample, radius, K=32):
		batch_size = xyz.shape[0]
		npoint = int(new_xyz.shape[1])
		if self.knn:
			idx_list = []
			for i in range(batch_size):
				index = faiss.IndexFlatL2(3) # make 3 dim index
				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				config.allow_soft_placement = True
				config.log_device_placement = False
				sess = tf.Session(config=config)
				index.add(tf.cast(xyz[i],tf.float32).eval(session=sess))
				I,D = index.search(tf.cast(new_xyz[i],tf.float32).eval(session=sess), K) # returns index and distance, I.shape = 
				idx_list.append(I)
			return tf.cast(
						tf.stack(idx_list,axis=0), 
						tf.int32
					)

		else:
			return tf.cast(
					tf.tile(
						tf.constant(
							np.arange(nsample).reshape((1,1,nsample))
							),
						[batch_size,npoint,1]
						),
					tf.int32
					)



# def preprocess_grouping_and_sampling(batch_point_clouds):
	