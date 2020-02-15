import numpy as np
import tensorflow as tf
import time 
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
		self.results[self.npoints[0]]['idx'] = idx_512
		# print('512',new_xyz_512.shape,idx_512.shape)
		new_xyz_128 = self.sampling(new_xyz_512, self.npoints[1])
		idx_128 = self.grouping(new_xyz_512, new_xyz_128, self.nsamples[1], self.radius[1])
		self.results[self.npoints[1]]['new_xyz'] = new_xyz_128
		self.results[self.npoints[1]]['idx'] = idx_128
		# print('128',new_xyz_128.shape,idx_128.shape)

		return None

	def sampling(self, xyz, npoint):
		batch_size = int(xyz.shape[0])
		ndataset = int(xyz.shape[1])
		new_xyz_list = []
		for i in range(batch_size):
			idx = np.arange(ndataset)
			sample_idx = np.random.choice(idx,npoint,replace=False).reshape(npoint)
			# batch_idx = np.repeat([i],npoint).astype(sample_idx.dtype)
			# gathernd_idx = tf.stack([batch_idx,sample_idx],axis=-1)
			new_xyz = np.take(xyz[i],sample_idx,axis=0)
			new_xyz_list.append(new_xyz)
		return np.stack(new_xyz_list,axis=0)

	def grouping(self, xyz, new_xyz, nsample, radius, K=32):
		xyz = xyz.astype(np.float32)
		new_xyz = new_xyz.astype(np.float32)
		batch_size = int(xyz.shape[0])
		npoint = int(new_xyz.shape[1])
		# print(type(xyz))
		# print(type(new_xyz))
		# print(xyz.shape)
		# print(new_xyz.shape)
		if self.knn:
			idx_list = []
			# config = tf.ConfigProto()
			# config.gpu_options.allow_growth = True
			# config.allow_soft_placement = True
			# config.log_device_placement = False
			# sess = tf.Session(config=config)
			# ngpus = faiss.get_num_gpus()
			# print("number of GPUs:", ngpus)
			
			for i in range(batch_size):
				# time1 = time.time()
				cpu_index = faiss.IndexFlatL2(3) # make 3 dim index
				# time2 = time.time()
				# gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
				#     cpu_index
				# )
				# print('batch_idx: ',i)
				# time3 = time.time()
				reference = xyz[i]#tf.cast(xyz[i],tf.float32).eval(session=sess)
				# time4 = time.time()
				cpu_index.add(reference)
				# time5 = time.time()
				query = new_xyz[i]#tf.cast(new_xyz[i],tf.float32).eval(session=sess)
				# time6 = time.time()
				I,_ = cpu_index.search(query,K) # returns index and distance, I.shape = (npoint,K)
				# time7 = time.time()
				idx_list.append(I)
				# print('time: ',time2-time1,time3-time2,time4-time3,time5-time4,time6-time5,time7-time6)
			return np.stack(idx_list,axis=0).astype(np.int32)

		else:
			return np.tile(
						np.arange(nsample).reshape((1,1,nsample)),
						[batch_size,npoint,1]
					).astype(np.int32)
			# tf.cast(
			# 		tf.tile(
			# 			tf.constant(
			# 				np.arange(nsample).reshape((1,1,nsample))
			# 				),
			# 			[batch_size,npoint,1]
			# 			),
			# 		tf.int32
			# 		)



# def preprocess_grouping_and_sampling(batch_point_clouds):
	