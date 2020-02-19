import numpy as np
import tensorflow as tf
import time 
import copy
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
		self.paging = config.paging
		if self.paging:
			self.page_size = config.page_size 
			'''
			absolute page sidelength is proportional to the sidelength of the pointcloud, i.e.
			self.page_size * pointcloud_sidelength = absolute_page_sidelength
			the space will be divided into (1/self.page_size)^3 grids with side_length = absolute_page_sidelength
			pointcloud_sidelength is computed on specific instance of each point cloud
			tune this hyperparameter to get appropriate number of non-vacant pages
			'''
			self.page_lists = [] # batch_size number of page_lists
			self.index_mappings = [] # batch_size number of mappings
			self.batch_zero_point = []
			self.batch_page_side_length = []
			
		self.npoints = [256,64]
		self.nsamples = [32,64]
		self.radius = [0.2,0.4]

		self.knn = config.knn
		self.K = [4,4] # KNN k, currently not used, should be adjustable with cmdline param
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
		if self.paging:
			self.page_init(batch_point_clouds)
			self.page_preprocess(batch_point_clouds)
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
			# print('points num',ndataset,npoint)
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
		nchannel = int(xyz.shape[-1])
		if self.knn:
			idx_list = []
			for i in range(batch_size):
				# time1 = time.time()
				cpu_index = faiss.IndexFlatL2(nchannel) # make 3 dim index
				# time2 = time.time()
				# time3 = time.time()
				reference = xyz[i]
				# print(xyz.shape)
				# time4 = time.time()
				cpu_index.add(reference)
				# time5 = time.time()
				query = new_xyz[i]
				# time6 = time.time()
				_,I = cpu_index.search(query,nsample) # returns index and distance, I.shape = (npoint,nsample)
				# time7 = time.time()
				idx_list.append(I)
				# print('time: ',time2-time1,time3-time2,time4-time3,time5-time4,time6-time5,time7-time6)
			return np.stack(idx_list,axis=0).astype(np.int32)

		else:
			raise NotImplementedError

	def page_init(self, point_clouds):
		'''
		To initialize the page list and the mapping from coordination to corner and fron corner to page list index 

		'''
		'''
		initialize the page_lists as empty page lists
		for each pointcloud in the batch:
			compute 3D side length of (the cube that exactly fits) each point cloud
			make a list of pages and a list of corresponding rear-upper-left corners
			establish a map from a page's rear-upper-left corner to its index in the page list(use dict temporarily)
			for all points in the pointcloud:
				round it to a page's rear-upper-left corner
				map this coordination to an index in the page list
				append the point to the page_list[index]
			discard the vacant pages in the page list
			discard the keys corresponding to a discarded page
		'''
		'''
		alternatively. potentially better:
		initialize the page_lists as empty page lists
		for each pointcloud in the batch:
			compute 3D side length of (the cube that exactly fits) each point cloud
			initialize a empty page list(literally containing no pages), and an empty corner-to-index mapping
			for each point in the point cloud:
				round it to a page's rear-upper-left corner
				if the corner is in the mapping, meaning that the page has been included into the page list
					get the index from the mapping
					append the point
				else:
					insert a new vacant page into the page list
					insert a new corner-to-index correspondence into the mapping
					append the point
		Another thing should be considered is the maximum number of points in each page
		'''
		# for point_cloud in point_clouds:
			# remember to print out the range of point clouds
		lower_limit = np.min(point_clouds, axis=1, keepdims=True)
		upper_limit = np.max(point_clouds, axis=1, keepdims=True)
		self.batch_zero_point = lower_limit
		self.batch_page_side_length = (upper_limit-lower_limit)*self.page_size

		batch_size = point_clouds.shape[0]
		for pointcloud in point_clouds:
			index_mapping = dict()
			page_list = []
			for point in pointcloud:
				corner = self.rounding(point)
				if corner is in index_mapping.keys():
					page_list[index_mapping[corner]].append(point)
					# page_list[index_mapping[corner]] = np.concatenate([page_list[index_mapping[corner]],[point]], axis=0)
				else:
					index_mapping[corner]=len(page_list)
					page_list.append([point])
			page_list_np = [np.array(page) for page in page_list]
			self.page_lists.append(page_list_np)



		
	def page_preprocess(self, batch_point_clouds):

		ndataset = batch_point_clouds.shape[1]
		stages = [ndataset]+self.npoints
		downsample_rate = []
		for i in range(len(self.npoints)):
			downsample_rate.append(stages[i]/stages[i+1])
		page_lists = self.page_lists
		xyz = batch_point_clouds
		for i in range(len(self.npoints)):
			# interpolation unconsidered currently
			if downsample_rate[i]<1:
				self.page_interpolation()
			new_xyz, new_page_lists = self.page_sampling(page_lists, downsample_rate[i])
			idx = self.page_grouping(page_lists, xyz, new_xyz, self.nsamples[i], self.radius[i])
			self.results[self.npoints[i]]['new_xyz'] = new_xyz
			self.results[self.npoints[i]]['idx'] = idx
			page_lists = copy.copy(new_page_lists)
			xyz = copy.copy(new_xyz) # is this deep_copy?

	def page_sampling(self, page_lists, downsample_rate, nsample):
		new_page_lists = []
		new_xyz_list = []
		for page_list in page_lists:
			new_page_list = []
			# new_xyz_list = []
			for page in page_list: 
			# keep the order in the original list is important because it keep "index_mapping" valid through the sampling process
			# or remake a index-mapping, or use a index-mapping invariant of list, such as hierarchical
				assert type(page)==np.ndarray
				npoint = page.shape[0]
				
				# Because npoint%downsample_rate might not be 0, there may not be exactly $nsample$ points in the sampled point cloud
				# this step is to ensure that we sample more points than necessary so that we can clip afterwards.
					# Another way of implementation is to sample nsample/npages points in each page, with replacement enabled
					# Potentially the former is better because dense pages contain dense information(which is not always true) and we don't want to lose it. 
					# it's not really clear which is better. Consult Bichen and Chenfeng.
						# Edit, if nsample < npages this approach is ill-defined
					# Another thought flashed into my mind is to merge sparse pages (with a hierarchical structure like octree) as the sampling proceeds. 
					# If a page only contains one point, it's better we can merge it with a neighboring page, saving computation.
					# Merging can also avoid always keeping an initial page(unless clipped off), which is the property of current implementation
					# But the initial pages could be beneficial if it's contributing coverage of the original point cloud? Possibly not?
					# Finally the original point cloud will be merged into one single point, what's the meaning of coverage in the intermediate process?
					# Is it OK to lose some pages as the sampling proceeds?
						# Yes, Sure. But it had better be the pages containing non-important features. Or ensure that this page is included in another page's receptive field.
				if npoint%downsample_rate!=0: 
					padding_page = np.concatenate([page]+[page[0]]*(downsample_rate-(npoint%downsample_rate)), axis=0)
					npoint = padding_page.shape[0]
				else:
					padding_page = page
				nsample = npoint//downsample_rate 

				# if nsample==0: # maybe build a hierarchical structure as the sampling makes pointcloud sparser?
				# 	continue
				idx = np.arange(npoint)
				print('page.shape')
				print(padding_page.shape)
				print('downsample rate', downsample_rate)
				sample_idx = np.random.choice(idx,nsample,replace=False).reshape(nsample) # enable 
				page_new_xyz = np.take(padding_page,sample_idx,axis=0)
				new_page_list.append(page_new_xyz)
				# new_xyz_list
			new_xyz = np.concatenate(new_page_list,axis=0)
			print('new_xyz.shape')
			print(new_xyz.shape)
			new_page_lists.append(new_page_list)
		batch_new_xyz = np.stack(new_xyz_list,axis=0)
		print('batch_new_xyz.shape')
		print(batch_new_xyz.shape)

		return batch_new_xyz, new_page_lists

	def search_from_matrix(xq, xb, thresh):
		# reference https://gist.github.com/mdouze/5e561023d98187b7ee69a76da0fa36e2
	    nq, d = xq.shape
	    nb, d2 = xb.shape
	    assert d == d2
	    res = faiss.RangeSearchResult(nq)
	    faiss.range_search_L2sqr(
	        faiss.swig_ptr(xq), 
	        faiss.swig_ptr(xb), 
	        d, nq, nb, 
	        thresh, res)
	    
	    # get pointers and copy them
	    lims = faiss.rev_swig_ptr(res.lims, nq + 1).copy()
	    nd = int(lims[-1])
	    D = faiss.rev_swig_ptr(res.distances, nd).copy()
	    I = faiss.rev_swig_ptr(res.labels, nd).copy()
	    return lims, D, I

	def page_grouping(self, page_lists, xyz, new_xyz, nsample, radius, knn=True):
		# requires: self.batch_zero_point, self.batch_page_side_length
		# output: idx [batch_size,npoints,nsample]
		if knn:
			return self.grouping(xyz, new_xyz, nsample, radius, K=32)
		
		for i in range(batch_size):
			time1 = time.time()
			lim, D, I = search_from_matrix(new_xyz[i], xyz[i], radius)
			time2 = time.time()
			index = faiss.IndexFlatL2(xyz.shape[1])
			index.add(search_space)
			lim2, D2, I2 = index.range_search(new_xyz, radius)
			time3 = time.time()
		print('time_search_from_matrix', time2-time1)
		print('time_search_from_index', time3-time2)
		raise NotImplementedError

			'''
			deprecated!
			corner_offset = np.array(
			[[-radius, -radius, -radius],
			 [radius, -radius, -radius],
			 [radius, radius, -radius],
			 [radius, radius, radius],
			 [-radius, radius, radius],
			 [-radius, -radius, radius],
			 [radius, -radius, radius],
			 [-radius, radius, -radius]
			])
			for xyz in new_xyz[i]:
				# actually this may be slower than direct query because of the current implementation
				corners_xyz = np.tile(xyz.reshape(1,3),(8,1))+corner_offset
				search_space_indices = self.query(corners_xyz,self.index_mappings[i],self.batch_zero_point[i],self.batch_page_side_length[i])
				search_space_list = [page_lists[i][j] for j in search_space_indices]
				search_space = np.concatenate(search_space_list,axis=0)
				print('search space shape')
				print(search_space.shape)
				time1 = time.time()
				lim, D, I = search_from_matrix([xyz], search_space, radius)
				time2 = time.time()
				print('time_search_from_matrix', time2-time1)
				index = faiss.IndexFlatL2(search_space.shape[1])
				index.add(search_space)
				lim2, D2, I2 = index.range_search([xyz], radius)
				time3 = time.time()
				print('time_search_from_index', time3-time2)
			'''


	def query(self, xyzs, index_mapping, zero_point, page_side_length):
		'''
		query newpoints for the page it's in w.r.t the old mapping
		input: xyzs is an array of coordinations shape (?,3)
				zero_point (3), page_side_length (3), index_mapping is as the format of self.index_mappings[0]
		outputs: search_space_index in accordance with the order of xyzs (len(xyz))
		'''
		# batch_size = new_xyz.shape[0]
		# 
		zero_points = zero_point[np.newaxis,:]
		page_side_lengths = page_side_length[np.newaxis,:]
		translated_xyz = xyzs - zero_points
		rounded_new_xyz = translated_new_xyz - (translated_new_xyz%batch_page_side_length)
		page_corners_new_xyz = rounded_new_xyz + batch_zero_point
		search_space_indices = []
		# for i in range(batch_size):
		for xyz in page_corners_new_xyz[i]:
			if xyz is in index_mapping.keys()
				search_space_indices.append(index_mapping[xyz]) # I'm not sure whether this will be troubleshooting because of float computational precision
		return search_space_indices




	def page_interpolation(self):
		raise NotImplementedError



# def preprocess_grouping_and_sampling(batch_point_clouds):
	