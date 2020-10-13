import multiprocessing
import numpy as np
from functools import partial 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import submodels_module as mb
import time
# import ray.util.multiprocessing.pool
from ray.util import ActorPool
import ray
import tensorflow as tf

def build_new_model():
	'transfers weight from old model to new model'
	# import tensorflow as tf
	s2a_params=[[1,8,10],'emb_cnn',1]
	s2a=mb.seq_to_assay_model(*s2a_params)
	s2a._model.set_model(s2a.get_best_trial()['hyperparam'],xa_len=16,cat_var_len=3,lin_or_sig=s2a.lin_or_sig)
	s2a.load_model(0)
	s2e_model=s2a._model.get_seq_embeding_layer_model()

	space=s2a.get_best_trial()['hyperparam']
	filters=int(space['filters'])
	kernel_size=int(space['kernel_size'])
	input_drop=space['input_drop']
	emb_dim=int(space['AA_emb_dim'])

	new_s2e=tf.keras.Sequential()	
	new_s2e.add(tf.keras.layers.Embedding(21,emb_dim,input_length=16))
	new_s2e.add(tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,activation='relu'))
	new_s2e.add(tf.keras.layers.GlobalMaxPool1D(name='seq_embedding'))
	new_s2e.build((None, 16))
	new_s2e.compile()
	new_s2e.set_weights(s2e_model.get_weights())
	new_s2e.save('best_emb_model')


@ray.remote
class getyield():
	def __init__(self,e2y):
		# import tensorflow as tf
		tf.config.optimizer.set_jit(True)
		self.s2e = tf.keras.models.load_model('best_emb_model')
		self.e2y=e2y
	def par_test(self,seqs):
		# seqs=ray.get(seqs_id)
		start=time.time()
		embeddings = self.s2e.predict(seqs)
		emb_iq = [np.concatenate([x, [1, 0]]) for x in embeddings]
		yield_iq = self.e2y.predict(emb_iq)
		emb_sh = [np.concatenate([x, [0, 1]]) for x in embeddings]
		yield_sh = self.e2y.predict(emb_sh)
		# print('compute time actor pool: %f'%(time.time()-start))
		return [yield_iq+yield_sh]

@ray.remote
class getyield_df():
	def __init__(self,e2y,seqs):
		# import tensorflow as tf
		tf.config.optimizer.set_jit(True)
		self.s2e = tf.keras.models.load_model('best_emb_model')
		self.e2y=e2y
		self.seqs=seqs

	def par_test(self):
		# start=time.time()
		embeddings = self.s2e.predict(self.seqs)
		emb_iq = [np.concatenate([x, [1, 0]]) for x in embeddings]
		yield_iq = self.e2y.predict(emb_iq)
		emb_sh = [np.concatenate([x, [0, 1]]) for x in embeddings]
		yield_sh = self.e2y.predict(emb_sh)
		# print('compute time ray normal: %f'%(time.time()-start))
		return [yield_iq+yield_sh]

def par_test_multiprocessing(e2y,seqs):
	'function to be called in parallel to get yield of seqs'
	start=time.time()
	import tensorflow as tf
	tf.config.optimizer.set_jit(True)
	s2e=tf.keras.models.load_model('best_emb_model')
	print('loading overhead: %f'%(time.time()-start))
	start=time.time()
	embeddings=s2e.predict(seqs)
	emb_iq=[np.concatenate([x,[1,0]]) for x in embeddings]
	yield_iq=e2y.predict(emb_iq)
	emb_sh=[np.concatenate([x,[0,1]]) for x in embeddings]
	yield_sh=e2y.predict(emb_sh)
	end=time.time()-start
	print('computing overall: %f'%end)
	return [yield_iq+yield_sh]

def par_test_serial(e2y,s2e,seqs):
	'function to be called in parallel to get yield of seqs'
	embeddings=s2e.predict(seqs)
	emb_iq=[np.concatenate([x,[1,0]]) for x in embeddings]
	yield_iq=e2y.predict(emb_iq)
	emb_sh=[np.concatenate([x,[0,1]]) for x in embeddings]
	yield_sh=e2y.predict(emb_sh)
	return [yield_iq+yield_sh]

def make_data(splits=8,n=10000):
	sample_seq_list = []
	for i in range(splits):
		inner_list = []
		for j in range(n):
			inner_list.append(np.random.randint(0, 20, 16))
		sample_seq_list.append(np.stack(inner_list))
	return sample_seq_list
def main():
	'main driver'
	#you still load the e2y model because it can be pickeled 
	s2a_params=[[1,8,10],'emb_cnn',1]
	e2y_params=['svm',1]
	e2y=mb.sequence_embeding_to_yield_model(s2a_params+[0],*e2y_params)
	e2y.load_model(0)
	e2y_model=e2y._model.model
	filled=partial(par_test_multiprocessing,e2y_model)

	# make the data

	sample_seq_list=make_data()

	# first serial
	# import tensorflow as tf
	# tf.config.optimizer.set_jit(True)
	# s2e = tf.keras.models.load_model('best_emb_model')
	# start=time.time()
	# a=par_test_serial(e2y_model,s2e,sample_seq_list[0])
	# end =time.time()-start
	# print('serial: %f'%end)

	# if ray.is_initialized() is True:
	# 	ray.shutdown()
	# ray.init(ignore_reinit_error=True)
	#
	# # init the pool
	# sample_seq_list = make_data(splits=50,n=10000)
	# init_pool=[getyield.remote(e2y_model) for _ in range(32)]
	# # # make an actor pool
	# pool = ActorPool(init_pool)
	# # # inputs in the the actor pool an anomyous function (like runge kunta in matlab)
	# # # a is the actor v is the input of the data
	# # sample_seq_list_id=ray.put(sample_seq_list)
	# res = pool.map_unordered(lambda actor, value: actor.par_test.remote(value), values=sample_seq_list)
	# print(list(res))
	# print('doing actor pool')
	# p=[]
	# for i in range(5):
	# 	start = time.time()
	# 	res = pool.map_unordered(lambda actor, value: actor.par_test.remote(value), values=sample_seq_list)
	# 	a=list(res)
	# 	p.append(time.time() - start)
	# 	print('actor pool: %i ,%f'%(i,p[-1]))
	# #
	#
	#
	# print('doing ray normal')
	# E=[]
	# ray.shutdown()
	# ray.init()
	# sample_seq_list = make_data(splits=32,n=8000)
	# inits = [getyield_df.remote(e2y_model, data) for data in sample_seq_list]
	# ray.get([actor.par_test.remote() for actor in inits]) # ray always runs super slow the first time so ignore.
	# for i in range(5):
	# 	start = time.time()
	# 	res = ray.get([actor.par_test.remote() for actor in inits])
	# 	E.append(time.time() - start)
	# 	print('normal ray: %f'%E[-1])
	# #evaluate yields in parallel
	# # make actor pool once, load the state of the two models.
	# # ray takes care of pointers in memory , so that we dont have to
	# ray.shutdown()

	m=[]
	print('multiprocessing')
	sample_seq_list=make_data(splits=50,n=10000)
	pool = multiprocessing.Pool(processes=32)
	# for _ in range(5):
	for i in range(5):
		print(i)
		start = time.time()
		(out)=pool.map(filled,sample_seq_list)
		yields=np.concatenate(out)
		yields=yields.flatten()
		end = time.time()
		m.append(end-start)
		print('multiprocessing total: %f'%m[-1])

	# print('ray pool : %f +/- %f' %(np.mean(p),np.std(p)))
	# print('ray normal: %f +/- %f' %(np.mean(E),np.std(E)))
	print('multiprocessing.pool: %f +/- %f'%(np.mean(m),np.std(m)))


if __name__ == '__main__':
	'Only run 1 at a time'
	# build_new_model()
	main()

# start=time.time()
# # res=pool.map(lambda actor, value: actor.par_test.remote(value), values=sample_seq_list)
# # print(list(res))
# # print('ray %f'%(time.time()-start))
# #
# #
# #
# # start = time.time()
# # res = pool.map(lambda actor, value: actor.par_test.remote(value), values=sample_seq_list)
# # print(list(res))
# # print('ray %f' % (time.time() - start))
# start = time.time()
# res=ray.get([actor.par_test.remote() for actor in inits])
# end=time.time()
# print(end-start)