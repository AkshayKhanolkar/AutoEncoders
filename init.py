from build import encoder
from layers import tanH, Sigmoid, RelU


def test_ae(	
			init_params,
			build_params,
			train_params,
			flagy = True,
		):
	
	ae = encoder(
			init_params = init_params,
			flagy = flagy 
		       )
			   
	ae.buildit(
			build_params = build_params,
			flagy = flagy
	          )    
	   
	ae.train(
	   		train_params = train_params,
			flagy = flagy
		     )
				
### Boiler Plate ###
if __name__ == '__main__':
    
	init_params = {
					"n_train_batches" : 4,
					"batch_size"	  : 4,
					"img_ht"          : 32,
					"img_wdt"         : 32,
					"tile_ht"         : 2,
					"tile_wdt"        : 2,
					"output_folder"   : 'mom_n',
					"rmse_folder"	  : 'rmse_mom_n',
					"disp_flag"       : True,
					"dataset"	  : 'face'
					
					}
					
	build_params = {
					"learning_rate"  :  0.5,
					"n_hidden_enc"   :  [ 256 ],
					"n_hidden_dec"   :  [ 256 ],
					"activation"	 :  [ Sigmoid ], 	
					"cost_fun"	 :  'rmse',
					"tied_weights"	 :  True,
					"LR_decay"	 :  0.0007,
					"begin_mom"	 :  0.4,
					"end_mom"	 :  0.85,
					"mom_thrs"	 :  100
					}
					
	train_params = {
					"training_epochs" : 400
					}
	
	flagy = True
					
	test_ae(
			init_params		= init_params,
			build_params    	= build_params,
			train_params    	= train_params,
			flagy			= flagy,
	)
	



