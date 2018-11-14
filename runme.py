import os
if __name__ == '__main__':
	print ('dataprocessing ...')
	os.system("python3 dataprocessing.py")
	print ('train model...')
	os.system('python3 run_model.py')