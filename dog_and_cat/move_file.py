import os
import os.path
import shutil

path_1 = "./data/train/"
path_2 = "./data/dog/"
path_3 = "./data/cat/"

for parent, dirnames, filenames in os.walk(path_1):

	for filename in filenames:
		#print filename
		if "dog" in filename:
			shutil.move(path_1 + filename, path_2)

		elif "cat" in filename:
			shutil.move(path_1 + filename, path_3)

		else:
			print "Error"

