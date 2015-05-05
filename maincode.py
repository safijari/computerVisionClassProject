"""

Classification of Objects in Satellite Images
	A Class Project by: Jariullah Safi, Michael Benjamin Seven Sechwarzenegger Master Chief Hauser

05-04-2015

This is the main file for the project. It runs the code over a few different test cases using various
combinations of filters and calculates classification accuracy based on overlap between processed 
images and hand made ground truth.

"""


# Import necessary libraries and functions
import cv2 # Image processing
import time
from skimage.segmentation import slic # For SLIC segmentation
import numpy as np
from sklearn.cluster import KMeans # For quick NN searches on K-means results
from scipy.io import loadmat
import cPickle
import pickle
import pdb

### Utility Functions ###
def applyFilterBank(im, ftype):
	"""
	inputs:
		im 		- a matrix containing a single image in bgr (from OpenCV)
		ftype 	- types of filter bank used
				Options are (as strings)
					- winn     # the 17 filters used by Minka and Winn in the paper
					- gabor1   # the smaller gabor filters from Hauser's bank
					- gabor2   # the larger gabor filters
					- gabor    # all gabor filters
					- all      # all fitlers combined
	outputs:
		fr      - a dictionary containing all filter responses as a dim dimensional matrix
		keys    - sorted list of keys in the dictionary
		dim     - dimension of the filter space
	"""

	# First, extract the (possibly) necessary channels
	im2 = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
	L   = array(im2[:,:,0], np.float32)
	a   = array(im2[:,:,1], np.float32)
	b   = array(im2[:,:,2], np.float32)
	B   = array(im[:,:,0], np.float32)
	G   = array(im[:,:,1], np.float32)
	R   = array(im[:,:,2], np.float32)

	gf = loadmat("data/GFBfile.mat")['GFBr']
	fr = {}

	if ftype == "winn" or ftype == "all":
		for i in [1,2,4,8]:
			# Calculating gaussians for each channel for 1,2, and 4
			# and only L for 8
			fr["gauss"+str(i)+"L"] = np.float32(cv2.GaussianBlur(L, (51,51), i))
			if i < 8:
				fr["gauss"+str(i)+"a"] = np.float32(cv2.GaussianBlur(a, (51,51), i))
				fr["gauss"+str(i)+"b"] = np.float32(cv2.GaussianBlur(b, (51,51), i))
				
			# Calculating the LoG for each gaussian of L
			fr["gauss"+str(i)+"LOG"] = cv2.Laplacian(fr["gauss"+str(i)+"L"], cv2.CV_32F, 51)
			if i == 8:
				fr.pop("gauss"+str(i)+"L")
			
		# Then calculate the derivatives in each direction
		for i in [2,4]:
			fr["gauss"+str(i)+"dx"] = cv2.Sobel(fr["gauss"+str(i)+"L"], cv2.CV_32F, 1, 0, ksize = 3)
			fr["gauss"+str(i)+"dy"] = cv2.Sobel(fr["gauss"+str(i)+"L"], cv2.CV_32F, 0, 1, ksize = 3)

	if ftype == "gabor1" or ftype == "gabor" or ftype == "all":
		for i in range(0,24,2):
			fr["gabor_B_"+str(i)] = cv2.filter2D(B, cv2.CV_32F, gf[:,:,i])
			fr["gabor_G_"+str(i)] = cv2.filter2D(G, cv2.CV_32F, gf[:,:,i])
			fr["gabor_R_"+str(i)] = cv2.filter2D(R, cv2.CV_32F, gf[:,:,i])

	if ftype == "gabor2" or ftype == "gabor" or ftype == "all":
		for i in range(1,24,2):
			fr["gabor_B_"+str(i)] = cv2.filter2D(B, cv2.CV_32F, gf[:,:,i])
			fr["gabor_G_"+str(i)] = cv2.filter2D(G, cv2.CV_32F, gf[:,:,i])
			fr["gabor_R_"+str(i)] = cv2.filter2D(R, cv2.CV_32F, gf[:,:,i])

	keys = sort(fr.keys())
	dim = (keys.shape)[0]

	return fr, keys, dim

def FR2data(FR, ftype, ims):
	"""
	inputs:
		FR 		- Dictionary of all filter responses
		ftype 	- String of filter types (see above function)
		ims 	- Image numbers to consider
	"""
	data = array([])

	for i in ims:
		fr, keys, dim = FR[ftype+'_image_'+str(i+1)]
		if data.shape[0] == 0:
			data = np.vstack([fr[key].flatten() for key in keys])
		else:
			data = np.hstack((data, [fr[key].flatten() for key in keys]))
	data = data.transpose()

	return data

def fitClusterCenters(data, numClust):
	"""
	inputs:
		data 		- the filter response flattened matrix
		numClust 	- number of cluster centers

	outputs:
		centers 	- learned or loaded cluster centers
	"""
	g = KMeans(numClust)
	cname = 'data/clustercenters/'+ft + str(numClust) + str(ims) + '.pkl'

	try:
		f = open(cname)
		centers = cPickle.load(f)
		f.close()
	except IOError:
		centers = cv2.kmeans(data, numClust,None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00000000001), 1, cv2.KMEANS_RANDOM_CENTERS)[2]
		f = open(cname, 'wb')
		cPickle.dump(centers, f)
		f.close()

	g.fit(centers)

	return g

def makeHistograms(filtresp, ft, im, ims, impal):
	"""
	inputs:
		filtresp 	- the dictionary of all filter responses
		ft 			- the filter bank type (see applyFilterBank)
		im 			- the dictionary of all images
		ims 		- the images being processed
		impal 		- the pallette image
	output:
		hist 		- an array of three histograms over the cluster centers
	"""
	hist = [zeros((1,numClust)), zeros((1,numClust)), zeros((1,numClust)),ones((1,numClust))]

	for i in ims:
		# get filter response matrix
		imlab = im['image_'+str(i+1)+'_labeled']
		fr, keys, dim = filtresp[ft+'_image_'+str(i+1)]
		fri = zeros((640,640,dim))
		for j in range(dim):
			fri[:,:,j] = fr[keys[j]]

		# get image labels
		labmat = zeros((640,640,3))
		#labmat = zeros((640,640,7))
		for m in range(3):
			#labmat[:,:,m] = (imlab[:,:,0] == impal[0][m][0])*(imlab[:,:,1] == impal[0][m][1])*(imlab[:,:,2] == impal[0][m][2])
			
			if m == 0:
				labmat[:,:,m] = (imlab[:,:,0] == impal[0][m][0])*(imlab[:,:,1] == impal[0][m][1])*(imlab[:,:,2] == impal[0][m][2])
			if m == 1:
				labmat[:,:,m] = (imlab[:,:,0] == impal[0][2][0])*(imlab[:,:,1] == impal[0][2][1])*(imlab[:,:,2] == impal[0][2][2]) + \
				(imlab[:,:,0] == impal[0][3][0])*(imlab[:,:,1] == impal[0][3][1])*(imlab[:,:,2] == impal[0][3][2])
			if m == 2:
				labmat[:,:,m] = (imlab[:,:,0] == impal[0][1][0])*(imlab[:,:,1] == impal[0][1][1])*(imlab[:,:,2] == impal[0][1][2]) + \
				(imlab[:,:,0] == impal[0][4][0])*(imlab[:,:,1] == impal[0][4][1])*(imlab[:,:,2] == impal[0][4][2]) + \
				(imlab[:,:,0] == impal[0][5][0])*(imlab[:,:,1] == impal[0][5][1])*(imlab[:,:,2] == impal[0][5][2])
			
			# add to the histogram
			x = []; y = []; hst = []; clusterNumbers = []
			(x,y) = nonzero(labmat[:,:,m])

			clusterNumbers = g.predict(fri[x,y])
			hst = histogram(clusterNumbers, range(numClust))
			hist[m] += bincount(clusterNumbers, minlength = numClust)
	for i in range(3):
		hist[i] = hist[i][0]
		hist[i] = hist[i]/sum(hist[i])
	hist[3] = (hist[3]/sum(hist[3])).transpose()
	#hist[6] = (hist[6]/sum(hist[6]))[0]

	return hist

def labelImage(filtresp, testImage, ft, hist, impalR):
	fr, keys, dim = filtresp[ft+'_image_'+str(testImage+1)]

	fri = zeros((640,640,dim))

	for j in range(dim):
		fri[:,:,j] = fr[keys[j]]

	for i in range(numSeg):
		#if i % 100 == 0:
		#	print str(float(i)/float(numSeg)*100) + '%'
		ww = where(seg == i)
		clusterNumbers = g.predict(fri[ww[0], ww[1]])

		histo = np.double(bincount(array(clusterNumbers).flatten(), minlength = numClust))
		histo /= sum(histo)

		dists = array([dot(histo,histO)/sqrt(norm(histo)*norm(histO)) for histO in hist]).flatten()
		for m in range(3):
			classIm[ww[0], ww[1], m] = impalR[argmax(dists), m]

	return classIm


if __name__ == '__main__':
	numImages = 4
	testImage = 0
	numClust = 2000
	numSeg = 30**2

	# Load images
	print "loading images"
	im = {}

	for i in range(numImages):
		im['image_'+str(i+1)] = cv2.imread('data/images/'+str(i+1)+'.png')
		im['image_'+str(i+1)+'_labeled'] = cv2.imread('data/labeled/'+str(i+1)+'.png')

	if 'filtresp' not in vars():
		try:
			f = open('/home/jariullah/fr.pkl', 'rb')
			filtresp = cPickle.load(f)
			f.close()
		except IOError:
			filtresp = {}
			for i in range(numImages):
				# Create filter responses
				for j in ["winn", "gabor1", "gabor2", "gabor", "all"]:
					filtresp[j+"_image_"+str(i+1)] = applyFilterBank(im['image_'+str(i+1)], j)
			f = open('/home/jariullah/fr.pkl', 'wb')
			cPickle.dump(filtresp, f,2)
			f.close()

	# Load pallete image
	impal = cv2.imread('data/labelPallette.png')
	#impal = array([np.vstack((impal[0], [0,0,0]))])
	impalR = np.vstack((impal[0,[0,2,5],:], [0,0,0]))

	for numClust in [1000]:
		for testImage in [3]:
			#for ft in [""" "winn", "gabor1", """ "gabor2", "gabor", "all"]:
			for ft in ["gabor1", "winn"]:
				print "processing " + ft + " for image " + str(testImage + 1) + " with numClust " + str(numClust)
				# Make visual dictionary (save?)
				print "making data array"
				ims = range(numImages)
				ims.pop(testImage)

				data = FR2data(filtresp, ft, ims)

				print "doing kmeans clustering"

				g = fitClusterCenters(data, numClust)

				# Make histograms
				
				hist = makeHistograms(filtresp, ft, im, ims, impal)

				# Segment and label test image
				seg = slic(im['image_'+str(testImage+1)][:,:,::-1],n_segments=numSeg, compactness = 15, sigma=1, convert2lab = True)
				classIm = zeros((640,640,3), dtype=uint8)

				print "labeling now"

				classIm = labelImage(filtresp, testImage, ft, hist, impalR)

				fname = ft + '_image_' + str(testImage+1) + '_numClust_' + str(numClust)

				cv2.imwrite('data/predictedImages/'+fname + '.png',classIm)

	# Compute test image accuracy