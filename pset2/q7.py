import numpy as np
import matplotlib.pyplot as plt

def compress_images(img, k_vals):
    '''
    given a greyscale img, compress through k-rank approximations

    attributes:
    - img: (height x width) array representing a greyscale image
    - k_vals: a list of k values where k is the rank of the approximation, 
    must be < height, width

    returns: 
    - images: a list of compressions corresponding to k values in ascending order
    '''
    # initialize images list
    images = []

    # create compressed image
    h, w = img.shape
    compressed = np.zeros((h, w))

    # perform SVD
    U, S_diag, VT = np.linalg.svd(img)
    
    # add outer products of the singular vectors up to the highest k value
    for idx in range(max(k_vals)+1):
        # get the singular value and vectors
        sigma = S_diag[idx]
        u = U[:, idx:idx+1]
        vt = VT[idx:idx+1, :]

        # compute and add the outer product to the compressed image
        compressed += sigma * u @ vt

        # if idx is a k value, add compressed image to the images list
        if idx in k_vals:
            images.append(compressed.copy())

    return images

file_path = 'pset2/data/mit.jpg'
img = plt.imread(file_path).mean(axis = 2) # take the mean along axis = 2 to get greyscale image

# make sure k_vals is sorted for correct labeling
k_vals = [10, 50, 100]
images = compress_images(img, k_vals)

fig, axs = plt.subplots(len(k_vals)//2+1, 2)
for idx in range(len(k_vals)):
    k = k_vals[idx]
    comp_img = images[idx]

    i = idx//2
    j = idx%2

    axs[i, j].set_title(f'{k}-rank approximation')
    axs[i, j].imshow(comp_img, cmap = 'Greys_r')

if len(k_vals) % 2:
    i, j = len(k_vals)//2, 1
else:
    i, j = len(k_vals)//2, 0

axs[i, j].set_title('original image')
axs[i, j].imshow(img, cmap = 'Greys_r')

plt.savefig('pset2/output/q7.png')
plt.show()