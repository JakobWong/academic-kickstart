+++
page_type = "post"

title = "How to compute optical flow based on camera motion and image depth map"
summary = "basic photographical geometry"
date = "2019-03-12"
authors = ["admin"]
tags = ["3d reconstruction", "optical flow", "camera motion"]
categories = ["vision"]

+++

Optical flow depicts the relative motion between pixels in image _1_ and pixels in pixels in image _2_. In computer vision, optical flows are usually represented by an array of size `[h, w, 2]`, where the _(i, j)_ th element `optical_flow[i,j,:]` is a 2-d flow vector describing the axis-wise motion of 2-d grid coordinates. The computation of optical flow between two images often requires finding the corresponding pixels between these images, which can be difficult in texture-less scenes. However, one can compute a relatively precise optical flow from image _1_ to image _2_ however difficult the scenario might be, as long as he know the depth map of image _1_ and the underlying camera motion from view point _1_ to view point _2_ we. Let's see how to do that!

It's good to start with a pixel point _p_ in image plane _1_ with coordinates _(u,v)_. Given the depth map of image _1_ and camera instrinsic of camera _1_, we can reproject _p_ back to 3-D space as _p'_. The coordinates of _p'_ is _(x1,y1,z1)_ under camera _1_ coordinates. Such process is described in mathematics as a map from a region of 2-D plane to a surface in 3-D space

$$ 
S: \Omega \rightarrow \mathbb{R}^3, \hspace{10pt} (x,y) \rightarrow S(x,y) \\\\\\
S(x,y) = \left( \frac{(x+o\_x)\cdot depth(x,y)}{f_x}, \frac{(y+o\_y)\cdot depth(x,y)}{f_y}, depth(x,y) \right) $$


Below comes the matlab program to implement this idea:
At first, we need to construct the normalized coordinate.

```Python
import numpy as np

def constructNormalizedCoordinate(h, w, K):
	"""
	Make normalized 2-D coordinate arrays for vectorized evaluations of 
	2-D scalar/vector fields over 2-D grids.

	INPUT:
		h:	height of the 2-D grid
		w:	width of the 2-D grid
		K:	intrinsic parameters of the camera

	OUTPUT:
		normalized_coordinates:  the normalized 2-D grid coordinates
	"""

	fx = K[0][0]
	fy = K[1][1]
	ox = K[0][2]
	oy = K[1][2]

	# make vertical lines and horizontal lines for the 2-D grid
	x = np.arange(0,w)
	y = np.arange(0,y)

	# normalize the grid coordinates
	tilde_x, tilde_y = np.meshgrid(x,y)
	tilde_x = (tilde_x - x0)/fx
	tilde_y = (tilde_y - y0)/fy

	# stack the grid coordinates along the 3rd axis 
	# and form a matrix of size [h,w,2]
	normalized_coordinates = np.stack([tilde_x,tilde_y],axis=-1)

	return normalized_coordinates
```

Using the normalized_coordinates, we then reproject the 2-D pixels onto 3-D space:

```python

def reoprojectDepthImage(depth, K):
	"""
	Reproject the depth image to 3D point clouds

	INPUT:
		depth: a 2-D matrix
		K: instrinsic parameters pf the camera

	OUTPUT:

	"""
	# choose the valid depth value
	valid_mask = np.nonzero(depth);

	tmp_depth = depth[...,np.newaxis] # the same as tmp_depth = np.expand_dims(depth,axis=-1)

	normalized_coords = constructNormalizedCoordinates(h, w, K)
	pointclouds = np.stack([normalized_coordinates*tmp_depth,depth],axis=-1)

	# convert pointclouds into a list
	pointclouds = np.reshape(pointclouds, -1, 3);
	pointclouds = np.array(pointclouds[valid_mask, :]);

	return pointclouds, valid_mask

```

Then, the rigid body motion from camera position _1_ to camera position _2_, depicted by the relative camera motion tells us how _p'_ is represented in camera _2_ coordinates. In other words, `p'' = rigid_body_motion(p')`, where `rigid_body_motion` is a function of `relative_camera_motion`.

```python
def warpPointClouds(pointclouds_list,relative_camera_motion):
	"""
	Warp the 3-D point clouds using rigid body motion

	INPUT:
		pointclouds_list: a matrix of size [num_points,3]
		relative_camera_motion: a homogeneous matrix of size [4,4]

	OUTPUT:
		warped_pointclouds: a matrix of size [num_points,3]
	"""

	warped_pointclouds = np.matmul(pointclouds_list,relative_camera_motion[:3,:3]) + relative_camera_motion[:3,3]

	return warped_pointclouds
```

Thereafter, we project _p''_ (camera _2_ coordinates) into 2-D image plane _2_.

```python
def projectPointCloud(pointclouds_list, K, h, w):
	"""
	Project the pointclouds into image coordinates

	INPUT:
		pointclouds: a matrix of size [num_point, 3]
		K: intrinsic camera parameters
		h: height of image data
		w: width of image data

	OUTPUT:
		img_coords: a matrix of size [num_pixel, 2]
		valid_mask: indices of valid elements
	"""

	depth = np.expand_dims(pointclouds[:,2],axis=-1)
	img_coords = pointclouds / depth
	img_coords = np.matmul(img_coords, np.transpose(K))
	img_coords = img_coords[:,:2]

	valid_mask = np.where((img_coords[:,0] > 0) and (img_coords[:,1] < w) and (img_coords[:,1] > 0) and img_coords[:,1] < h)
	valid_mask = img_coords[valid_mask, :] 

	return img_coords, valid_mask
```

The codes above are python implementation of [this github project](https://github.com/LucasSheng/dense-rgbd-visual-odometry). Thank _LucasSheng_ for such great Matlab implementation.