+++
page_type = "post"

title = "How to implement depth-aware convolutional layer in tensorflow"
summary = "By factorizing depth-aware convolution as a series of `tf.nn.conv2d` operations, we can obtain a simple and elegant implementation."
date = "2019-03-14"
authors = ["admin"]
tags = ["deep learning", "basic convolutional operations", "3d reconstruction"]
categories = ["vision"]

[[gallery_item]]
album = "img"
image = "img1.png"
caption = "DAconv_matrix"


[[gallery_item]]
album = "img2"
image = "img2.png"
caption = "DAconv_vector"

+++

Let's start with a brief review over Depth-Aware Convolution (DA Conv). For better understanding, you are highly recommended to read [the original paper](https://arxiv.org/abs/1803.06791).

A conventional conv operation is weighted summation of a local grid. For each pixel location $p\_0$ valid for conv operation, the output of 2D conv on $p\_0$ is 
$$ conv(p\_0, I, \vec{w}) = \sum\_{p \in \Omega\_{p\_0}} \vec{w}(p) \cdot I(p)$$
where $\Omega\_p$ denotes the grid neibourhood with center $p\_0$, $\vec{w}$ denotes the convolutional kernel, $I$ denotes the image intensity.

However, such fixed grid computation structure is limited to model geometric variance. To incorporating the geometric information from depth images into conv operation, we need some modification. Specifically, DA Conv (Depth-Aware Conv) exploits the depth correlation between pixels, by simply adding depth similarity term to the computation, resulting in two sets of weights in convolution: $1)$ the learnable convolution kernel $\vec{w}$; $2)$ depth similarity $F\_{\mathbf{D}}$ between two pixels. The output of a DAConv operation thus become:
$$\begin{aligned} 
DAconv(p\_0, I, \vec{w}, F\_{\mathbf{D}}) = \sum\_{p \in \Omega\_{p\_0}} \vec{w}(p) \cdot F\_{\mathbf{D}}(p,p\_0) \cdot I(p)\end{aligned}$$
And $F\_{\mathbf{D}}(p\_i,p\_j)$ is defined as
$$ F\_{\mathbf{D}}(p\_i,p\_j) = \text{exp}(- \alpha |\mathbf{D}(p\_i) - \mathbf{D}(p\_j)|)$$

where $\alpha$ is a constant. $F\_{\mathbf{D}}$ is choosen such that pixels with similar depths have more impact on each other.

With a quick glimpse on the definition, one might think it is impossible to implement DAconv using basic tensorflow operations. However, with two sets of specially designed kernel, we can convert DAconv into series of conv.

We can visualize the computation of DAconv as a combination of two 2-D matrix operations:

{{< gallery album="img1" >}}

which can be rewritten as 

{{< gallery album="img2" >}}

note that for each convolution center $p\_0$, we need to get three tensors of size `[1,1,kernel_size * kernel_size]` so as to compute DAconv. The tensors $F\_{\mathbf{D}}$ and $I$ can be obtained by convolving the image intensity with two sets of kernel respectively.

First, consider a set of kernels $ \\{ \text{depKernel} \_{i,j} \\} \_{\hspace{3pt} i \in \\{1,\dots,h\\},\hspace{3pt} j \in \\{1,\dots,w\\}}$, with $\text{depKernel} \_{\lfloor{\frac{\text{kernel_size} - 1}{2}}\rfloor,\\, \lfloor{\frac{\text{kernel_size} - 1}{2}}\rfloor} = \mathbf{\large{0}} \_{\small \text{kernel_size} \times \small \text{kernel_size}}$

and $\text{depKernel}\_{\text{i},\text{j}} = $
$$
\begin{aligned}
 & & & & & \hspace{5pt} \downarrow j^{th} \hspace{3pt}col	&	\\\\\\
\end{aligned}
$$
$$
\left[
\begin{matrix} 
	&	   &		&		&0		&			&	\\\\\\
    &      &		&		&\vdots	&			&	\\\\\\
   0& \dots&\dots	&\dots	&1		&\dots		&0	\\\\\\
	&	   &		&-1		&\vdots	&			&	\\\\\\
	&	   &		&	    &\vdots &			&	\\\\\\
	&	   &		&	    &\vdots &			&	\\\\\\
	&	   &		&		&0		&			&	\\\\\\
\end{matrix}
\right]
\begin{matrix}
 \leftarrow i^{th} \hspace{5pt}row \\\\\\
 \\\\\\
 \\\\\\
 \\\\\\
\end{matrix}
$$

such kernel set can be implemented by the following lines of code in python

```python
import tensorflow as tf
import numpy as np
def depth_aware_conv(img, depth, NC, kernel_size, conv_kernel):
    """
    Implement Depth-Aware Convolutional layer using basic tf operations
    
    Assuming that conv_kernel has shape [1,1,kernel_size*kernel_size*NC]

    """

    # create Depth-Aware Kernel set
    center = np.int(np.ceil(kernel_size/2))
    depKernel = np.zeros((kernel_size*kernel_size,1,kernel_size*kernel_size),dtype=np.float32) # (H,W,input_channel,output_channel)
    for i in range(kernel_size*kernel_size):
        depKernel[(center-1)*(kernel_size+1),0,i] = -1
        depKernel[i,0,i] = depKernel[i,0,i] + 1
    depKernel = np.reshape(depKernel,(kernel_size,kernel_size,1,kernel_size*kernel_size))
    depKernel = tf.constant(depKernel)
```

Similarly, we may deduce $\\{ \text{reshapeKernel}\_{\text{i},\text{j}}\\} = $
$$
\begin{aligned}
 & & & & & \hspace{5pt} \downarrow j^{th} \hspace{3pt}col	&	\\\\\\
\end{aligned}
$$
$$
\left[
\begin{matrix} 
	&	   &		&		&0		&			&	\\\\\\
    &      &		&		&\vdots	&			&	\\\\\\
   0& \dots&\dots	&\dots	&1		&\dots		&0	\\\\\\
	&	   &		&		&\vdots	&			&	\\\\\\
	&	   &		&	    &\vdots &			&	\\\\\\
	&	   &		&	    &\vdots &			&	\\\\\\
	&	   &		&		&0		&			&	\\\\\\
\end{matrix}
\right]
\begin{matrix}
 \leftarrow i^{th} \hspace{5pt}row \\\\\\
 \\\\\\
 \\\\\\
 \\\\\\
\end{matrix}
$$

which, as well, can be obtained by these lines of code:

```python
    # create Reshape Kernel set
    reshapeKernel = np.zeros((kernel_size*kernel_size,1,kernel_size*kernel_size),dtype=np.float32)
    for i in range(kernel_size*kernel_size):
        reshapeKernel[i,0,i] = 1
    reshapeKernel = np.reshape(reshapeKernel,(kernel_size,kernel_size,1,kernel_size*kernel_size))    
    reshapeKernel = tf.constant(reshapeKernel)
```

With the lines of code above, we succeed in turning $\text{kernel_size} \times \text{kernel_size}$ 2D grid into $1 \times 1 \times \text{kernel_size}^2$ 3D tensor and thereby making it possible to compute DAconv by tensorflow operations:

```python
    # reshape [n,n] 2D local grid to [1,1,n*n] 3D local tensor, channel by channel
    F_D = tf.nn.conv2d(depth,depKernel,strides=[1,1,1,1],padding='VALID')
    F_D = tf.exp(-1 * tf.abs(F_D))
    I = tf.nn.conv2d(img[:,:,:,0][...,np.newaxis],reshapeKernel,strides=[1,1,1,1],padding='VALID')
    output = tf.reduce_sum(F_D*I*conv_kernel,axis=-1)
    
    for i in range(1,NC):
        I = tf.nn.conv2d(img[:,:,:,i][...,np.newaxis],reshapeKernel,strides=[1,1,1,1],padding='VALID')
        output += tf.reduce_sum(F_D*I*conv_kernel,axis=-1)
```

To sum up, we manage to implement Depth-Aware Convolution in tensorflow:

```python
import tensorflow as tf
import numpy as np
def depth_aware_conv(img, depth, NC, kernel_size, conv_kernel):
    """
    Implement Depth-Aware Convolutional layer using basic tf operations
    
    Assuming that conv_kernel has shape [1,1,kernel_size*kernel_size*NC]

    """

    # create Depth-Aware Kernel set
    center = np.int(np.ceil(kernel_size/2))
    depKernel = np.zeros((kernel_size*kernel_size,1,kernel_size*kernel_size),dtype=np.float32) # (H,W,input_channel,output_channel)
    for i in range(kernel_size*kernel_size):
        depKernel[(center-1)*(kernel_size+1),0,i] = -1
        depKernel[i,0,i] = depKernel[i,0,i] + 1
    depKernel = np.reshape(depKernel,(kernel_size,kernel_size,1,kernel_size*kernel_size))
    depKernel = tf.constant(depKernel)

    # create Reshape Kernel set
    reshapeKernel = np.zeros((kernel_size*kernel_size,1,kernel_size*kernel_size),dtype=np.float32)
    for i in range(kernel_size*kernel_size):
        reshapeKernel[i,0,i] = 1
    reshapeKernel = np.reshape(reshapeKernel,(kernel_size,kernel_size,1,kernel_size*kernel_size))    
    reshapeKernel = tf.constant(reshapeKernel)

    # reshape [n,n] 2D local grid to [1,1,n*n] 3D local tensor, channel by channel
    F_D = tf.nn.conv2d(depth,depKernel,strides=[1,1,1,1],padding='VALID')
    F_D = tf.exp(-1 * tf.abs(F_D))
    I = tf.nn.conv2d(img[:,:,:,0][...,np.newaxis],reshapeKernel,strides=[1,1,1,1],padding='VALID')
    output = tf.reduce_sum(F_D*I*conv_kernel,axis=-1)
    
    for i in range(1,NC):
        I = tf.nn.conv2d(img[:,:,:,i][...,np.newaxis],reshapeKernel,strides=[1,1,1,1],padding='VALID')
        output += tf.reduce_sum(F_D*I*conv_kernel,axis=-1)

    return output

```

### REFERENCE

_Wang, Weiyue, and Ulrich Neumann. "Depth-aware cnn for rgb-d segmentation." Proceedings of the European Conference on Computer Vision (ECCV). 2018._