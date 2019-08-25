Principal Component Analysis
============================

Principal Component Analysis (herefrom referred to as PCA) is one of the most commonly used dimension reduction methods in scientific research. In short, PCA finds directions with largest variance in the data through orthogonalization of the vector space that the data lies in. PCA is relatively simple to implement compared to other methods and produces deterministic output, making it one of the most popular dimension reduction methods. However, PCA involves heavy matrix operations, which can be increasingly burdening as the data becomes large. Here, we exploit the distributed properties of User-Defined Functions (UDF) architecture by conjoining variations of PCA algorithms to obtain scalability for PCA.

How PCA works
-------------

.. image:: ./images/pca_diagram.png

As mentioned above, PCA heavily exploits the distributed nature of the UDF architecutre. First, using the partitioned data from the partitioning stage of UDF, we perform standard PCA on each of the partitions. Note that the current implementation utilizes :meth:`~libertem.udf.UDFFrameMixin.process_partition` instead of :meth:`~libertem.udf.UDFPartitionMixin.process_frame` to provide additional boost in the performance of the algorithm. Then, for each partition, we store the values for the loading matrix (matrix multiplication of the left singular matrix with the singular values), and the component matrix which is of our primary interest since the component matrix serves as a projection matrix into the lower-dimensional space and thereby achieving dimension reduction.

Once PCA is independently applied on all partitions, we proceed to `merge` these results of independent PCA results, which is done rather naturally by implementing the :meth:`~libertem.udf.UDF.merge` method in UDF class. More specifically, given the results of the PCA on each partition, we first reconstruct the original matrix using the loading and the component matrices, and then reducing the dimension of the data matrix via `hyperbox` method. `Hyperbox` method is a way to reduce the dimension of the loading matrix by finding the maximal `hyperbox` that contains the span of the vectors in the loading matrix. In doing so, we can find a simpler description for the loading matrix, which then reduces the dimension of the data, thereby leading to an increase in the overall performance. At each iteration, once `hyperbox` method is performed for the partition, we use Incremental PCA algorithm (IPCA) to incrementally update the component matrix by fitting the post-`hyperbox` data matrix. One major advantage of IPCA is that it can easily handle large-size data that standard PCA algorithm cannot, and UDF architecture served as a basis for implementing Incremental PCA.

How to use PCA
--------------
Using the UDF interface of LiberTEM, PCA requires two parameters: the data with which one wishes to reduce the dimension and the number of components to retain for dimension reduction. Note that the number of components is crucial in determining how much of the information in the original data one would like to retain. For instance, if one specifies a large number of components, then the reconstruction error would be small and most of the information is captured in the PCA-projected data. However, in this case, PCA would take longer time to process. Therefore, one needs to be aware of the trade-offs between performance and accuracy when determining how many number of components to use. Once one has specified the number of components, PCA can be used as following:

.. include:: ./pca.py
   :code:

The above code then returns a projected matrix with reduced dimension. A row of this projected matrix represents an image and a column of this projected matrix represents feature vectors that comprise the image. 

Additional Information
----------------------
For additional information on PCA, including its performance in comparison with standard PCA and various testing schemes to ensure the credibility of the method, please `follow this link to a jupyter notebook. <pca.ipynb>`_

Reference
---------
D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3, pp. 125-141, May 2008.
