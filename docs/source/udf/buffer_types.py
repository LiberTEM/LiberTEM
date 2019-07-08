from libertem.udf import UDF


class TestBuffer(UDF):

    def get_result_buffers(self):
        """
        Initialize a buffer of different types

        Suppose our dataset has dimension (14, 14, 32, 32),
        where the first two dimensions represent the navigation
        dimension and the last two dimensions represent the signal dimension.

        The first buffer "single_buffer" specifies a buffer of
        dimension (16, 16) with entries of type float32.

        The second buffer "sig_buffer" specifies a buffer whose
        dimension is initialized to be the same as the signal dimension.
        With 'extra_shape = (2,)', two additional dimensions are introduced
        in addition to the signal dimension, so the resulting
        dimension for this buffer will be (32, 32, 2). One can store, for
        instance, two float32 numbers of interest for each frame as a result.

        The third buffer "nav_buffer" specifies a buffer whose
        dimension is initialized to be the same as the navigation dimension.
        With 'extra_shape = (1, 2)', two additional dimensions are introduced
        in addition to the navigation dimension, so the resulting dimension
        for this buffer will be (14, 14, 1, 2).
        """
        return {
            "single_buffer": self.buffer(
                kind="single",
                extra_shape=(16, 16),
                dtype="float32",
            ),

            "sig_buffer": self.buffer(
                kind="sig",
                extra_shape=(2,),
                dtype="float32",
            ),

            "nav_buffer": self.buffer(
                kind="nav",
                extra_shape=(1, 2),
                dtype="float32",
            )
        }
