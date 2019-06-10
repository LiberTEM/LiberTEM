from libertem.udf import UDF

class test_buffer(UDF):

	def get_results_buffer(self):
		"""
		Initialize a buffer of different types

		The first buffer "single_buffer" specifies a buffer of
		dimension (16, 16) with entries of type float32

		The second buffer "sig_buffer" specifies a buffer whose
		first component dimension is increased by 2 from that of
		signal dimension

		The third buffer "nav_buffer" specifies a buffer whose
		first component dimension is increased by 1 and the second
		component dimension is increased by 2 from that of navigation
		dimension
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