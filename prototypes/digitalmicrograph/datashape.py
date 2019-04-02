import numpy as np

data = np.array(((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)))

image = DM.CreateImage(data)
image.ShowImage()

buffer = image.GetNumArray()

if (buffer == data):
    print("original data and image buffer are the same")
else:
    print("original data and image buffer are different")
    print("Shape of data: ", data.shape)
    print("Shape of buffer: ", buffer.shape)
    print("Data:")
    print(data)
    print("Buffer:")
    print(buffer)
