import numpy as np
import nexusformat.nexus as nx


def build_detector(rawdata):
        pixel_size = nx.NXfield(value=150e-6, units='m')
        beam_center_x = nx.NXfield(value=0, units='m')
        beam_center_y = nx.NXfield(value=0, units='m')
        camera_length = nx.NXfield(value=0.7, units='m')
        return nx.NXdetector(
                x_pixel_size=pixel_size,
                y_pixel_size=pixel_size,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                layout='area',
                camera_length=camera_length,
                data=rawdata,
        )

def build_scan(scan_x, scan_y):
    x_coordinates = np.linspace(-10e-9, 10e-9, scan_x)
    y_coordinates = np.linspace(-10e-9, 10e-9, scan_y)
    return nx.NXbeam(scan_x=x_coordinates, scan_y=y_coordinates)

root = nx.NXroot()
root.insert(nx.NXentry(), name='entry')

e = root.entries['entry']

e.insert(nx.NXinstrument(), 'instrument')
i = e.entries['instrument']
i.insert(build_scan(50, 50), 'scan')
rawdata = np.random.random((50, 50, 20, 20))
i.insert(build_detector(rawdata), 'detector')

e.insert(nx.NXdata(nx.NXlink(root.entry.instrument.detector.data), ('scan_x', 'scan_y', 'sensor_x', 'sensor_y')), 'data')

root.save("test_nexus_file.nxs")
