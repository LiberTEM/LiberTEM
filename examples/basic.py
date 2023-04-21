import matplotlib.pyplot as plt
import libertem.api as lt


if __name__ == '__main__':
    # A path to a Quantum Detectors Merlin header file
    # Adapt to your data and data format
    dataset_path = './path_to_dataset.hdr'

    # Create a Context object to load data and run analyses
    ctx = lt.Context()
    # The key 'mib' tells LiberTEM which format to load
    # it is possible to supply 'auto' and the Context will
    # try to auto-detect the correct dataset format
    ds = ctx.load('mib', path=dataset_path)

    # Create a sum-over-disk analysis, i.e. brightfield image
    # Values for disk centre x/y and radius in pixels
    disk_sum_analysis = ctx.create_disk_analysis(ds, cx=32, cy=32, r=8)
    disk_sum_result = ctx.run(disk_sum_analysis, progress=True)

    # Plot the resulting brightfield image
    plt.imshow(disk_sum_result.intensity.raw_data)
    plt.show()
