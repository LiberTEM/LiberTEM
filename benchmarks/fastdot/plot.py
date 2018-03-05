import sys
import math
import seaborn
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s [csv-files]\n" % sys.argv[0])
        sys.exit(1)

    files = sys.argv[1:]

    for i, filename in enumerate(files):
        df = pd.read_csv(filename)

        if 'wall_time' in df:
            values = 'wall_time'
        else:
            values = 'hot'

        df['framesize'] = df['framesize'].apply(lambda x: int(math.sqrt(x)))

        table = df.pivot_table(
            values=values,
            columns='stackheight',
            index=['framesize', 'maskcount']
        )

        plt.figure(i)
        plt.title(filename)
        cmap = "RdYlGn_r"
        seaborn.heatmap(table, annot=True, cmap=cmap, fmt=".4f")

    plt.show()


if __name__ == "__main__":
    main()
