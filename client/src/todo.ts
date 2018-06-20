import { openDataset } from './dataset/api';
import { startJob } from './job/api';



export async function initialize() {
    const dsResponseEMPAD1 = openDataset({
        name: "test dataset",
        path: "/test/index.json",
        tileshape: [1, 8, 128, 128],
        type: "HDFS",
    });
    const dsResponseEMPAD2 = openDataset({
        name: "e field mapping acquisition 8",
        path: "/e-field-acquisition_8_0tilt_0V/index.json",
        tileshape: [1, 8, 128, 128],
        type: "HDFS",
    });
    const dsResponseEMPAD3 = openDataset({
        name: "e field mapping acquisition 10",
        path: "/e-field-acquisition_10_0tilt_40V/index.json",
        tileshape: [1, 8, 128, 128],
        type: "HDFS",
    });
    const dss = {
        eField1: await dsResponseEMPAD2,
        eField2: await dsResponseEMPAD3,
        test: await dsResponseEMPAD1,
    };

    await startJob(
        dss.test.dataset,
        [
            { shape: "ring", cx: 64, cy: 65, ri: 0, ro: 5 },
            { shape: "ring", cx: 64, cy: 65, ri: 0, ro: 45 },
            { shape: "ring", cx: 64, cy: 65, ri: 50, ro: 63 },
        ]
    );
}