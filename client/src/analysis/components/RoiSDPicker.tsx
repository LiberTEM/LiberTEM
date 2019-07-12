import * as React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes, SDFrameParams } from "../../messages";
import * as analysisActions from "../actions";

const useRoiSDPicker = ({ analysisId, enabled, jobIndex, roiParameters}: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    jobIndex: number,
    analysisId: string;
    roiParameters: SDFrameParams;
}) => {

    const dispatch = useDispatch();
    let x;
    let y;
    let width;
    let height;

    if("shape" in roiParameters.roi) {
        ({ x, y, width, height } = roiParameters.roi);
    }

    React.useEffect(() => {
        const handle = setTimeout(() => {
            if (enabled) {
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                    type: AnalysisTypes.SD_FRAMES,
                    parameters: roiParameters,
                }))
            }
        }, 100);

        return () => clearTimeout(handle);
    }, [analysisId, enabled, jobIndex, x, y, width, height]);

    return {
    };
};

export { useRoiSDPicker };

