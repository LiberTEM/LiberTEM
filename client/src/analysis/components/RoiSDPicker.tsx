import * as React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes, SDFrameParams } from "../../messages";
import * as analysisActions from "../actions";


const useRoiSDPicker = ({ analysisId, enabled, jobIndex, roiParameters, shapes}: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    jobIndex: number,
    analysisId: string;
    roiParameters: SDFrameParams;
    shapes: string;
}) => {

    if (shapes==="rect")
    {const dispatch = useDispatch();
    let x;
    let y;
    let width;
    let height;

    if("width" in roiParameters.roi) {
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
    }, [analysisId, enabled, jobIndex, x, y, width, height]);}


    if (shapes==="disk")
    {const dispatch = useDispatch();
    let cx;
    let cy;
    let r;

    if("r" in roiParameters.roi) {
        ({ cx, cy, r } = roiParameters.roi);
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
    }, [analysisId, enabled, jobIndex, cx, cy, r]);}
    

    return {
    };
};

export { useRoiSDPicker };

