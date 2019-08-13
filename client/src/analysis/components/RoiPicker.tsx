import * as React from "react";
import { useDispatch } from "react-redux";

import { AnalysisTypes, FrameParams } from "../../messages";
import * as analysisActions from "../actions";

const useRoiPicker = ({ analysisId, enabled, jobIndex, roiParameters, analys}: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    jobIndex: number,
    analysisId: string;
    roiParameters: FrameParams;
    analys: AnalysisTypes.SD_FRAMES|AnalysisTypes.SUM_FRAMES
}) => {


    
    const dispatch = useDispatch();

    React.useEffect(() => {
        const handle = setTimeout(() => {
            if ((enabled)&&(analys===AnalysisTypes.SD_FRAMES)) {
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                    type: AnalysisTypes.SD_FRAMES,
                    parameters: roiParameters,
                }))
            }
        }, 100);

        return () => clearTimeout(handle);
    }, [analysisId, enabled, jobIndex, JSON.stringify(roiParameters)]);


    React.useEffect(() => {
        const handle = setTimeout(() => {
            if ((enabled)&&(analys===AnalysisTypes.SUM_FRAMES)) {
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                    type: AnalysisTypes.SUM_FRAMES,
                    parameters: roiParameters,
                }))
            }
        }, 100);

        return () => clearTimeout(handle);
    }, [analysisId, enabled, jobIndex, JSON.stringify(roiParameters)]);


   

    return {
    };
};

export { useRoiPicker };

