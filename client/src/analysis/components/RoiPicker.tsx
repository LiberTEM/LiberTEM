import * as React from "react";
import { useDispatch } from "react-redux";

import { AnalysisTypes, FrameParams } from "../../messages";
import * as analysisActions from "../actions";

const useRoiPicker = ({ analysisId, enabled, jobIndex, roiParameters, analysis}: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    jobIndex: number,
    analysisId: string;
    roiParameters: FrameParams;
    analysis: AnalysisTypes.SD_FRAMES|AnalysisTypes.SUM_FRAMES
}) => {


    
    const dispatch = useDispatch();

    React.useEffect(() => {
        const handle = setTimeout(() => {
            if ((enabled)&&(analysis===AnalysisTypes.SD_FRAMES)) {
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                    type: AnalysisTypes.SD_FRAMES,
                    parameters: roiParameters,
                }))
            }
        }, 100);

        return () => clearTimeout(handle);
    }, [analysis, analysisId, enabled, jobIndex, JSON.stringify(roiParameters)]);


    React.useEffect(() => {
        const handle = setTimeout(() => {
            if ((enabled)&&(analysis===AnalysisTypes.SUM_FRAMES)) {
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                    type: AnalysisTypes.SUM_FRAMES,
                    parameters: roiParameters,
                }))
            }
        }, 100);

        return () => clearTimeout(handle);
    }, [analysis, analysisId, enabled, jobIndex, JSON.stringify(roiParameters)]);


   

    return {
    };
};

export { useRoiPicker };

