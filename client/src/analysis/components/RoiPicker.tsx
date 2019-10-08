import * as React from "react";
import { useDispatch } from "react-redux";
import { AnalysisDetails, AnalysisTypes, FrameParams } from "../../messages";
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
            if (enabled) {
                // work around typescript bug in 3.2.X
                // explicit cast should be removed when upgrading ts
                const analysisDetails = {
                    type: analysis,
                    parameters: roiParameters,
                } as AnalysisDetails
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, analysisDetails))
            }
        }, 100);

        return () => clearTimeout(handle);
    }, [analysis, analysisId, enabled, jobIndex, JSON.stringify(roiParameters)]);

    return {
    };
};

export { useRoiPicker };

