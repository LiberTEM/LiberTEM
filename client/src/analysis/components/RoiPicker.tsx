import * as React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes, FrameParams } from "../../messages";
import * as analysisActions from "../actions";


const useRoiPicker = ({ analysisId, enabled, jobIndex, roiParameters, analysis }: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    jobIndex: number,
    analysisId: string;
    roiParameters: FrameParams;
    analysis: AnalysisTypes.SD_FRAMES | AnalysisTypes.SUM_FRAMES
}) => {
    const dispatch = useDispatch();

    React.useEffect(() => {
        const handle = setTimeout(() => {
            if (enabled) {
                const analysisDetails = {
                    type: analysis,
                    parameters: roiParameters,
                };
                dispatch(analysisActions.Actions.run(analysisId, jobIndex, analysisDetails))
            }
        }, 100);

        return () => clearTimeout(handle);
        // rules-of-hooks can't be statically validated here
        // eslint-disable-next-line
    }, [analysis, analysisId, enabled, jobIndex, JSON.stringify(roiParameters), dispatch]);

    return {
    };
};

export { useRoiPicker };

