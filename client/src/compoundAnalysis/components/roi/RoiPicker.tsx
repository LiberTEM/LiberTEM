import * as React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes, FrameParams } from "../../../messages";
import * as analysisActions from "../../actions";


const useRoiPicker = ({ compoundAnalysisId, enabled, analysisIndex, roiParameters, analysisType }: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    analysisIndex: number,
    compoundAnalysisId: string;
    roiParameters: FrameParams;
    analysisType: AnalysisTypes.SD_FRAMES | AnalysisTypes.SUM_FRAMES
}) => {
    const dispatch = useDispatch();

    React.useEffect(() => {
        const handle = setTimeout(() => {
            if (enabled) {
                const analysisDetails = {
                    analysisType,
                    parameters: roiParameters,
                };
                dispatch(analysisActions.Actions.run(compoundAnalysisId, analysisIndex, analysisDetails))
            }
        }, 100);

        return () => clearTimeout(handle);
        // rules-of-hooks can't be statically validated here
        // eslint-disable-next-line
    }, [analysisType, compoundAnalysisId, enabled, analysisIndex, JSON.stringify(roiParameters), dispatch]);

    return {
    };
};

export { useRoiPicker };

