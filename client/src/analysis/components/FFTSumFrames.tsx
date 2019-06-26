import React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes } from "../../messages";
import * as analysisActions from "../actions";

const useFFTSumFrames = ({
    enabled, jobIndex, analysisId,
}: {
    enabled: boolean, jobIndex: number, analysisId: string,
}) => {
    const dispatch = useDispatch();

    React.useEffect(() => {
        if (enabled) {
            dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                type: AnalysisTypes.FFTSUM_FRAMES,
                parameters: {},
            }));
        }
    }, [analysisId, enabled, jobIndex]);
};

export default useFFTSumFrames;
