import React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes } from "../../messages";
import * as analysisActions from "../actions";

const useFFTSumFrames = ({
    enabled, analysisIndex: jobIndex, compoundAnalysisId: analysisId, real_rad, real_centerx, real_centery
}: {
    enabled: boolean, analysisIndex: number, compoundAnalysisId: string, real_rad: number | null, real_centerx: number | null, real_centery: number | null
}) => {
    const dispatch = useDispatch();

    React.useEffect(() => {
        if (enabled) {
            dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                analysisType: AnalysisTypes.FFTSUM_FRAMES,
                parameters: { real_rad, real_centerx, real_centery },
            }));
        }
    }, [analysisId, enabled, jobIndex, real_rad, real_centerx, real_centery, dispatch]);
};

export default useFFTSumFrames;
