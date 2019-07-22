import React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes } from "../../messages";
import * as analysisActions from "../actions";

const useSumFrames = ({
    enabled, jobIndex, analysisId,
}: {
    enabled: boolean, jobIndex: number, analysisId: string,
}) => {
    const dispatch = useDispatch();

    // FIXME: effect won't re-run when parameters change
    React.useEffect(() => {
        if (enabled) {
            dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                type: AnalysisTypes.SUM_FRAMES,
                parameters: {roi: {}},
            }));
        }
    }, [analysisId, enabled, jobIndex]);
};

export default useSumFrames;