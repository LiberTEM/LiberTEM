import React from "react";
import { AnalysisTypes } from "../../messages";
import * as analysisActions from "../actions";

const useSumFrames = ({
    enabled, jobIndex, analysisId, run
}: {
    enabled: boolean, jobIndex: number, analysisId: string,
    run: typeof analysisActions.Actions.run
}) => {
    React.useEffect(() => {
        if (enabled) {
            run(analysisId, jobIndex, {
                type: AnalysisTypes.SUM_FRAMES,
                parameters: {},
            });
        }
    }, [analysisId, enabled, jobIndex]);
};

export default useSumFrames;