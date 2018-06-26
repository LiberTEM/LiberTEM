import * as React from "react";
import { assertNotReached } from '../../helpers';
import { Analysis, AnalysisTypes } from "../types";
import DiskMaskAnalysis from "./DiskMaskAnalysis";
import RingMaskAnalysis from "./RingMaskAnalysis";

interface AnalysisProps {
    analysis: Analysis,
}

const AnalysisComponent: React.SFC<AnalysisProps> = ({ analysis }) => {
    switch (analysis.details.type) {
        case AnalysisTypes.APPLY_DISK_MASK: {
            return <DiskMaskAnalysis analysis={analysis} parameters={analysis.details.parameters} />;
        };
        case AnalysisTypes.APPLY_RING_MASK: {
            return <RingMaskAnalysis analysis={analysis} parameters={analysis.details.parameters} />;
        }
    }

    return assertNotReached("unknown analysis type");
}

export default AnalysisComponent;