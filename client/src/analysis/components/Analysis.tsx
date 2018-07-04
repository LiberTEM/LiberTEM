import * as React from "react";
import { connect } from "react-redux";
import { assertNotReached } from '../../helpers';
import { AnalysisTypes } from "../../messages";
import { RootReducer } from "../../store";
import { AnalysisState } from "../types";
import CenterOfMassAnalysis from "./CenterOfMassAnalysis";
import DiskMaskAnalysis from "./DiskMaskAnalysis";
import RingMaskAnalysis from "./RingMaskAnalysis";

interface AnalysisProps {
    analysis: AnalysisState,
}

const mapStateToProps = (state: RootReducer, ownProps: AnalysisProps) => {
    return {
        dataset: state.dataset.byId[ownProps.analysis.dataset],
    }
}

type MergedProps = AnalysisProps & ReturnType<typeof mapStateToProps>;

const AnalysisComponent: React.SFC<MergedProps> = ({ analysis, dataset }) => {
    switch (analysis.details.type) {
        case AnalysisTypes.APPLY_DISK_MASK: {
            return <DiskMaskAnalysis dataset={dataset} analysis={analysis} parameters={analysis.details.parameters} />;
        };
        case AnalysisTypes.APPLY_RING_MASK: {
            return <RingMaskAnalysis dataset={dataset} analysis={analysis} parameters={analysis.details.parameters} />;
        }
        case AnalysisTypes.CENTER_OF_MASS: {
            return <CenterOfMassAnalysis dataset={dataset} analysis={analysis} parameters={analysis.details.parameters} />;
        }
    }

    return assertNotReached("unknown analysis type");
}

export default connect(mapStateToProps)(AnalysisComponent);