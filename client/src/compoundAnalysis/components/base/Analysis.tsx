import * as React from "react";
import { useSelector } from "react-redux";
import { DatasetStatus } from "../../../messages";
import { RootReducer } from "../../../store";
import { CompoundAnalysisMetadata, CompoundAnalysisState } from "../../types";

interface AnalysisDispatcherProps {
    analysis: CompoundAnalysisState,
}

const AnalysisDispatcherComponent: React.FC<AnalysisDispatcherProps> = ({ analysis }) => {
    const dataset = useSelector((state: RootReducer) => state.datasets.byId[analysis.dataset])

    if (dataset.status !== DatasetStatus.OPEN) {
        return null;
    }

    const AnalysisComponent = CompoundAnalysisMetadata[analysis.details.mainType].component;
    if (!AnalysisComponent) {
        throw new Error("unknown analysis type");
    }

    return <AnalysisComponent dataset={dataset} compoundAnalysis={analysis} />;
}

export default AnalysisDispatcherComponent;