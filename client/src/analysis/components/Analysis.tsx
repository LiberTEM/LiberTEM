import * as React from "react";
import { useSelector } from "react-redux";
import { DatasetStatus } from "../../messages";
import { RootReducer } from "../../store";
import { AnalysisMetadata, AnalysisState } from "../types";

interface AnalysisDispatcherProps {
    analysis: AnalysisState,
}

const AnalysisDispatcherComponent: React.SFC<AnalysisDispatcherProps> = ({ analysis }) => {
    const dataset = useSelector((state: RootReducer) => state.datasets.byId[analysis.dataset])

    if (dataset.status !== DatasetStatus.OPEN) {
        return null;
    }

    const AnalysisComponent = AnalysisMetadata[analysis.mainAnalysisType].component;
    if (!AnalysisComponent) {
        throw new Error("unknown analysis type");
    }

    return <AnalysisComponent dataset={dataset} analysis={analysis} />;
}

export default AnalysisDispatcherComponent;