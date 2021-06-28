import * as React from "react";
import { CompoundAnalysisReducerState } from "../../reducers";
import Analysis from "./Analysis";

interface AnalysisProps {
    analyses: CompoundAnalysisReducerState,
}

const AnalysisList: React.FC<AnalysisProps> = ({ analyses }) => (<>{
    analyses.ids.map(analysisId => <Analysis key={analysisId} analysis={analyses.byId[analysisId]} />)
}</>);

export default AnalysisList;