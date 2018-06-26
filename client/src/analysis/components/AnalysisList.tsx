import * as React from "react";
import { AnalysisReducerState } from "../reducers";
import Analysis from "./Analysis";

interface AnalysisProps {
    analyses: AnalysisReducerState,
}

const AnalysisList: React.SFC<AnalysisProps> = ({ analyses }) => {
    return (<>{
        analyses.ids.map(analysisId => <Analysis key={analysisId} analysis={analyses.byId[analysisId]} />)
    }</>);
}

export default AnalysisList;