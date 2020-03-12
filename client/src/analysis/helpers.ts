import { useSelector } from 'react-redux';
import { CompoundAnalysisState } from '../compoundAnalysis/types';
import { RootReducer } from '../store';

export const useAnalysis = (compoundAnalysisId: string, analysisIdx: number) => {
    const compoundAnalysis = useSelector((state: RootReducer) => state.compoundAnalyses.byId[compoundAnalysisId])
    // tslint:disable-next-line:no-console
    console.log(compoundAnalysis);
    const analysisId = compoundAnalysis.details.analyses[analysisIdx];
    return useSelector((state: RootReducer) => state.analyses.byId[analysisId]);
}

export const useAutoStart = (compoundAnalysis: CompoundAnalysisState, analysisIdx: number) => {
    const analysisId = compoundAnalysis.details.analyses[analysisIdx];
    const analysis = useSelector((state: RootReducer) => state.analyses.byId[analysisId]);
    return analysis ? analysis.doAutoStart : false;
}