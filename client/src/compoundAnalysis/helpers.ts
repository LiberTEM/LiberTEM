import { AnalysisReducerState } from "../analysis/reducers";
import { JobReducerState } from "../job/reducers";
import { JobRunning } from "../job/types";
import { CompoundAnalysisState } from "./types";

export const getAnalysisStatus = (compoundAnalysis: CompoundAnalysisState, analyses: AnalysisReducerState, jobs: JobReducerState, analysisIdxsToInclude: number[] = []): "idle" | "busy" => {
    let filteredAnalyses = compoundAnalysis.details.analyses;

    if (analysisIdxsToInclude.length > 0) {
        filteredAnalyses = filteredAnalyses.filter((analysisId: string, idx: number) => {
            return analysisIdxsToInclude.indexOf(idx) !== -1;
        })
    }

    return filteredAnalyses.reduce((prevValue: "idle" | "busy", analysisId: string) => {
        const analysis = analyses.byId[analysisId];
        if(!analysis) {
            return prevValue; // no analysis, so "all jobs" are done
        }
        const allDone = analysis.jobs.every(
            jobId => jobs.byId[jobId].running === JobRunning.DONE
        );
        return allDone ? prevValue : "busy";
    }, "idle");
}