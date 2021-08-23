import { AnalysisReducerState } from "../analysis/reducers";
import { JobReducerState } from "../job/reducers";
import { JobRunning } from "../job/types";
import { CompoundAnalysisState } from "./types";

export const getAnalysisStatus = (
    compoundAnalysis: CompoundAnalysisState,
    analyses: AnalysisReducerState,
    jobs: JobReducerState,
    analysisIdxsToInclude: number[] = []
): "idle" | "busy" => {
    let filteredAnalyses = compoundAnalysis.details.analyses;

    if (analysisIdxsToInclude.length > 0) {
        filteredAnalyses = filteredAnalyses.filter((analysisId: string, idx: number) => analysisIdxsToInclude.indexOf(idx) !== -1)
    }

    return filteredAnalyses.reduce((prevValue: "idle" | "busy", analysisId: string) => {
        const analysis = analyses.byId[analysisId];
        if(!analysis) {
            return prevValue; // no analysis, so "all jobs" are done
        }
        analysis.jobs.forEach((jobId) => {
            if (!jobs.byId[jobId]) {
                // eslint-disable-next-line no-console
                console.error(`could not find job id ${jobId} for analysis ${analysisId}`);
            }
        });
        const allDone = analysis.jobs.every(
            jobId => jobs.byId[jobId] ? jobs.byId[jobId].running === JobRunning.DONE : true
        );
        return allDone ? prevValue : "busy";
    }, "idle");
}


/**
 * Check if there is a finished job that is being displayed
 * 
 * @param compoundAnalysis 
 * @param analyses 
 * @param jobs 
 * @param analysisIdxsToInclude 
 * @returns true iff all displayedJobs of the given analyses are DONE (also false if there are no analyses or displayed jobs)
 */
export const haveDisplayResult = (
    compoundAnalysis: CompoundAnalysisState,
    analyses: AnalysisReducerState,
    jobs: JobReducerState,
    analysisIdxsToInclude: number[] = []
): boolean => {
    let filteredAnalyses = compoundAnalysis.details.analyses;

    if (analysisIdxsToInclude.length > 0) {
        filteredAnalyses = filteredAnalyses.filter((analysisId: string, idx: number) => analysisIdxsToInclude.indexOf(idx) !== -1)
    }

    if (filteredAnalyses.length === 0) {
        return false;
    }

    return filteredAnalyses.reduce((prevValue: boolean, analysisId: string) => {
        const analysis = analyses.byId[analysisId];
        if (!analysis) {
            return false; // no analysis, so we don't have a result
        }
        if (!analysis.displayedJob) {
            return false;
        }
        const displayedJob = jobs.byId[analysis.displayedJob];
        if (!displayedJob) {
            return false;
        }

        return displayedJob.running === JobRunning.DONE && prevValue;
    }, true);
}
