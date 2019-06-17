import { JobReducerState } from "../job/reducers";
import { JobRunning } from "../job/types";
import { AnalysisState } from "./types";

export const getAnalysisStatus = (analysis: AnalysisState, jobs: JobReducerState, jobIdxsToInclude: number[] = []): "idle" | "busy" => {
    let filteredJobs = analysis.jobs;

    if (jobIdxsToInclude.length > 0) {
        filteredJobs = analysis.jobs.filter((jobId: string, idx: number) => {
            return jobIdxsToInclude.indexOf(idx) !== -1;
        })
    }

    return filteredJobs.reduce((prevValue: "idle" | "busy", jobId: string) => {
        const isDone = jobs.byId[jobId].running === JobRunning.DONE;
        return isDone ? prevValue : "busy";
    }, "idle");
}