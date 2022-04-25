import { AnalysisDetails } from '../messages';

export type JobList = string[];

export interface Analysis {
    id: string,
    dataset: string,
    displayedJob?: string,
    details: AnalysisDetails,
    jobs: JobList,
}

export type AnalysisState = Analysis;
