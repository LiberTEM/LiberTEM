import * as React from "react";
import { connect } from "react-redux";
import ResultImage from "../../job/components/ResultImage";
import { JobReducerState } from "../../job/reducers";
import { DatasetState } from "../../messages";
import { RootReducer } from "../../store";
import { AnalysisState } from "../types";

export interface FrameViewProps {
    analysis: AnalysisState,
    dataset: DatasetState,
}

type MergedProps = FrameViewProps & ReturnType<typeof mapStateToProps>;

const FrameView: React.SFC<MergedProps> = ({ analysis, dataset, job }) => {
    const { shape } = dataset.params;

    const imageWidth = shape[3];
    const imageHeight = shape[2];

    if (job === undefined) {
        return null;
    }

    return (
        <ResultImage job={job} idx={0} width={imageWidth} height={imageHeight} />
    );
}

const getJob = (analysis: AnalysisState, jobs: JobReducerState) => {
    const jobId = analysis.jobs.FRAME;
    if (jobId === undefined) {
        return;
    }
    return jobs.byId[jobId];
}

const mapStateToProps = (state: RootReducer, ownProps: FrameViewProps) => {
    return {
        job: getJob(ownProps.analysis, state.jobs),
    }
}

export default connect(mapStateToProps)(FrameView);