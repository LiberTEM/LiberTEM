import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { RootReducer } from '../store';
import Job from "./Job";
import { JobReducerState } from "./reducers";

interface JobListProps {
    jobs: JobReducerState
}

const JobList: React.SFC<JobListProps> = ({ jobs }) => {
    return <>{jobs.ids.map(jobId => <Job job={jobs.byId[jobId]} key={jobId} />)}</>;
}

const mapStateToProps = (state: RootReducer) => {
    return {
        jobs: state.job,
    }
}

const mapDispatchToProps = (dispatch: Dispatch) => {
    return {}; // TODO:
}

export default connect(mapStateToProps, mapDispatchToProps)(JobList);