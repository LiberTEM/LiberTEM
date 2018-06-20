import * as React from "react";
import { connect, Dispatch } from "react-redux";
import Job from "./Job";
import { JobReducerState } from "./reducers";

interface JobsProps {
    jobs: JobReducerState
}

const JobsComponent : React.SFC<JobsProps> = ({ jobs }) => {
    return <>{jobs.ids.map(jobId => <Job job={jobs[jobId]} key={jobId} />)}</>;
} 

// FIXME: we need a type for state here!
// maybe can be inferred from the return type of the root reducer?
const mapStateToProps = (state: any) => {
    return {
        jobs: state.job,
    }
}

const mapDispatchToProps = (dispatch: Dispatch) => {
    return {}; // TODO:
}

export default connect(mapStateToProps, mapDispatchToProps)(JobsComponent);