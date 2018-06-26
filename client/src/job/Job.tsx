import * as React from "react";
import { connect } from "react-redux";
import { Image } from 'semantic-ui-react';
import { RootReducer } from "../store";
import { Job } from "./types";

interface JobProps {
    job: Job,
}

interface ExternalJobProps {
    job: string,
}

const JobComponent: React.SFC<JobProps> = ({ job }) => {
    return (
        <div>
            {job.results.map((res, idx) => {
                return <Image src={res.imageURL} key={idx} />
            })}
        </div>
    );
};

const mapStateToProps = (state: RootReducer, ownProps: ExternalJobProps) => {
    return {
        job: state.job.byId[ownProps.job],
    };
};

export default connect(mapStateToProps)(JobComponent);