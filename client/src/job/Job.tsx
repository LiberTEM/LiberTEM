import * as React from "react";
import { Image } from 'semantic-ui-react';
import { DeepReadonly } from "utility-types";
import { Job } from "./types";

interface JobProps {
    job: DeepReadonly<Job>,
}

const JobComponent: React.SFC<JobProps> = ({ job }) => {
    return (
        <div>
            {job.results.map((res, idx) => {
                return <Image src={res.imageURL} key={idx} />
            })}
        </div>
    );
}

export default JobComponent;