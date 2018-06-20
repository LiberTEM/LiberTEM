import * as React from "react";
import { Job } from "./types";

interface JobProps {
    job: Job,
}

const JobComponent: React.SFC<JobProps> = ({ job }) => {
    // tslint:disable-next-line
    console.log(job);
    return (
        <ul>
            {job.results.map((res, idx) => {
                return <li key={idx}><img src={res.imageURL}/></li>
            })}
        </ul>
    );
}

export default JobComponent;