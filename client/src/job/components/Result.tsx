import * as React from "react";
import { Image } from 'semantic-ui-react';
import { JobState } from "../types";
import PlaceholderImage from "./PlaceholderImage";

interface ResultProps {
    width: number,
    height: number,
    job: JobState,
    idx: number,
}

const Result: React.SFC<ResultProps> = ({ job, idx, width, height }) => {
    const result = job.results[idx];
    return (
        <PlaceholderImage width={width} height={height}>
            <Image style={{ width: "100%", height: "auto", imageRendering: "pixelated" }} src={result.imageURL} width={width} height={height} />
        </PlaceholderImage>
    );
};

export default Result;