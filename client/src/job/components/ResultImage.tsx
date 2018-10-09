import * as React from "react";
import { JobState } from "../types";

interface ResultImageProps {
    job: JobState,
    idx: number,
    width: number,
    height: number,
}

const ResultImage: React.SFC<ResultImageProps> = ({ job, idx, width, height }) => {
    const style: React.CSSProperties = {
        width: "100%",
        height: "auto",
        imageRendering: "pixelated"
    };
    const result = job.results[idx];
    if (result === undefined) {
        return (
            <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        )
    }
    return (
        <image style={style} xlinkHref={result.imageURL} width={width} height={height} />
    );
}

export default ResultImage;