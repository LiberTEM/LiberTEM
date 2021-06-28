import * as React from "react";
import { JobState } from "../types";
import styles from "./ResultImage.module.css";

interface ResultImageProps {
    job: JobState,
    channel: number,
    width: number,
    height: number,
}

const ResultImage: React.FC<ResultImageProps> = ({ job, channel, width, height }) => {
    const result = job.results[channel];
    if (result === undefined) {
        return (
            <svg className={styles.fallback} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        )
    }
    return (
        <image className={styles.default} xlinkHref={result.imageURL} width={width} height={height} />
    );
}

export default ResultImage;