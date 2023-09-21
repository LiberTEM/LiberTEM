import React from "react";


const Circle: React.FC<{ arc: number }> = ({ arc }) => {
    const r = 70;
    const circumference = 2 * Math.PI * r;
    const offset = ((1 - arc) * circumference);
    const color = "black";
    return (
        <circle
            r={r}
            cx={100}
            cy={100}
            fill="transparent"
            stroke={offset !== circumference ? color : ""}
            strokeWidth="1rem"
            strokeDasharray={circumference}
            strokeDashoffset={arc ? offset : 0}
        ></circle>
    );
};

const ProgressPie: React.FC<{ progress: number }> = ({ progress }) => {
    return (
        <svg width="1em" height="1em" viewBox="0 0 200 200">
            <g transform={`rotate(-90 ${"100 100"})`}>
                <Circle arc={progress} />
            </g>
        </svg>
    )
}


export default ProgressPie;