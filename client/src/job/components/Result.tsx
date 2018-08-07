import * as React from "react";
import { connect } from "react-redux";
import * as analysisActions from '../../analysis/actions';
import { AnalysisState } from "../../analysis/types";
import { DatasetState } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import HandleParent from "../../widgets/HandleParent";
import { JobState } from "../types";

interface ResultProps {
    width: number,
    height: number,
    job: JobState,
    dataset: DatasetState,
    analysis: AnalysisState,
    idx: number,
}


const mapDispatchToProps = {
    setPreview: analysisActions.Actions.setPreview
};

type MergedProps = ResultProps & DispatchProps<typeof mapDispatchToProps>;

class Result extends React.Component<MergedProps> {
    public onCenterChange = (x: number, y: number) => {
        this.props.setPreview(this.props.analysis.id, {
            mode: "PICK",
            pick: {
                x: Math.round(x),
                y: Math.round(y)
            }
        })
    }

    public render() {
        const { analysis, job, idx, width, height } = this.props;
        const { x, y } = analysis.preview.pick;
        const result = job.results[idx];

        return (
            <svg style={{ border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
                <image style={{ width: "100%", height: "auto", imageRendering: "pixelated" }} xlinkHref={result.imageURL} width={width} height={height} />
                {analysis.preview.mode === "PICK" ?
                    <HandleParent width={width} height={height}>
                        <DraggableHandle x={x} y={y} withCross={true}
                            onDragMove={this.onCenterChange}
                            constraint={inRectConstraint(width, height)} />
                    </HandleParent> : null}
            </svg>
        );
    }
};

export default connect(null, mapDispatchToProps)(Result);