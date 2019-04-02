import * as React from "react";
import { connect } from "react-redux";
import styled from 'styled-components';
import * as analysisActions from '../../analysis/actions';
import { AnalysisState, JobKind } from "../../analysis/types";
import { DatasetState } from "../../messages";
import BusyWrapper from "../../widgets/BusyWrapper";
import HandleParent from "../../widgets/HandleParent";
import { HandleRenderFunction } from "../../widgets/types";
import { JobRunning, JobState } from "../types";
import ResultImage from "./ResultImage";

interface ResultProps {
    width: number,
    height: number,
    job: JobState,
    dataset: DatasetState,
    analysis: AnalysisState,
    extraHandles?: HandleRenderFunction,
    extraWidgets?: React.ReactElement<SVGElement>,
    idx: number,
    kind: JobKind,
}

const ResultWrapper = styled.svg`
    display: block;
    border: 1px solid black;
    width: 100%;
    height: auto;
`;

const mapDispatchToProps = {
    updateParameters: analysisActions.Actions.updateParameters,
};

type MergedProps = ResultProps & DispatchProps<typeof mapDispatchToProps>;

class Result extends React.Component<MergedProps> {
    public renderHandles() {
        const { width, height, extraHandles } = this.props;
        let handles: HandleRenderFunction[] = [];
        if (extraHandles) {
            handles = [...handles, extraHandles];
        }

        return (
            <HandleParent width={width} height={height} handles={handles} />
        )
    }

    public render() {
        const { job, idx, width, height, extraWidgets } = this.props;
        const busy = job.running !== JobRunning.DONE;

        return (
            <BusyWrapper busy={busy}>
                <ResultWrapper width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
                    <ResultImage job={job} idx={idx} width={width} height={height} />
                    {extraWidgets}
                    {this.renderHandles()}
                </ResultWrapper>
            </BusyWrapper>
        );
    }
};

export default connect(null, mapDispatchToProps)(Result);