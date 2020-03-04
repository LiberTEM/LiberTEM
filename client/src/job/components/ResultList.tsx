import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { RootReducer } from "../../store";
import { HandleRenderFunction } from "../../widgets/types";
import { JobRunning, JobState } from "../types";
import Result from "./Result";
import Selectors from "./Selectors";

interface ResultListProps {
    width: number,
    height: number,
    selectors?: React.ReactElement<any>,
    extraHandles?: HandleRenderFunction,
    extraWidgets?: React.ReactElement<SVGElement>,
    subtitle?: React.ReactNode,
}

interface ExternalResultListProps {
    compoundAnalysis: string,
    analysisIndex: number,
}

const mapStateToProps = (state: RootReducer, ownProps: ExternalResultListProps) => {
    const compoundAnalysis = state.compoundAnalyses.byId[ownProps.compoundAnalysis];
    const analysis = state.analyses.byId[compoundAnalysis.analyses[ownProps.analysisIndex]];

    return {
        jobsById: state.jobs.byId,
        analysis,
        compoundAnalysis,
        analysisIndex: ownProps.analysisIndex,
    };
};

type MergedProps = ResultListProps & ReturnType<typeof mapStateToProps>;

interface ResultListState {
    selectedChannel: number,
}

class ResultList extends React.Component<MergedProps, ResultListState> {
    public state: ResultListState = { selectedChannel: 0 };

    public selectChannel = (e: React.SyntheticEvent, data: DropdownProps) => {
        const value = data.value as number;
        this.setState({ selectedChannel: value });
    }

    public render() {
        const {
            selectors, children, width, height,
            extraHandles, extraWidgets, subtitle,
            analysis, jobsById,
        } = this.props;
        let msg;
        let currentResult = (
            // placeholder:
            <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        );
        const job = jobsById[analysis?.displayedJob];
        if (!job || !analysis) {
            msg = <>&nbsp;</>;
        } else {
            currentResult = (
                <Result job={job}
                    extraHandles={extraHandles}
                    extraWidgets={extraWidgets}
                    width={width} height={height}
                    channel={this.state.selectedChannel}
                />
            );
            if (job.running === JobRunning.DONE) {
                const dt = (job.endTimestamp - job.startTimestamp) / 1000;
                msg = <>Analysis done in {dt.toFixed(3)}s</>;
            } else {
                msg = <>Analysis running...</>;
            }
        }
        return (
            <div>
                {currentResult}
                {children}
                <Selectors>
                    <ResultImageSelector job={job} handleChange={this.selectChannel} selectedImg={this.state.selectedChannel} />
                    {selectors}
                </Selectors>
                <p>{subtitle} {msg}</p>
            </div>
        );
    }
}

interface ImageSelectorProps {
    job?: JobState,
    handleChange: (e: React.SyntheticEvent, data: DropdownProps) => void,
    selectedImg: number,
}

const ResultImageSelector: React.SFC<ImageSelectorProps> = ({ job, handleChange, selectedImg }) => {
    if (!job) {
        return null;
    }
    const availableImages = job.results.map((result, idx) => ({ text: result.description.title, value: idx }));
    return (
        <>
            <div>
                Channel:{' '}
                <Dropdown
                    inline={true}
                    options={availableImages}
                    value={selectedImg}
                    onChange={handleChange}
                />
            </div>
        </>
    )
}


export default connect(mapStateToProps)(ResultList);
