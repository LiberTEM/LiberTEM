import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { AnalysisTypes } from "../../messages";
import { RootReducer } from "../../store";
import { JobRunning, JobState } from "../types";
import Result from "./Result";
import { JobKind } from "../../analysis/types";

interface ResultListProps {
    width: number,
    height: number,
    kind: JobKind,
    extraHandles?: React.ReactElement<SVGElement>,
}

interface ExternalResultListProps {
    analysis: string,
    kind: JobKind,
}

const mapStateToProps = (state: RootReducer, ownProps: ExternalResultListProps) => {
    const analysis = state.analyses.byId[ownProps.analysis];
    const jobId = analysis.jobs[ownProps.kind];
    const job = jobId ? state.jobs.byId[jobId] : undefined;
    const ds = job ? state.datasets.byId[job.dataset] : undefined;
    const pickCoords = analysis.frameDetails.type === AnalysisTypes.SUM_FRAMES ? null : <>Pick: x={analysis.frameDetails.parameters.x}, y={analysis.frameDetails.parameters.y} &emsp;</>;

    return {
        currentJob: job,
        jobsById: state.jobs.byId,
        analysis,
        dataset: ds,
        pickCoords,
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

    public getJob = () => {
        const { currentJob, analysis, jobsById, kind } = this.props;
        if (!currentJob) {
            return;
        }
        if (currentJob.results.length > 0) {
            return currentJob;
        }
        const history = analysis.jobHistory[kind];
        if (history.length > 0) {
            return jobsById[history[0]];
        }
        return;
    }

    public render() {
        const { analysis, dataset, children, width, height, pickCoords, extraHandles } = this.props;
        let msg;
        let img = (
            <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        );
        const job = this.getJob();
        if (!job || !dataset) {
            msg = <>&nbsp;</>;
        } else {
            img = (
                <Result analysis={analysis} job={job} dataset={dataset}
                    extraHandles={extraHandles}
                    width={width} height={height}
                    idx={this.state.selectedChannel}
                />
            );
            if (job.running === JobRunning.DONE) {
                const dt = (job.endTimestamp - job.startTimestamp) / 1000;
                msg = <>Analysis done in {dt.toFixed(3)} seconds</>;
            } else {
                msg = <>Analysis running...</>;
            }
        }
        return (
            <div>
                {img}
                {children}
                <ResultImageSelector job={job} handleChange={this.selectChannel} selectedImg={this.state.selectedChannel} />
                <p>{pickCoords} {msg}</p>
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
