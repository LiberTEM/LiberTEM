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
    analysis: string,
    jobIndex: number,
}

const mapStateToProps = (state: RootReducer, ownProps: ExternalResultListProps) => {
    const analysis = state.analyses.byId[ownProps.analysis];
    const jobId = analysis.jobs[ownProps.jobIndex];
    const job = jobId ? state.jobs.byId[jobId] : undefined;
    const ds = job ? state.datasets.byId[job.dataset] : undefined;

    return {
        currentJob: job,
        jobsById: state.jobs.byId,
        analysis,
        dataset: ds,
        jobIndex: ownProps.jobIndex,
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
        const { currentJob, analysis, jobsById, jobIndex } = this.props;
        if (!currentJob) {
            return;
        }
        if (currentJob.results.length > 0) {
            return currentJob;
        }
        const history = analysis.jobHistory[jobIndex];

        if (history === undefined) {
            return;
        }
        for (const tmpJobId of history) {
            const tmpJob = jobsById[tmpJobId];
            if (tmpJob.results.length > 0) {
                return tmpJob;
            }
        }
        return;
    }

    public render() {
        const {
            selectors, analysis, dataset, children, width, height, jobIndex,
            extraHandles, extraWidgets, subtitle,
        } = this.props;
        let msg;
        let currentResult = (
            // placeholder:
            <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        );
        const job = this.getJob();
        if (!job || !dataset) {
            msg = <>&nbsp;</>;
        } else {
            currentResult = (
                <Result analysis={analysis} job={job} dataset={dataset}
                    extraHandles={extraHandles}
                    extraWidgets={extraWidgets}
                    width={width} height={height}
                    jobIndex={jobIndex}
                    idx={this.state.selectedChannel}
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
