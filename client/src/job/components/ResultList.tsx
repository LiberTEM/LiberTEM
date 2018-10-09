import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { RootReducer } from "../../store";
import { JobState } from "../types";
import Result from "./Result";

interface ResultListProps {
    width: number,
    height: number,
}

interface ExternalResultListProps {
    analysis: string,
}

const mapStateToProps = (state: RootReducer, ownProps: ExternalResultListProps) => {
    const analysis = state.analyses.byId[ownProps.analysis];
    const jobId = analysis.jobs.RESULT;
    const job = jobId ? state.jobs.byId[jobId] : undefined;
    const ds = job ? state.datasets.byId[job.dataset] : undefined;

    return {
        currentJob: job,
        jobsById: state.jobs.byId,
        analysis,
        dataset: ds,
    };
};

type MergedProps = ResultListProps & ReturnType<typeof mapStateToProps>;

interface ResultListState {
    selectedImg: number,
}

class ResultList extends React.Component<MergedProps, ResultListState> {
    public state: ResultListState = { selectedImg: 0 };

    public selectImage = (e: React.SyntheticEvent, data: DropdownProps) => {
        const value = data.value as number;
        this.setState({ selectedImg: value });
    }

    public getJob = () => {
        const { currentJob, analysis, jobsById } = this.props;
        if (!currentJob) {
            return;
        }
        if (currentJob.results.length > 0) {
            return currentJob;
        }
        const history = analysis.jobHistory.RESULT;
        if (history.length > 0) {
            return jobsById[history[0]];
        }
        return;
    }

    public render() {
        const { analysis, dataset, width, height } = this.props;
        let msg;
        let img = (
            <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        );
        const job = this.getJob();
        if (!job || !dataset) {
            msg = <p>&nbsp;</p>;
        } else {
            img = (
                <Result analysis={analysis} job={job} dataset={dataset} width={width} height={height} idx={this.state.selectedImg} />
            );
            if (job.startTimestamp && job.endTimestamp) {
                const dt = (job.endTimestamp - job.startTimestamp) / 1000;
                msg = <p>Analysis done in {dt.toFixed(3)} seconds</p>;
            } else {
                msg = <p>Analysis running...</p>;
            }
        }
        return (
            <div>
                {img}
                <ResultImageSelector job={job} handleChange={this.selectImage} selectedImg={this.state.selectedImg} />
                {msg}
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
                Image:{' '}
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
