import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { RootReducer } from "../../store";
import { JobState } from "../types";
import Result from "./Result";

interface ResultProps {
    width: number,
    height: number,
}

interface ExternalResultProps {
    job: string,
    analysis: string,
}

const mapStateToProps = (state: RootReducer, ownProps: ExternalResultProps) => {
    const job = ownProps.job ? state.jobs.byId[ownProps.job] : undefined;
    const ds = (job !== undefined) ? state.datasets.byId[job.dataset] : undefined;
    const analysis = state.analyses.byId[ownProps.analysis];

    return {
        job,
        analysis,
        dataset: ds,
    };
};

type MergedProps = ResultProps & ReturnType<typeof mapStateToProps>;

interface ResultState {
    selectedImg: number,
}

class ResultList extends React.Component<MergedProps, ResultState> {
    public state: ResultState = { selectedImg: 0 };

    public selectImage = (e: React.SyntheticEvent, data: DropdownProps) => {
        const value = data.value as number;
        this.setState({ selectedImg: value });
    }

    public render() {
        const { job, analysis, dataset, width, height } = this.props;
        let msg;
        let img = (
            <svg style={{ border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
        );
        if (!job || !dataset) {
            msg = <p>&nbsp;</p>;
        } else {
            if (job.results.length > 0) {
                img = (
                    <Result analysis={analysis} job={job} dataset={dataset} width={width} height={height} idx={this.state.selectedImg} />
                );
            }
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
