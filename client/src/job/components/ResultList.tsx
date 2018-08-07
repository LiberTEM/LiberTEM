import * as React from "react";
import { connect } from "react-redux";
import { RootReducer } from "../../store";
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
    const job = ownProps.job ? state.job.byId[ownProps.job] : undefined;
    const ds = (job !== undefined) ? state.dataset.byId[job.dataset] : undefined;
    const analysis = state.analyses.byId[ownProps.analysis];

    return {
        job,
        analysis,
        dataset: ds,
    };
};

type MergedProps = ResultProps & ReturnType<typeof mapStateToProps>;

const ResultList: React.SFC<MergedProps> = ({ job, analysis, dataset, width, height }) => {
    let msg;
    let imgs = [
        <svg style={{ border: "1px solid black", width: "100%", height: "auto" }} width={width} height={height} viewBox={`0 0 ${width} ${height}`} key={-1} />
    ];
    if (!job || !dataset) {
        msg = <p>&nbsp;</p>;
    } else {
        if (job.results.length > 0) {
            imgs = (job.results.map((res, idx) => {
                return (
                    <Result analysis={analysis} job={job} dataset={dataset} width={width} height={height} idx={idx} key={idx} />
                );
            }))
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
            {imgs}
            {msg}
        </div>
    );
};

export default connect(mapStateToProps)(ResultList);
