import * as React from "react";
import { connect } from "react-redux";
import { RootReducer } from "../../store";
import PlaceholderImage from "./PlaceholderImage";
import Result from "./Result";

interface ResultProps {
    width: number,
    height: number,
}

interface ExternalResultProps {
    job: string,
}

const mapStateToProps = (state: RootReducer, ownProps: ExternalResultProps) => {
    return {
        job: ownProps.job ? state.job.byId[ownProps.job] : undefined,
    };
};

type MergedProps = ResultProps & ReturnType<typeof mapStateToProps>;

const ResultList: React.SFC<MergedProps> = ({ job, width, height }) => {
    let msg;
    let imgs = [
        <PlaceholderImage width={width} height={height} key={-1} />
    ];
    if (!job) {
        msg = <p>&nbsp;</p>;
    } else {
        if (job.results.length > 0) {
            imgs = (job.results.map((res, idx) => {
                return (
                    <Result job={job} width={width} height={height} idx={idx} key={idx} />
                );
            }))
        }
        if (job.startTimestamp && job.endTimestamp) {
            const dt = (job.endTimestamp - job.startTimestamp) / 1000;
            msg = <p>Analysis done in {dt} seconds</p>;
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
