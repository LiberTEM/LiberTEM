import * as React from "react";
import { connect } from "react-redux";
import { Button, Segment } from "semantic-ui-react";
import { MaskDefDisk } from "../../job/api";
import JobComponent from "../../job/Job";
import * as analysisActions from "../actions";
import { Analysis } from "../types";

interface AnalysisProps {
    parameters: MaskDefDisk,
    analysis: Analysis,
}

const mapDispatchToProps = {
    runAnalysis: analysisActions.Actions.run,
}

type MergedProps = AnalysisProps & DispatchProps<typeof mapDispatchToProps>

const DiskMaskAnalysis: React.SFC<MergedProps> = ({ analysis, parameters, runAnalysis }) => {
    const { id, currentJob } = analysis;
    const onClick = () => runAnalysis(id);

    return (
        <Segment>
            DiskMaskAnalysis {analysis.id}
            <Button primary={true} onClick={onClick}>Apply</Button>
            {currentJob !== "" ? <JobComponent job={currentJob} /> : null}
        </Segment>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(DiskMaskAnalysis);