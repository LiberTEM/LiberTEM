import * as _ from "lodash";
import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Button, Segment } from "semantic-ui-react";
import { MaskDefRing } from "../../job/api";
import JobComponent from "../../job/Job";
import Ring from "../../widgets/Ring";
import * as analysisActions from "../actions";
import { Analysis } from "../types";

interface AnalysisProps {
    parameters: MaskDefRing,
    analysis: Analysis,
}

function defaultDebounce<T extends (...args: any[]) => any>(fn: T, delay: number = 50) {
    return _.debounce(fn, delay, { maxWait: delay });
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }));
        }),
        handleRIChange: defaultDebounce((ri: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ri }));
        }),
        handleROChange: defaultDebounce((ro: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ro }));
        }),
        handleApply: () => dispatch(analysisActions.Actions.run(ownProps.analysis.id))
    }
}

type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const RingMaskAnalysis: React.SFC<MergedProps> = ({ analysis, parameters, handleCenterChange, handleRIChange, handleROChange, handleApply }) => {
    const { currentJob } = analysis;

    return (
        <Segment>
            <p>RingMaskAnalysis {analysis.id}</p>
            <Ring cx={parameters.cx} cy={parameters.cy} ri={parameters.ri} ro={parameters.ro}
                imageWidth={128} imageHeight={128} onCenterChange={handleCenterChange} onRIChange={handleRIChange} onROChange={handleROChange} />
            <Button primary={true} onClick={handleApply}>Apply</Button>
            {currentJob !== "" ? <JobComponent job={currentJob} /> : null}
        </Segment>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(RingMaskAnalysis);