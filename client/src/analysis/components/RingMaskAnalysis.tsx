import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { defaultDebounce } from "../../helpers";
import { DatasetState, MaskDefRing } from "../../messages";
import Ring from "../../widgets/Ring";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";
import Preview from "./Preview";

interface AnalysisProps {
    parameters: MaskDefRing,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            // FIXME: updateParameters doesn't seem to be typed strong enough
            // the following doesn't raise a type error:
            // dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { foo: "bar" }));
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }));
        }),
        handleRIChange: defaultDebounce((ri: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ri }));
        }),
        handleROChange: defaultDebounce((ro: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ro }));
        }),
    }
}

type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const RingMaskAnalysis: React.SFC<MergedProps> = ({ analysis, dataset, parameters, handleCenterChange, handleRIChange, handleROChange }) => {
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const image = <Preview dataset={dataset} analysis={analysis} />

    return (
        <AnalysisItem analysis={analysis} dataset={dataset} title="Ring analysis" subtitle={
            <>Ring: center=({parameters.cx.toFixed(2)},{parameters.cy.toFixed(2)}), ri={parameters.ri.toFixed(2)}, ro={parameters.ro.toFixed(2)}</>
        }>
            <Ring cx={parameters.cx} cy={parameters.cy} ri={parameters.ri} ro={parameters.ro}
                imageWidth={imageWidth} imageHeight={imageHeight} image={image}
                onCenterChange={handleCenterChange} onRIChange={handleRIChange} onROChange={handleROChange} />
        </AnalysisItem>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(RingMaskAnalysis);