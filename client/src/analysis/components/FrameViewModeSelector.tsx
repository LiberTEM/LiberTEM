import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { AnalysisTypes } from "../../messages";
import { RootReducer } from "../../store";
import * as analysisActions from '../actions';
import { AnalysisState, FrameMode } from "../types";

const frameViewModeOptions = [
    {
        text: "Average",
        value: AnalysisTypes.SUM_FRAMES,
    },
    {
        text: "Pick",
        value: AnalysisTypes.PICK_FRAME,
    }
]

const mapDispatchToProps = {
    setFrameViewMode: analysisActions.Actions.setFrameViewMode,
}

const mapStateToProps = (state: RootReducer, ownProps: PMSProps) => {
    const dataset = state.datasets.byId[ownProps.analysis.dataset]
    const shape = dataset.params.shape;
    const scanWidth = shape[1];
    const scanHeight = shape[0];
    return {
        scanWidth,
        scanHeight,
    }
}

interface PMSProps {
    analysis: AnalysisState,
}

type MergedProps = PMSProps & DispatchProps<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

class FrameViewModeSelector extends React.Component<MergedProps> {
    public handleChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        const value = data.value as FrameMode;
        const { analysis, scanWidth, scanHeight } = this.props;
        let initialParams = {};
        if (value === AnalysisTypes.PICK_FRAME) {
            initialParams = {
                x: Math.round(scanWidth / 2),
                y: Math.round(scanHeight / 2),
            }
        }
        this.props.setFrameViewMode(analysis.id, value, initialParams);
    }

    public render() {
        const { analysis } = this.props;

        return (
            <>
                <div>
                    Mode:{' '}
                    <Dropdown
                        inline={true}
                        options={frameViewModeOptions}
                        value={analysis.frameDetails.type}
                        onChange={this.handleChange}
                    />
                </div>
            </>
        )
    }

}

export default connect(mapStateToProps, mapDispatchToProps)(FrameViewModeSelector);