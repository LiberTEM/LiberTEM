import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import * as analysisActions from '../actions';
import { AnalysisState, PreviewMode } from "../types";
const previewModeOptions = [
    {
        text: "Average",
        value: "AVERAGE",
    },
    {
        text: "Pick",
        value: "PICK",
    }
]

const mapDispatchToProps = {
    setPreviewMode: analysisActions.Actions.setPreviewMode,
}

interface PMSProps {
    analysis: AnalysisState,
}

type MergedProps = PMSProps & DispatchProps<typeof mapDispatchToProps>;

class PreviewModeSelector extends React.Component<MergedProps> {
    public handleChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        const value = data.value as PreviewMode;
        const { analysis } = this.props;
        this.props.setPreviewMode(analysis.id, value);
    }

    public render() {
        const { analysis } = this.props;

        return (
            <>
                <div>
                    Mode:{' '}
                    <Dropdown
                        inline={true}
                        options={previewModeOptions}
                        value={analysis.preview.mode}
                        onChange={this.handleChange}
                    />
                </div>
            </>
        )
    }

}

export default connect(null, mapDispatchToProps)(PreviewModeSelector);