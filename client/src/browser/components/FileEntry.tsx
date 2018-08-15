import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Icon } from "semantic-ui-react";
import { DirectoryListingDetails } from "../../messages";
import * as browserActions from '../actions';

interface FileEntryProps {
    path: string,
    style: object,
    details: DirectoryListingDetails
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: FileEntryProps) => {
    return {
        select: () => dispatch(browserActions.Actions.select(ownProps.path, ownProps.details.name)),
    };
}

type MergedProps = FileEntryProps & ReturnType<typeof mapDispatchToProps>;

class FileEntry extends React.Component<MergedProps> {
    public onClick = (e: React.MouseEvent) => {
        this.props.select();
    }

    public render() {
        const { details, style } = this.props;
        const myStyle = {
            cursor: "pointer",
            ...style,
        }

        return (
            <div onClick={this.onClick} style={myStyle}><Icon name="file outline" />{details.name}</div>
        );
    }
}

export default connect(null, mapDispatchToProps)(FileEntry);