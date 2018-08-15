import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Icon } from "semantic-ui-react";
import { DirectoryListingDetails } from "../../messages";
import * as browserActions from '../actions';

const mapDispatchToProps = (dispatch: Dispatch, ownProps: FileEntryProps) => {
    return {
        list: (e: React.MouseEvent) => {
            dispatch(browserActions.Actions.list(ownProps.path, ownProps.details.name));
            window.setTimeout(() => ownProps.onChange(), 0);
        },
    };
}

interface FileEntryProps {
    path: string,
    style: object,
    details: DirectoryListingDetails
    onChange: () => void
}

type MergedProps = FileEntryProps & ReturnType<typeof mapDispatchToProps>;

const FolderEntry: React.SFC<MergedProps> = ({ list, details, style, onChange }) => {
    const myStyle = {
        cursor: "pointer",
        ...style,
    }
    return (
        <div onClick={list} style={myStyle}><Icon name="folder" /> {details.name}</div>
    );
}

export default connect(null, mapDispatchToProps)(FolderEntry);
