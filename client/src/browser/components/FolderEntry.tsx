import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { DirectoryListingDetails } from "../../messages";
import * as browserActions from '../actions';
import FileBrowserEntry from "./FileBrowserEntry";

const mapDispatchToProps = (dispatch: Dispatch, ownProps: FolderEntryProps) => {
    return {
        list: () => {
            dispatch(browserActions.Actions.list(ownProps.path, ownProps.details.name));
            window.setTimeout(() => ownProps.onChange(), 0);
        },
    };
}

interface FolderEntryProps {
    path: string,
    style: object,
    details: DirectoryListingDetails,
    onChange: () => void,
}

type MergedProps = FolderEntryProps & ReturnType<typeof mapDispatchToProps>;

const FolderEntry: React.SFC<MergedProps> = ({ list, details, style, onChange }) => {
    return (
        <FileBrowserEntry onClick={list} style={style} details={details} icon="folder" />
    )
}

export default connect(null, mapDispatchToProps)(FolderEntry);
