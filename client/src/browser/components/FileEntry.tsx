import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { DirectoryListingDetails } from "../../messages";
import * as browserActions from '../actions';
import FileBrowserEntry from "./FileBrowserEntry";

interface FileEntryProps {
    path: string,
    style: Record<string, unknown>,
    details: DirectoryListingDetails,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: FileEntryProps) => ({
    select: () => dispatch(browserActions.Actions.select(ownProps.path, ownProps.details.name)),
})

type MergedProps = FileEntryProps & ReturnType<typeof mapDispatchToProps>;

class FileEntry extends React.Component<MergedProps> {
    public render() {
        const { details, style, select } = this.props;

        return (
            <FileBrowserEntry onClick={select} style={style} details={details}
                icon="file outline" />
        )
    }
}

export default connect(null, mapDispatchToProps)(FileEntry);