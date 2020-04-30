import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { DirectoryListingDetails } from "../../messages";
import * as browserActions from '../actions';
import FileBrowserEntry from "./FileBrowserEntry";

interface FileEntryProps {
    path: string,
    style: object,
    isOpenStack: boolean,
    details: DirectoryListingDetails,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: FileEntryProps) => {
    return {
        select: () => dispatch(browserActions.Actions.select(ownProps.path, ownProps.details.name)),
        toggleFile: () => dispatch(browserActions.Actions.toggleFile(ownProps.details.index)),
    };
}

type MergedProps = FileEntryProps & ReturnType<typeof mapDispatchToProps>;

class FileEntry extends React.Component<MergedProps> {
    public render() {
        const { details, style, select, toggleFile, isOpenStack } = this.props;

        return (
            <FileBrowserEntry onClick={select} style={style} details={details}
                onToggleChange={toggleFile} icon="file outline" isOpenStack={isOpenStack} isFile={true} />
        )
    }
}

export default connect(null, mapDispatchToProps)(FileEntry);