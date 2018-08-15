import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { FixedSizeList as List } from 'react-window';
import { Button } from "semantic-ui-react";
import { DirectoryListingDetails } from "../../messages";
import { RootReducer } from "../../store";
import * as browserActions from '../actions';
import FileEntry from "./FileEntry";
import FolderEntry from "./FolderEntry";

const mapStateToProps = (state: RootReducer) => {
    const { browser } = state;
    return {
        files: browser.files,
        dirs: browser.dirs,
        path: browser.path,
    };
}

const mapDispatchToProps = (dispatch: Dispatch) => {
    return {
        list: (path: string) => dispatch(browserActions.Actions.list(path)),
        cancel: () => dispatch(browserActions.Actions.cancel()),
    };
}

type MergedProps = ReturnType<typeof mapStateToProps> & ReturnType<typeof mapDispatchToProps>;

interface EntryFnArgs {
    index: number,
    style: object
}
type EntryFn = (arg: EntryFnArgs) => void

const listRef = React.createRef<List>();

const scrollToTop = () => {
    if (listRef.current === null) {
        return;
    }
    listRef.current.scrollToItem(0);
}

function sortByKey<T extends object>(array: T[], getKey: (item: T) => any) {
    return array.sort((a, b) => {
        const x = getKey(a);
        const y = getKey(b);
        return ((x < y) ? -1 : ((x > y) ? 1 : 0));
    });
}

const FileBrowser: React.SFC<MergedProps> = ({ files, dirs, path, cancel }) => {
    const getSortKey = (item: DirectoryListingDetails) => item.name.toLowerCase();
    const dirEntries = sortByKey(dirs, getSortKey).map((dir) => (style: object) => <FolderEntry style={style} onChange={scrollToTop} path={path} details={dir} />);
    const fileEntries = sortByKey(files, getSortKey).map((f) => ((style: object) => <FileEntry style={style} path={path} details={f} />));
    const entries = dirEntries.concat(fileEntries);
    const entryFn: EntryFn = ({ index, style }) => {
        return entries[index](style)
    };
    return (
        <>
            <p>{path}</p>
            <List ref={listRef} height={300} width="100%" itemCount={entries.length} itemSize={35}>
                {entryFn}
            </List>
            <Button onClick={cancel}>Cancel</Button>
        </>
    );
}

export default connect(mapStateToProps, mapDispatchToProps)(FileBrowser);