import * as React from "react";
import { connect } from "react-redux";
import { FixedSizeList as List } from "react-window";
import { Dispatch } from "redux";
import { Button, Header, Segment } from "semantic-ui-react";
import { DirectoryListingDetails } from "../../messages";
import { RootReducer } from "../../store";
import * as browserActions from '../actions';
import FileBrowserHeader from "./FileBrowserHeader";
import FileEntry from "./FileEntry";
import FolderEntry from "./FolderEntry";
import PathBar from "./PathBar";

const mapStateToProps = (state: RootReducer) => {
    const { browser, config } = state;
    return {
        files: browser.files,
        dirs: browser.dirs,
        path: browser.path,
        drives: browser.drives,
        places: browser.places,
        isLoading: browser.isLoading,
        starred: config.starred,
    };
}

const mapDispatchToProps = (dispatch: Dispatch) => ({
    cancel: () => dispatch(browserActions.Actions.cancel()),
});

type MergedProps = ReturnType<typeof mapStateToProps> & ReturnType<typeof mapDispatchToProps>;

interface EntryFnArgs {
    index: number,
    style: Record<string, unknown>,
}
type EntryFn = (arg: EntryFnArgs) => void

const listRef = React.createRef<List>();

const scrollToTop = () => {
    if (listRef.current === null) {
        return;
    }
    listRef.current.scrollToItem(0);
}

const sortByKey = <K, T>(array: T[], getKey: (item: T) => K) => (
    array.sort((a, b) => {
        const x = getKey(a);
        const y = getKey(b);
        return ((x < y) ? -1 : ((x > y) ? 1 : 0));
    })
);

const FileBrowser: React.FC<MergedProps> = ({ files, dirs, path, drives, places, starred, cancel, isLoading }) => {
    const getSortKey = (item: DirectoryListingDetails) => item.name.toLowerCase();
    const dirEntries = sortByKey(dirs, getSortKey).map((dir) => (style: Record<string, unknown>) => <FolderEntry style={style} onChange={scrollToTop} path={path} details={dir} />);
    const fileEntries = sortByKey(files, getSortKey).map((f) => ((style: Record<string, unknown>) => <FileEntry style={style} path={path} details={f} />));
    const entries = dirEntries.concat(fileEntries);

    const cellFn: EntryFn = ({ index, style }) => entries[index](style);

    let list = (
        <List style={{ overflowY: "scroll" }} ref={listRef} height={300} width="100%" itemCount={entries.length} itemSize={35}>
            {cellFn}
        </List>
    );

    if (isLoading) {
        // FIXME: hardcoded height
        list = (
            <Segment loading style={{ height: "300px" }} />
        )
    }

    return (
        <Segment.Group>
            <Segment>
                <Header as="h2">Open dataset</Header>
            </Segment>
            <Segment>
                <PathBar currentPath={path} drives={drives} places={places} starred={starred} onChange={scrollToTop} />
            </Segment>
            <Segment>
                <FileBrowserHeader />
                {list}
            </Segment>
            <Segment>
                <Button onClick={cancel}>Cancel</Button>
            </Segment>
        </Segment.Group>
    );
}

export default connect(mapStateToProps, mapDispatchToProps)(FileBrowser);