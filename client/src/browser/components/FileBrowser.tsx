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
    const { browser } = state;
    return {
        files: browser.files,
        dirs: browser.dirs,
        path: browser.path,
        isOpenStack: browser.isOpenStack,
        drives: browser.drives,
        places: browser.places,
        isLoading: browser.isLoading,
    };
}

const mapDispatchToProps = (dispatch: Dispatch) => {
    return {
        cancel: () => dispatch(browserActions.Actions.cancel()),
        selectAll: () => dispatch(browserActions.Actions.selectAll()),
        selectFiles: () => dispatch(browserActions.Actions.selectFiles()),
        toggleStack: () => dispatch(browserActions.Actions.toggleStack()),
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

const FileBrowser: React.SFC<MergedProps> = ({ files, dirs, path, drives, places, cancel, isOpenStack, isLoading, selectAll,toggleStack, selectFiles }) => {
    const getSortKey = (item: DirectoryListingDetails) => item.name.toLowerCase();
    const dirEntries = sortByKey(dirs, getSortKey).map((dir) => (style: object) => <FolderEntry style={style} onChange={scrollToTop} path={path} details={dir} />);
    const fileEntries = sortByKey(files, getSortKey).map((f,i) => ((style: object) => <FileEntry style={style} path={path} details={f} isOpenStack={isOpenStack} index={i} />));
    const entries = dirEntries.concat(fileEntries);

    const cellFn: EntryFn = ({ index, style }) => {
        return entries[index](style)
    }

    let list = (
        <List style={{ overflowY: "scroll" }} ref={listRef} height={300} width="100%" itemCount={entries.length} itemSize={35}>
            {cellFn}
        </List>
    );

    if (isLoading) {
        // FIXME: hardcoded height
        list = (
            <Segment loading={true} style={{ height: "300px" }} />
        )
    }

    return (
        <Segment.Group>
            <Segment>
                <Header as="h2">Open dataset</Header>
            </Segment>
            <Segment>
                <PathBar currentPath={path} drives={drives} places={places} onChange={scrollToTop} />
            </Segment>
            <Segment>
                <FileBrowserHeader />
                {list}
            </Segment>
            {
                isOpenStack ? (
                    <Segment>
                        <Button onClick={cancel}>Cancel</Button>
                        <Button onClick={selectAll}>Select All</Button>
                        <Button onClick={toggleStack}>Close Stack</Button>
                        <Button primary={true} onClick={selectFiles}>Open</Button>
                    </Segment>
                ) : (
                    <Segment>
                        <Button onClick={cancel}>Cancel</Button>
                        <Button onClick={toggleStack}>Open Stack</Button>
                    </Segment>
                )
            }
        </Segment.Group>
    );
}

export default connect(mapStateToProps, mapDispatchToProps)(FileBrowser);