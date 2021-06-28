import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Dropdown, Menu } from "semantic-ui-react";
import * as configActions from '../../config/actions';
import * as browserActions from '../actions';
import { FSPlaces } from "../types";
import PathDropDownItem from "./PathDropDownItem";
import PathInput from "./PathInput";
import RecentFiles from "./RecentFiles";

const mapDispatchToProps = (dispatch: Dispatch, ownProps: PathBarProps) => ({
    refresh: () => {
        dispatch(browserActions.Actions.list(ownProps.currentPath));
        window.setTimeout(() => ownProps.onChange(), 0);
    },
    handleInputChange: (path: string) => {
        dispatch(browserActions.Actions.list(path));
        window.setTimeout(() => ownProps.onChange(), 0);
    },
    goUp: () => {
        dispatch(browserActions.Actions.list(ownProps.currentPath, '..'));
        window.setTimeout(() => ownProps.onChange(), 0);
    },
    toggleStar: () => {
        dispatch(configActions.Actions.toggleStar(ownProps.currentPath));
    }
});

interface PathBarProps {
    currentPath: string,
    onChange: () => void,
    drives: string[],
    places: FSPlaces,
    starred: string[],
}

type MergedProps = ReturnType<typeof mapDispatchToProps> & PathBarProps;

const PathBar: React.FC<MergedProps> = ({ currentPath, drives, places, starred, onChange, refresh, goUp, handleInputChange, toggleStar }) => {
    const driveOptions = drives.map((path) => ({ key: path, text: path }));
    const placeOptions = Object.keys(places).map((key) => ({ key: places[key].path, text: places[key].title }));
    const starOptions = starred.map((path) => ({ key: path, text: path }));
    const isStarred = starred.includes(currentPath);
    const starredIcon = isStarred ? "star" : "star outline";
    const bookmarkTitle = isStarred ? "Remove from bookmarks" : "Bookmark this location";

    return (
        <Menu>
            <RecentFiles />
            <Dropdown text="Go to..." floating item>
                <Dropdown.Menu>
                    <Dropdown.Header content="Drives" />
                    {driveOptions.map((option) =>
                        <PathDropDownItem key={option.key} value={option.key} content={option.text} onChange={onChange} />
                    )}
                    <Dropdown.Header content="Places" />
                    {placeOptions.map((option) =>
                        <PathDropDownItem key={option.key} value={option.key} content={option.text} onChange={onChange} />
                    )}
                    <Dropdown.Header content="Starred" />
                    {starOptions.length === 0 && 
                        <Dropdown.Item key="empty" value="empty" content="No bookmarks yet" />
                    }
                    {starOptions.map((option) =>
                        <PathDropDownItem key={option.key} value={option.key} content={option.text} onChange={onChange} />
                    )}
                </Dropdown.Menu>
            </Dropdown>
            <Menu.Item title={bookmarkTitle} icon={starredIcon} onClick={toggleStar} />
            <Menu.Item icon="arrow up" onClick={goUp} />
            <Menu.Item style={{ flexGrow: 1 }}>
                <PathInput onChange={handleInputChange} initialPath={currentPath} />
            </Menu.Item>
            <Menu.Item icon="refresh" onClick={refresh} />
        </Menu>
    );
}


export default connect(null, mapDispatchToProps)(PathBar);