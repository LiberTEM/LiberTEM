import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Dropdown, Menu } from "semantic-ui-react";
import * as browserActions from '../actions';
import { FSPlaces } from "../types";
import PathDropDownItem from "./PathDropDownItem";
import PathInput from "./PathInput";
import RecentFiles from "./RecentFiles";

const mapDispatchToProps = (dispatch: Dispatch, ownProps: PathBarProps) => {
    return {
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
    };
}

interface PathBarProps {
    currentPath: string,
    onChange: () => void,
    drives: string[],
    places: FSPlaces,
}

type MergedProps = ReturnType<typeof mapDispatchToProps> & PathBarProps;

const PathBar: React.SFC<MergedProps> = ({ currentPath, drives, places, onChange, refresh, goUp, handleInputChange }) => {
    const driveOptions = drives.map((path) => ({ key: path, text: path }));
    const placeOptions = Object.keys(places).map((key) => ({ key: places[key].path, text: places[key].title }))
    return (
        <Menu>
            <RecentFiles />
            <Dropdown text="Go to..." floating={true} item={true}>
                <Dropdown.Menu>
                    <Dropdown.Header content="Drives" />
                    {driveOptions.map((option) => {
                        return <PathDropDownItem key={option.key} value={option.key} content={option.text} onChange={onChange} />
                    })}
                    <Dropdown.Header content="Places" />
                    {placeOptions.map((option) => {
                        return <PathDropDownItem key={option.key} value={option.key} content={option.text} onChange={onChange} />
                    })}
                </Dropdown.Menu>
            </Dropdown>
            <Menu.Item icon="arrow up" onClick={goUp} />
            <Menu.Item style={{ flexGrow: 1 }}>
                <PathInput onChange={handleInputChange} initialPath={currentPath} />
            </Menu.Item>
            <Menu.Item icon="refresh" onClick={refresh} />
        </Menu>
    );
}


export default connect(null, mapDispatchToProps)(PathBar);