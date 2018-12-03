import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Dropdown, Input, Menu } from "semantic-ui-react";
import { FSPlace } from "../../messages";
import * as browserActions from '../actions';
import PathDropDownItem from "./PathDropDownItem";
import RecentFiles from "./RecentFiles";

const mapDispatchToProps = (dispatch: Dispatch, ownProps: PathBarProps) => {
    return {
        refresh: () => {
            dispatch(browserActions.Actions.listFullPath(ownProps.currentPath));
            window.setTimeout(() => ownProps.onChange(), 0);
        },
    };
}

interface PathBarProps {
    currentPath: string,
    onChange: () => void,
    onUp?: () => void,
    drives: string[],
    places: FSPlace[],
}

type MergedProps = ReturnType<typeof mapDispatchToProps> & PathBarProps;

const PathBar: React.SFC<MergedProps> = ({ currentPath, drives, places, onChange, refresh }) => {
    const driveOptions = drives.map((path) => ({ key: path, text: path }));
    const placeOptions = places.map((place) => ({ key: place.path, text: place.title }))
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
            <Menu.Item style={{ flexGrow: 1 }}>
                <Input disabled={true} value={currentPath} />
            </Menu.Item>
            <Menu.Item icon="refresh" onClick={refresh} />
        </Menu>
    );
}


export default connect(null, mapDispatchToProps)(PathBar);