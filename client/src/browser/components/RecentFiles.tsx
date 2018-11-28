import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Dropdown, DropdownItemProps } from "semantic-ui-react";
import { DatasetTypes } from "../../messages";
import { RootReducer } from "../../store";
import * as browserActions from '../actions';


type DropdownOptions = Array<{
    text: string,
    value: {
        type: DatasetTypes,
        path: string,
    },
}>;

const mapStateToProps = (state: RootReducer) => {
    return {
        lastOpened: state.config.lastOpened,
        fileHistory: state.config.fileHistory,
        separator: state.config.separator,
    };
}

const mapDispatchToProps = (dispatch: Dispatch) => {
    return {
        select: (path: string) => dispatch(browserActions.Actions.selectFullPath(path)),
    };
}

type MergedProps = ReturnType<typeof mapStateToProps> & ReturnType<typeof mapDispatchToProps>;

const RecentFiles: React.SFC<MergedProps> = ({ lastOpened, fileHistory, select }) => {

    const recentFiles: DropdownOptions = fileHistory.filter(path => lastOpened[path]).map(path => {
        const item = lastOpened[path];
        return {
            text: item.path,
            value: {
                type: item.type,
                path: item.path,
            },
        };
    });

    const onClick = (e: React.MouseEvent<HTMLDivElement>, data: DropdownItemProps) => data.value && select(data.value.toString())

    return (
        <Dropdown direction="left" text="Recent" icon="ellipsis vertical" floating={true} labeled={true} button={true} className='icon'>
            <Dropdown.Menu>
                <Dropdown.Header content="recent datasets" />
                {recentFiles.map((option, idx) => (
                    <Dropdown.Item key={idx} value={option.value.path} content={option.text} onClick={onClick} />
                ))}
            </Dropdown.Menu>
        </Dropdown>
    );
}


export default connect(mapStateToProps, mapDispatchToProps)(RecentFiles);