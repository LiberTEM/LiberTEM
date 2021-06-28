import * as React from 'react';
import { connect } from "react-redux";
import { Dispatch } from 'redux';
import { Dropdown, DropdownItemProps } from "semantic-ui-react";
import * as browserActions from '../actions';

type PathDropdownItemProps = DropdownItemProps & {
    onChange: () => void,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: PathDropdownItemProps) => ({
    list: () => {
        if (ownProps.value !== undefined) {
            dispatch(browserActions.Actions.list(ownProps.value.toString()));
            window.setTimeout(() => ownProps.onChange(), 0);
        }
    },
})

type MergedProps = ReturnType<typeof mapDispatchToProps> & PathDropdownItemProps;

const PathDropDownItem: React.FC<MergedProps> = ({ list, ...props }) => {
    const newProps = {
        onClick: list,
        ...props,
    }
    return <Dropdown.Item {...newProps} />;
}

export default connect(null, mapDispatchToProps)(PathDropDownItem);