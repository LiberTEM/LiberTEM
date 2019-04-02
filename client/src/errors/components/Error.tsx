import * as React from 'react';
import { connect } from 'react-redux';
import { Dispatch } from 'redux';
import { Message } from 'semantic-ui-react';
import * as errorActions from '../actions';
import { ErrorMessage } from '../reducers';

const mapDispatchToProps = (dispatch: Dispatch, ownProps: ErrorProps) => {
    return {
        dismiss: () => dispatch(errorActions.Actions.dismiss(ownProps.error.id)),
    }
}

interface ErrorProps {
    error: ErrorMessage,
}

type MergedProps = ErrorProps & ReturnType<typeof mapDispatchToProps>;

const ErrorComponent: React.SFC<MergedProps> = ({ dismiss, error }) => {
    return (
        <Message negative={true} onDismiss={dismiss}>{error.msg}</Message>
    )
}

export default connect(null, mapDispatchToProps)(ErrorComponent);