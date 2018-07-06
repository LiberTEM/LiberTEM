import * as React from "react";
import { connect } from "react-redux";
import { Message } from "semantic-ui-react";
import { RootReducer } from "../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        errorList: state.errors.errorList,
    }
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const ErrorList: React.SFC<MergedProps> = ({ errorList }) => {
    const numShown = 3;
    const latestErrors = errorList.slice(errorList.length - numShown)
    return (
        <>
            {latestErrors.map((error, idx) => <Message negative={true} key={idx}>{error.msg}</Message>)}
        </>
    );
}

export default connect(mapStateToProps)(ErrorList);