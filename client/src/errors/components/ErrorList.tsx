import * as React from "react";
import { connect } from "react-redux";
import { Modal } from "semantic-ui-react";
import { RootReducer } from "../../store";
import Error from "./Error";

const mapStateToProps = (state: RootReducer) => {
    return {
        errors: state.errors,
    }
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const ErrorList: React.SFC<MergedProps> = ({ errors }) => {
    const numShown = 3;
    const latestErrors = errors.ids.slice(Math.max(0, errors.ids.length - numShown));
    const showModal = errors.ids.length > 0;
    return (
        <Modal open={showModal}>
            {latestErrors.map(error => <Error error={errors.byId[error]} key={error} />)}
        </Modal>
    );
}

export default connect(mapStateToProps)(ErrorList);