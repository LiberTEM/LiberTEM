import * as React from "react";
import { connect, useDispatch } from "react-redux";
import { Modal } from "semantic-ui-react";
import { ChannelStatusCodes } from "../../channel/reducers";
import { useDismissEscape } from "../../helpers/hooks";
import { RootReducer } from "../../store";
import { Actions } from "../actions";
import Error from "./Error";

const mapStateToProps = (state: RootReducer) => ({
    errors: state.errors,
    channelConnected: (state.channelStatus.status === ChannelStatusCodes.CONNECTED ||
        state.channelStatus.status === ChannelStatusCodes.READY),
});

type MergedProps = ReturnType<typeof mapStateToProps>;

const ErrorList: React.FC<MergedProps> = ({ errors, channelConnected }) => {
    const numShown = 3;
    const latestErrors = errors.ids.slice(Math.max(0, errors.ids.length - numShown));
    const showModal = errors.ids.length > 0 && channelConnected;

    const dispatch = useDispatch();

    const doDismiss = () => {
        dispatch(Actions.dismissAll());
    }

    useDismissEscape(doDismiss);

    return (
        <Modal open={showModal}>
            {latestErrors.map(error => <Error error={errors.byId[error]} key={error} />)}
        </Modal>
    );
}

export default connect(mapStateToProps)(ErrorList);