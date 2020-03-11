import * as React from "react";
import { connect, useDispatch } from "react-redux";
import { Modal } from "semantic-ui-react";
import { RootReducer } from "../../store";
import { Actions } from "../actions";
import Error from "./Error";

const mapStateToProps = (state: RootReducer) => {
    return {
        errors: state.errors,
        clusterConnected: state.clusterConnection.status === "connected",
        channelConnected: (state.channelStatus.status === "connected" ||
            state.channelStatus.status === "ready"),
    }
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const ErrorList: React.SFC<MergedProps> = ({ errors, clusterConnected, channelConnected }) => {
    const numShown = 3;
    const latestErrors = errors.ids.slice(Math.max(0, errors.ids.length - numShown));
    const showModal = errors.ids.length > 0 && clusterConnected && channelConnected;

    const dispatch = useDispatch();

    React.useEffect(() => {
        const handleEsc = (ev: KeyboardEvent) => {
            if(ev.code === "Escape" || ev.keyCode === 27) {
                dispatch(Actions.dismissAll());
            }
        }
        document.addEventListener("keyup", handleEsc);

        return () => {
            document.removeEventListener("keyup", handleEsc);
        };
    });

    return (
        <Modal open={showModal}>
            {latestErrors.map(error => <Error error={errors.byId[error]} key={error} />)}
        </Modal>
    );
}

export default connect(mapStateToProps)(ErrorList);