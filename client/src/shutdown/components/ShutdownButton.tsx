import * as React from "react";
import { connect } from "react-redux";
import { Button, Header, Icon, Modal } from "semantic-ui-react";
import * as channelActions from "../../channel/actions";
import { DispatchProps } from "../../helpers/props";
import { RootReducer } from "../../store";
import { doShutdown } from "../api";
import { ChannelStatusCodes } from "../../channel/reducers";

const mapDispatchToProps = {
    closeLoopAction: channelActions.Actions.closeloop,
    shutdownAction: channelActions.Actions.shutdown,
};

const mapStateToProps = (state: RootReducer) => ({
    channel: state.channelStatus.status,
})

type MergedProps = DispatchProps<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

class ShutdownButton extends React.Component<MergedProps> {
    public state = {
        modal: false,
        shutdown: false,
    };

    public modalOpen = () => {
        this.setState({ modal: true });
    };

    public modalClose = () => {
        this.setState({ modal: false });
    };

    public handleShutdown = () => {
        this.setState({ shutdown: true });
        void doShutdown().then(() => {
            const timestamp = Date.now();
            this.props.closeLoopAction(timestamp);
        });
    };

    public componentDidUpdate() {
        if (this.props.channel === ChannelStatusCodes.WAITING && this.state.shutdown) {
            const timestamp = Date.now();
            this.modalClose();
            this.props.shutdownAction(timestamp);
        }
    }

    public render() {
        return (
            <Modal
                trigger={
                    <Button
                        content="Shutdown"
                        icon="shutdown"
                        onClick={this.modalOpen}
                        disabled={this.state.shutdown}
                        labelPosition="left"
                        floated="right"
                    />
                }
                open={this.state.modal}
                closeOnDimmerClick={false}
                onClose={this.modalClose}
                size="mini"
            >
                <Header icon="shutdown" content="Confirm shutdown" />
                <Modal.Content>
                    <p>Do you want to shutdown ?</p>
                </Modal.Content>
                <Modal.Actions>
                    <Button onClick={this.modalClose} disabled={this.state.shutdown}>
                        <Icon name="remove" /> Cancel
                    </Button>
                    <Button primary loading={this.state.shutdown} disabled={this.state.shutdown} onClick={this.handleShutdown}>
                        <Icon name="checkmark" /> Shutdown
                    </Button>
                </Modal.Actions>
            </Modal>
        );
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(ShutdownButton);
