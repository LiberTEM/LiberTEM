import * as React from "react";
import { connect } from "react-redux";
import { Button, Header, Icon, Modal } from "semantic-ui-react";
import { handleSubmit } from "../api";
import { RootReducer } from "../../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        channel: state.channelStatus.status,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

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
        handleSubmit();
    };

    //rename renderForm
    public handleModal() {
        const { shutdown } = this.state;
        const { channel } = this.props;

        if (channel === "waiting" && shutdown) {
            this.setState({ shutdown: false });
            this.modalClose();
        }
    }

    public render() {
        {
            this.handleModal();
        }
        return (
            <Modal trigger={<Button content="Shutdown" icon="shutdown" onClick={this.modalOpen} labelPosition="left" floated="right" />} open={this.state.modal} onClose={this.modalClose} size="mini">
                <Header icon="shutdown" content="Confirm shutdown" />
                <Modal.Content>
                    <p>Do you want to shutdown ?</p>
                </Modal.Content>
                <Modal.Actions>
                    <Button onClick={this.modalClose} disabled={this.state.shutdown}>
                        <Icon name="remove" /> Cancel
                    </Button>
                    <Button primary={true} loading={this.state.shutdown} disabled={this.state.shutdown} onClick={this.handleShutdown}>
                        <Icon name="checkmark" /> Shutdown
                    </Button>
                </Modal.Actions>
            </Modal>
        );
    }
}

export default connect(mapStateToProps)(ShutdownButton);
