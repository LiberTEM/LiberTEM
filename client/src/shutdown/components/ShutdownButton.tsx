import * as React from "react";
import { Button, Header, Icon, Modal } from "semantic-ui-react";
import { handleSubmit } from "../api";

class ShutdownButton extends React.Component {
    public state = {
        modelOpen: false,
        shutdown: false,
    };

    public handleOpen = () => {
        this.setState({ modelOpen: true });
    };

    public handleClose = () => {
        this.setState({ modelOpen: false });
    };

    public handleShutdown = () => {
        this.setState({ shutdown: true });
        handleSubmit();
    };

    public render() {
        return (
            <Modal
                trigger={<Button content="Shutdown" icon="shutdown" onClick={this.handleOpen} labelPosition="left" floated="right" />}
                open={this.state.modelOpen}
                onClose={this.handleClose}
                size="mini"
            >
                <Header icon="shutdown" content="Confirm shutdown" />
                <Modal.Content>
                    <p>Do you want to shutdown ?</p>
                </Modal.Content>
                <Modal.Actions>
                    <Button onClick={this.handleClose}>
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

export default ShutdownButton;
