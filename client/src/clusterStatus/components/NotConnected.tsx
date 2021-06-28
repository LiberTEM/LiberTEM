import * as React from "react";
import { List, Modal } from "semantic-ui-react";

const NotConnected: React.FC = () => (
    <Modal.Content>
        <List>
            <List.Item>
                <List.Content>
                    Not Connected
                </List.Content>
            </List.Item>
        </List>
    </Modal.Content>
);

export default NotConnected;