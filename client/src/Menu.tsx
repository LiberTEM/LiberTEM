import React from "react";
import { Button, Menu, Modal, Popup } from "semantic-ui-react";
import About from "./About";
import ClusterStatus from "./clusterStatus/components/Cluster"
import QuitButton from "./shutdown/components/ShutdownButton";
import MyProgress from "./progress/components/Progress";

const HeaderMenu: React.FC = () => (

    <Menu fixed="top">
        <Menu.Item>
            <Modal trigger={<Button content="About" />}>
                <Popup.Header>About LiberTEM</Popup.Header>
                <Popup.Content>
                    <About />
                </Popup.Content>
            </Modal>
        </Menu.Item>
        <Menu.Item position="right" style={{flexGrow: 1}}>
            <MyProgress /> 
        </Menu.Item>
        <Menu.Menu position="right">
            <Menu.Item>
                <ClusterStatus />
            </Menu.Item>
            <Menu.Item>
                <QuitButton />
            </Menu.Item>
        </Menu.Menu>
    </Menu>
);

export default HeaderMenu;
