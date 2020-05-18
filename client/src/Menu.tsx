import React from 'react'
import { Button, Menu, Modal, Popup } from 'semantic-ui-react'
import About from './About';
import QuitButton from './quit/components/QuitButton';


const HeaderMenu: React.SFC = () => {
    return(
        <Menu fixed="top">
            <Menu.Item>
            <Modal trigger={
                        <Button content="About" />
                    }>
                        <Popup.Header>About LiberTEM</Popup.Header>
                        <Popup.Content>
                            <About />
                        </Popup.Content>
                    </Modal>
            </Menu.Item>
            <Menu.Menu position="right">
                <Menu.Item>
                    <QuitButton />
                </Menu.Item>
            </Menu.Menu>
        </Menu>
    )
}

export default HeaderMenu