import * as React from "react";
import { Button, Icon } from "semantic-ui-react";
import { handleSubmit } from "../api"

const QuitButton: React.SFC = () => {
    return(
        <Button icon={true} floated="right" labelPosition="left" onClick={handleSubmit}>
            <Icon name="shutdown"/>
            Quit
        </Button>
    )
}

export default QuitButton
