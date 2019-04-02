import * as React from "react";
import { connect } from "react-redux";
import { Header, Icon, Message } from "semantic-ui-react";
import { RootReducer } from "../../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        busy: state.openDataset.busy,
        path: state.openDataset.busyPath,
    };
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const DatasetOpenSpinner: React.SFC<MergedProps> = ({ busy, path }) => {
    if (!busy) {
        return null;
    }
    return (
        <>
            <Header as="h2" dividing={true}>Loading...</Header>
            <Message icon={true}>
                <Icon name='cog' loading={true} />
                <Message.Content>
                    <Message.Header>Detecting parameters for {path}</Message.Header>
                </Message.Content>
            </Message>
        </>
    );
}

export default connect(mapStateToProps)(DatasetOpenSpinner);