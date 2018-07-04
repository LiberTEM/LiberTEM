
import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps, Segment } from "semantic-ui-react";
import * as clusterActions from "../../cluster/actions";
import { getEnumValues } from "../../helpers";
import { ClusterTypeMetadata, ClusterTypes, ConnectRequestParams } from "../../messages";
import LocalConnectionForm from "./LocalConnectionForm";
import TCPConnectionForm from "./TCPConnectionForm";


const mapDispatchToProps = {
    connectToCluster: clusterActions.Actions.connect,
};

type MergedProps = DispatchProps<typeof mapDispatchToProps>;

const clusterTypeKeys = getEnumValues(ClusterTypes);
const clusterTypeOptions = clusterTypeKeys.map(t => ({
    text: ClusterTypeMetadata[ClusterTypes[t as any]].label,
    value: ClusterTypes[t as any],
}));

interface ConnectionParamsState {
    clusterType: ClusterTypes
}

class ClusterConnectionForm extends React.Component<MergedProps, ConnectionParamsState> {
    public state = {
        clusterType: ClusterTypes.LOCAL,
    }

    public setType = (type: ClusterTypes) => {
        this.setState({
            clusterType: type,
        });
    }

    public handleChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        const value = data.value as ClusterTypes;
        this.setType(value);
    }

    public handleSubmit = (params: ConnectRequestParams) => {
        this.props.connectToCluster(params);
    }

    public renderForm() {
        const { clusterType } = this.state;
        switch (clusterType) {
            case ClusterTypes.LOCAL: {
                return <LocalConnectionForm onSubmit={this.handleSubmit} />
            }
            case ClusterTypes.TCP: {
                return <TCPConnectionForm onSubmit={this.handleSubmit} />
            }
        }
    }

    public render() {
        return (
            <>
                <div>
                    <Dropdown
                        inline={true}
                        options={clusterTypeOptions}
                        value={this.state.clusterType}
                        onChange={this.handleChange}
                    />
                </div>
                <Segment>
                    {this.renderForm()}
                </Segment>
            </>
        )
    }
}

export default connect(null, mapDispatchToProps)(ClusterConnectionForm);