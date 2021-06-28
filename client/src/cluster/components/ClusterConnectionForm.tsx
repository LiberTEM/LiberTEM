
import * as React from "react";
import { connect } from "react-redux";
import { Dropdown, DropdownProps, Segment } from "semantic-ui-react";
import * as clusterActions from "../../cluster/actions";
import { getEnumValues } from "../../helpers";
import { DispatchProps } from "../../helpers/props";
import { ClusterTypeMetadata, ClusterTypes, ConnectRequestParams } from "../../messages";
import { RootReducer } from "../../store";
import LocalConnectionForm from "./LocalConnectionForm";
import TCPConnectionForm from "./TCPConnectionForm";


const mapDispatchToProps = {
    connectToCluster: clusterActions.Actions.connect,
};

const mapStateToProps = (state: RootReducer) => ({
    config: state.config,
    lastConnectionType: state.config.lastConnection.type
})

type MergedProps = DispatchProps<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

const clusterTypeKeys = getEnumValues(ClusterTypes);
const clusterTypeOptions = clusterTypeKeys.map(t => ({
    text: ClusterTypeMetadata[ClusterTypes[t]].label,
    value: ClusterTypes[t],
}));

interface ConnectionParamsState {
    clusterType: ClusterTypes
}

class ClusterConnectionForm extends React.Component<MergedProps, ConnectionParamsState> {
    public state = {
        clusterType: this.props.lastConnectionType,
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
        const { config } = this.props;

        switch (clusterType) {
            case ClusterTypes.LOCAL: {
                return <LocalConnectionForm config={config} onSubmit={this.handleSubmit} />
            }
            case ClusterTypes.TCP: {
                return <TCPConnectionForm config={config} onSubmit={this.handleSubmit} />
            }
        }
    }

    public render() {
        return (
            <>
                <div>
                    <Dropdown
                        inline
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

export default connect(mapStateToProps, mapDispatchToProps)(ClusterConnectionForm);