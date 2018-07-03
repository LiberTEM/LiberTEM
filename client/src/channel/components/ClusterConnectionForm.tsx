
import * as React from "react";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { getEnumValues } from "../../helpers";
import { ClusterTypeMetadata, ClusterTypes, ConnectRequestParams } from "../../messages";
import { connectToCluster } from "../api";
import LocalConnectionForm from "./LocalConnectionForm";
import TCPConnectionForm from "./TCPConnectionForm";

interface ClusterConnectionProps {
    onSubmit: (e: any) => void,
}

type MergedProps = ClusterConnectionProps;

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
        // tslint:disable-next-line:no-console
        console.log(params);
        connectToCluster(params);
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
                <p>
                    <Dropdown
                        inline={true}
                        options={clusterTypeOptions}
                        value={this.state.clusterType}
                        onChange={this.handleChange}
                    />
                </p>
                {this.renderForm()}
            </>
        )
    }
}

export default ClusterConnectionForm;