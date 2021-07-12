import * as React from "react";
import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { List, Modal } from "semantic-ui-react";
import { v4 as uuid } from 'uuid';
import { HostDetails } from "../../messages";
import { getClusterDetail } from "../api"
import * as errorActions from "../../errors/actions";

interface LocalStatusProps {
    localCore: number;
    cudas: number[];
}

const LocalStatus: React.FC<LocalStatusProps> = ({ localCore, cudas }) => {
    let cudaText: string;

    const intialDetails: HostDetails[] = [
        {
            host: "",
            cpu: 0,
            cuda: 0,
            service: 0,
        },
    ];

    const [clustDetails, setDetails] = useState<HostDetails[]>(intialDetails)

    const dispatch = useDispatch();

    useEffect(() => {
        getClusterDetail().then(newDetails => {
            setDetails(newDetails.details);
        }).catch(() => {
            const id = uuid();
            const timestamp = Date.now();
            dispatch(errorActions.Actions.generic(id, "Could not copy to clipboard", timestamp));
        })
    }, []);

    if (cudas.length === 0) {
        cudaText = "None selected";
    } else {
        const ids = cudas
            .map(id => ` ${id}`)
            .join(",");
        cudaText = `GPU ${ids}`;
    }

    return (
        <Modal.Content>
            <List>
                <List.Item>
                    <List.Content as="h4">Connected to local cluster</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Number of local core : {localCore}</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Number of CPU workers : {clustDetails[0].cpu} </List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Number of CUDA workers : {clustDetails[0].cuda}</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Selected CUDA devices : {cudaText}</List.Content>
                </List.Item>
            </List>
        </Modal.Content>
    );
};

export default LocalStatus;
