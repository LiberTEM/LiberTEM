import * as React from "react";
import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { List, Modal } from "semantic-ui-react";
import { v4 as uuid } from 'uuid';
import { HostDetails } from "../../messages";
import { getClusterDetail } from "../api"
import * as errorActions from "../../errors/actions";

interface LocalStatusProps {
    localCores: number;
    cudas: Record<number, number>;
}

const LocalStatus: React.FC<LocalStatusProps> = ({ localCores: localCores, cudas }) => {
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
        }).catch((e: Error) => {
            const id = uuid();
            const timestamp = Date.now();
            dispatch(errorActions.Actions.generic(id, `Could not fetch cluster details: ${e.toString()}`, timestamp));
        })
    }, []);

    const nonEmptyCudas = Object.entries(cudas)
        .filter(([, num_workers]) => num_workers > 0)

    if (Object.keys(nonEmptyCudas).length === 0) {
        cudaText = "None selected";
    } else {
        cudaText = nonEmptyCudas
            .map(([id, num_workers]) => `GPU ${id} (${num_workers} workers)`)
            .join(',');
    }

    return (
        <Modal.Content>
            <List>
                <List.Item>
                    <List.Content as="h4">Connected to local cluster</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Number of local core : {localCores}</List.Content>
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
