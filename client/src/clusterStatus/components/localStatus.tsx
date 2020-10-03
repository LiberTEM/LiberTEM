import * as React from "react";
import { useEffect, useState } from "react";
import { List, Modal } from "semantic-ui-react";
import { HostDetails } from "../../messages";
import { getClusterDetail } from "../api"

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

    useEffect(() => {
        const updateDetails = async () => {
            await getClusterDetail().then(newDetails => {
                setDetails(newDetails.details);
            });
        };
        updateDetails();
    }, []);

    if (cudas.length === 0) {
        cudaText = "None selected";
    } else {
        const ids = cudas
            .map(id => {
                return ` ${id}`;
            })
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
