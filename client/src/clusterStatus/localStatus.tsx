import * as React from "react";
import { useEffect, useState } from "react";
import { List, Modal } from "semantic-ui-react";
import { HostDetails } from "../messages";

interface LocalStatusProps {
    localCore: number;
    cudas: number[];
    details: HostDetails[];
}

const LocalStatus: React.FC<LocalStatusProps> = ({ localCore, cudas, details }) => {
    const [cudaText, setcudaText] = useState("");

    useEffect(() => {
        if (cudas.length === 0) {
            setcudaText("Not selected");
        } else {
            const ids = cudas
                .map(id => {
                    return ` ${id}`;
                })
                .join(",");
            setcudaText(`GPU ${ids}`);
        }
    }, [cudas]);

    return (
        <Modal.Content>
            <List>
                <List.Item>
                    <List.Content as="h4">Connected to local cluster</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Local core : {localCore}</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>CPU : {details[0].cpu} </List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>CUDA : {details[0].cuda}</List.Content>
                    <List.Content>Selected CUDA devices : {cudaText}</List.Content>
                </List.Item>
            </List>
        </Modal.Content>
    );
};

export default LocalStatus;
