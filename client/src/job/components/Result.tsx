import * as React from "react";
import { Image, Portal, Segment } from "semantic-ui-react";
import { getPickFrameURL } from "../../dataset/api";
import { defaultDebounce } from "../../helpers";
import { DatasetState } from "../../messages";
import { JobState } from "../types";
import PlaceholderImage from "./PlaceholderImage";

interface ResultProps {
    width: number,
    height: number,
    job: JobState,
    dataset: DatasetState,
    idx: number,
}


interface ResultState {
    popupOpen: boolean,
    x?: number,
    y?: number,
}

interface ResultPopupProps {
    open: boolean,
    x?: number,
    y?: number,
    onClose: () => void,
    dataset: DatasetState,
}

const ResultPopup: React.SFC<ResultPopupProps> = ({ open, x, y, onClose, dataset }) => {
    if (x === undefined || y === undefined) {
        return null;
    }
    return (
        <Portal onClose={onClose} open={open}>
            <Segment style={{ left: '50%', position: 'fixed', top: '50%', zIndex: 1000 }}>
                <img src={getPickFrameURL(dataset, x, y)} />
                <p>Frame at x={x} y={y}</p>
            </Segment>
        </Portal>
    );
}

class Result extends React.Component<ResultProps, ResultState> {
    public state: ResultState = { popupOpen: false };

    public setStateDebounced = defaultDebounce((obj: object) => {
        this.setState(obj);
    })

    public pickCoords = (e: React.MouseEvent<HTMLElement>) => {
        const { width, height } = this.props;
        const target = e.target as HTMLElement;
        const targetRect = target.getBoundingClientRect();
        const browserX = e.clientX - targetRect.left
        const browserY = e.clientY - targetRect.top;
        const imgX = Math.round((browserX / targetRect.width) * width);
        const imgY = Math.round((browserY / targetRect.height) * height);

        return {
            x: imgX,
            y: imgY,
        }
    }

    public onPick = (e: React.MouseEvent<HTMLElement>) => {
        const { x, y } = this.pickCoords(e);
        this.setState({ popupOpen: true, x, y })
    }

    public onClosePopup = () => {
        this.setState({ popupOpen: false });
    }

    public onMouseMove = (e: React.MouseEvent<HTMLElement>) => {
        if (!this.state.popupOpen) {
            return;
        }
        const { x, y } = this.pickCoords(e);
        this.setStateDebounced({ x, y })
    };

    public render() {
        const { job, dataset, idx, width, height } = this.props;
        const { x, y, popupOpen } = this.state;
        const result = job.results[idx];

        return (
            <PlaceholderImage width={width} height={height}>
                <ResultPopup dataset={dataset} open={popupOpen} x={x} y={y} onClose={this.onClosePopup} />
                <Image onClick={this.onPick} onMouseMove={this.onMouseMove} style={{ width: "100%", height: "auto", imageRendering: "pixelated" }} src={result.imageURL} width={width} height={height} />
            </PlaceholderImage>
        );
    }
};

export default Result;