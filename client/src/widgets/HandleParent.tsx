import * as React from "react";
import { DraggableHandle } from "./DraggableHandle";
import { HandleRenderFunction } from "./types";

export interface HandleParentProps {
    width: number,
    height: number,
    onKeyboardEvent?: React.KeyboardEventHandler<SVGElement>,
    handles: HandleRenderFunction[],
}

export class HandleParent extends React.Component<HandleParentProps> {
    public currentHandle: DraggableHandle | undefined;

    public handleDragStart = (h: DraggableHandle): void => {
        this.currentHandle = h;
    }

    public handleDrop = (): void => {
        this.currentHandle = undefined;
    }

    public handleMouseMove = (e: React.MouseEvent<SVGElement>): void => {
        if (this.currentHandle) {
            return this.currentHandle.externalMouseMove(e);
        }
    }

    public handleMouseLeave = (): void => {
        if (this.currentHandle) {
            this.currentHandle.externalLeave();
        }
    }

    public handleMouseUp = (): void => {
        if (this.currentHandle) {
            this.currentHandle.externalMouseUp();
        }
    }

    public render() {
        const { width, height } = this.props;
        const styles = {
            outline: "1px dashed black"
        }
        return (
            <g
                onMouseMove={this.handleMouseMove}
                onMouseLeave={this.handleMouseLeave}
                onMouseUp={this.handleMouseUp}
                onKeyDown={this.props.onKeyboardEvent}
                style={styles}
                tabIndex={0}
            >
                <rect style={{ fill: "transparent" }}
                    x={0} y={0} width={width} height={height}
                />
                {this.renderHandles()}
            </g>
        );
    }

    public renderHandles() {
        const { handles, onKeyboardEvent } = this.props;
        // we need to inform the handle when there are move/up/leave events
        // on this parent element, for which we need to know the current handle.
        // so we pass the handle a dragstart/drop function and kindly ask it
        // to call us if it starts to be dragged or is dropped.
        return handles.map((h, i) => {
            const elem = h(this.handleDragStart, this.handleDrop, onKeyboardEvent);
            if (React.isValidElement(elem)) {
                return React.cloneElement(elem, { key: i });
            }
            return null;
        });
    }
}

export default HandleParent;