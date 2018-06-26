import * as React from "react";
import { DraggableHandle } from "./DraggableHandle";

export interface HandleParentProps {
    width: number,
    height: number,
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

    public handleMouseLeave = (e: React.MouseEvent<SVGElement>): void => {
        if (this.currentHandle) {
            this.currentHandle.externalLeave(e);
        }
    }

    public handleMouseUp = (e: React.MouseEvent<SVGElement>): void => {
        if (this.currentHandle) {
            this.currentHandle.externalMouseUp(e);
        }
    }

    public render() {
        const { width, height } = this.props;
        return (
            <g
                onMouseMove={this.handleMouseMove}
                onMouseLeave={this.handleMouseLeave}
                onMouseUp={this.handleMouseUp}
            >
                <rect style={{ fill: "transparent" }}
                    x={0} y={0} width={width} height={height}
                />
                {this.renderChildren()}
            </g>
        );
    }

    public renderChildren() {
        const { children } = this.props;
        return React.Children.map(children, child => {
            if (!React.isValidElement(child)) {
                return child;
            }
            const newProps = {
                parentOnDragStart: this.handleDragStart,
                parentOnDrop: this.handleDrop,
            };
            return React.cloneElement(child, newProps);
        })
    }
}

export default HandleParent;