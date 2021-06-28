import * as React from "react";
import styled from 'styled-components';
import { Point2D } from "../basicTypes";
import { handleKeyEvent, ModifyCoords } from "./kbdHandler";

export type HandleProps = {
    x: number,
    y: number,
    scale: number,
    withCross?: boolean,
    focusRef: React.RefObject<SVGGElement>,
} & React.SVGProps<SVGGElement>;

const StyledCircle = styled.circle`
    stroke: red;
    stroke-width: 1;
    fill: transparent;
    g:focus > & {
        stroke: lightgreen;
    }
`;

const Cross = styled.path`
    stroke: red;
    stroke-width: 1;
    fill: transparent;
    g:focus > & {
        stroke: lightgreen;
    }
`;

const FocusGroup = styled.g`
    &:focus { outline: none; }
`;

const Handle: React.FC<HandleProps> = ({ scale, x, y, withCross, focusRef, ...args }) => {
    const r = 3;
    // scaleMatrix is needed to set the origin of the scale
    const scaleMatrix = `matrix(${scale}, 0, 0, ${scale}, ${x - scale * x}, ${y - scale * y})`;
    const style: React.CSSProperties = { transform: scaleMatrix };
    const crossSpec = `
        M${x - 4 - r / 2} ${y} L ${x - r / 2} ${y} M${x + r / 2} ${y} L ${x + 4 + r / 2} ${y}
        M${x} ${y - 4 - r / 2} L ${x} ${y - r / 2} M${x} ${y + r / 2} L ${x} ${y + 4 + r / 2}
    `;
    const cross = withCross ? <Cross d={crossSpec} style={style} /> : null;
    return (
        <FocusGroup {...args} ref={focusRef}>
            <StyledCircle cx={x} cy={y} r={r} style={style} />
            {cross}
        </FocusGroup>
    )
}

export interface DraggableHandleProps {
    x: number,
    y: number,
    withCross?: boolean,
    imageWidth?: number,
    onDragMove?: (x: number, y: number) => void,
    parentOnDragStart: (h: DraggableHandle) => void,
    parentOnDrop: (x: number, y: number) => void,
    onKeyboardEvent?: (e: React.KeyboardEvent<SVGElement>) => void,
    constraint?: (p: Point2D) => Point2D,
}

export const getScalingFactor = (elem: SVGElement): number => {
    const svg = elem.ownerSVGElement;
    if (svg === null) {
        throw new Error("no owner SVG element?");
    }
    const inWidthAttr = svg.getAttribute("width");
    if (inWidthAttr === null) {
        throw new Error("no width on SVG element?");
    }
    const inWidth = +inWidthAttr;
    const svgMeasurements = svg.getBoundingClientRect();
    return svgMeasurements.width / inWidth;
};

const relativeCoords = (e: React.MouseEvent, parent: SVGElement) => {
    const f = getScalingFactor(parent);
    const parentPos = parent.getBoundingClientRect();
    const res = {
        x: (e.pageX - (parentPos.left + window.pageXOffset)) / f,
        y: (e.pageY - (parentPos.top + window.pageYOffset)) / f,
    }
    return res;
}

/**
 * stateful draggable handle, to be used as part of <svg/>
 */
export class DraggableHandle extends React.Component<DraggableHandleProps> {
    public posRef: React.RefObject<SVGRectElement>;
    public focusRef: React.RefObject<SVGGElement>;

    public state = {
        dragging: false,
        drag: { x: 0, y: 0 },
    }

    public constructor(props: DraggableHandleProps) {
        super(props);
        this.posRef = React.createRef<SVGRectElement>();
        this.focusRef = React.createRef<SVGGElement>();
    }

    // mousemove event from outside (delegated from surrounding element)
    public externalMouseMove = (e: React.MouseEvent<SVGElement>): void => {
        this.move(e);
    }

    // mouseleave event from outside (delegated from surrounding element)
    public externalLeave = (): void => {
        this.stopDrag();
    }

    // mouseup event from outside (delegated from surrounding element)
    public externalMouseUp = (): void => {
        this.stopDrag();
    }

    public applyConstraint = (p: Point2D): Point2D => {
        const { constraint } = this.props;
        if (constraint) {
            return constraint(p);
        } else {
            return p;
        }
    }

    public startDrag = (e: React.MouseEvent<SVGElement>): void => {
        e.preventDefault();
        const { parentOnDragStart } = this.props;
        if (this.posRef.current) {
            this.setState({
                dragging: true,
                drag: this.applyConstraint(relativeCoords(e, this.posRef.current)),
            });
            if (parentOnDragStart) {
                parentOnDragStart(this);
            }
            if (this.focusRef.current && this.focusRef.current.focus) {
                this.focusRef.current.focus();
            }
        } else {
            throw new Error("startDrag without posRef");
        }
    }

    public move = (e: React.MouseEvent<SVGElement>): void => {
        const { onDragMove } = this.props;
        if (!this.state.dragging) {
            return;
        }
        if (this.posRef.current) {
            this.setState({
                drag: this.applyConstraint(relativeCoords(e, this.posRef.current)),
            }, () => {
                if (onDragMove) {
                    const constrained = this.applyConstraint(this.state.drag)
                    onDragMove(constrained.x, constrained.y);
                }
            })
        } else {
            throw new Error("move without posRef");
        }
    }

    public stopDrag = (): void => {
        const { parentOnDrop } = this.props;
        const { dragging, drag } = this.state;
        if (!dragging) {
            return;
        }
        this.setState({
            dragging: false,
        })
        if (parentOnDrop) {
            parentOnDrop(drag.x, drag.y);
        }
    }

    public handleKeyDown = (e: React.KeyboardEvent<SVGElement>): void => {
        const update = (fn: ModifyCoords) => {
            const { x, y, onDragMove } = this.props;
            const newCoords = fn(x, y);
            const constrained = this.applyConstraint(newCoords);
            if (onDragMove) {
                onDragMove(constrained.x, constrained.y);
            }
        }
        handleKeyEvent(e, update);
    }

    public renderCommon(x: number, y: number): JSX.Element {
        const { imageWidth } = this.props;
        const scale = imageWidth === undefined ? 1 : imageWidth / 128;
        // empty zero-size <rect> as relative position reference
        return (
            <g>
                <rect
                    style={{ visibility: "hidden" }}
                    ref={this.posRef}
                    x={0} y={0} width={0} height={0}
                />
                <Handle scale={scale} x={x + .5} y={y + .5} withCross={this.props.withCross}
                    focusRef={this.focusRef}
                    onMouseUp={this.stopDrag}
                    onMouseMove={this.move}
                    onMouseDown={this.startDrag}
                    onKeyDown={this.handleKeyDown}
                    tabIndex={0}
                />
            </g>
        );
    }

    public renderDragging(): JSX.Element {
        const { x, y } = this.state.drag;
        return this.renderCommon(x, y);
    }

    public render(): JSX.Element {
        const { x, y } = this.props;
        // either render from state (when dragging) or from props
        if (this.state.dragging) {
            return this.renderDragging();
        } else {
            return this.renderCommon(x, y);
        }
    }
}

export default DraggableHandle;
