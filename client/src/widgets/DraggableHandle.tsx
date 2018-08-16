import * as React from "react";

export type HandleProps = {
    x: number,
    y: number,
    scale: number,
    withCross?: boolean,
} & React.SVGProps<SVGCircleElement>;

const Handle: React.SFC<HandleProps> = ({ scale, x, y, withCross, ...args }) => {
    const r = 3;
    // scaleMatrix is needed to set the origin of the scale
    const scaleMatrix = `matrix(${scale}, 0, 0, ${scale}, ${x - scale * x}, ${y - scale * y})`;
    const style: React.CSSProperties = { transform: scaleMatrix, stroke: "red", strokeWidth: 1, fill: "transparent" };
    const crossSpec = `
        M${x - r / 2} ${y} L ${x + r / 2} ${y}
        M${x} ${y - r / 2} L ${x} ${y + r / 2}
    `;
    const cross = withCross ? <path d={crossSpec} style={style} /> : null;
    return (
        <g {...args}>
            <circle cx={x} cy={y} r={r} style={style} />
            {cross}
        </g>
    )
}

export interface DraggableHandleProps {
    x: number,
    y: number,
    withCross?: boolean,
    imageWidth?: number,
    onDragMove?: (x: number, y: number) => void,
    parentOnDragStart?: (h: DraggableHandle) => void,
    parentOnDrop?: (x: number, y: number) => void,
    constraint?: (p: Point2D) => Point2D,
}

function getScalingFactor(elem: SVGElement): number {
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
}

function relativeCoords(e: React.MouseEvent, parent: SVGElement) {
    const f = getScalingFactor(parent);
    const parentPos = parent.getBoundingClientRect();
    const res = {
        x: (e.pageX - (parentPos.left + window.scrollX)) / f,
        y: (e.pageY - (parentPos.top + window.scrollY)) / f,
    }
    return res;
}

/**
 * stateful draggable handle, to be used as part of <svg/>
 */
export class DraggableHandle extends React.Component<DraggableHandleProps> {
    public posRef: SVGElement | null;

    public state = {
        dragging: false,
        drag: { x: 0, y: 0 },
    }

    // mousemove event from outside (delegated from surrounding element)
    public externalMouseMove = (e: React.MouseEvent<SVGElement>): void => {
        this.move(e);
    }

    // mouseleave event from outside (delegated from surrounding element)
    public externalLeave = (e: React.MouseEvent<SVGElement>): void => {
        this.stopDrag(e);
    }

    // mouseup event from outside (delegated from surrounding element)
    public externalMouseUp = (e: React.MouseEvent<SVGElement>): void => {
        this.stopDrag(e);
    }

    public applyConstraint = (p: Point2D) => {
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
        if (this.posRef) {
            this.setState({
                dragging: true,
                drag: this.applyConstraint(relativeCoords(e, this.posRef)),
            });
            if (parentOnDragStart) {
                parentOnDragStart(this);
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
        if (this.posRef) {
            this.setState({
                drag: this.applyConstraint(relativeCoords(e, this.posRef)),
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

    public stopDrag = (e: React.MouseEvent<SVGElement>): void => {
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

    public renderCommon(x: number, y: number) {
        const { imageWidth } = this.props;
        const scale = imageWidth === undefined ? 1 : imageWidth / 128;
        // empty zero-size <rect> as relative position reference
        return (
            <g>
                <rect
                    style={{ visibility: "hidden" }}
                    ref={e => this.posRef = e}
                    x={0} y={0} width={0} height={0}
                />
                <Handle scale={scale} x={x} y={y} withCross={this.props.withCross}
                    onMouseUp={this.stopDrag}
                    onMouseMove={this.move}
                    onMouseDown={this.startDrag}
                />
            </g>
        );
    }

    public renderDragging() {
        const { x, y } = this.state.drag;
        return this.renderCommon(x, y);
    }

    public render() {
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