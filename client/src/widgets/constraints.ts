import { Point2D } from "../basicTypes";

type Constraint2D = (point: Point2D) => Point2D;


export const inRectConstraint = (width: number, height: number): Constraint2D => (p: Point2D): Point2D => ({
    x: Math.max(0, Math.min(width - 1, p.x)),
    y: Math.max(0, Math.min(height - 1, p.y)),
});

export const dist = (cx: number, cy: number, x: number, y: number): number => {
    const dx = cx - x;
    const dy = cy - y;
    return Math.sqrt(dx * dx + dy * dy);
}

export const cbToRadius = (cx: number, cy: number, cb: ((r: number) => void) | undefined) => (x: number, y: number): (number | void) => cb && cb(dist(cx, cy, x, y))

export const keepOnCY = (cy: number) => (p: Point2D): Point2D => ({
    x: p.x,
    y: cy,
});

export const keepXLargerThan = (otherX: number) => (p: Point2D): Point2D => ({
    x: otherX > p.x ? otherX : p.x,
    y: p.y,
});

export const keepXSmallerThan = (otherX: number) => (p: Point2D): Point2D => ({
    x: otherX < p.x ? otherX : p.x,
    y: p.y,
});

export const riConstraint = (outerPos: number, cy: number) => (p: Point2D): Point2D => (
    keepXLargerThan(outerPos)(keepOnCY(cy)(p))
)

export const roConstraints = (innerPos: number, cy: number) => (p: Point2D): Point2D => (
    keepXSmallerThan(innerPos)(keepOnCY(cy)(p))
);