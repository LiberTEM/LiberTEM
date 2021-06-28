import { Point2D } from "../basicTypes";

// from https://stackoverflow.com/a/45100420/540644
export const getPathArc = (center: { x: number, y: number }, start: number, end: number, radius: number): string => {
    if (end === start) { end += 360; }
    let degree = end - start;
    degree = degree < 0 ? (degree + 360) : degree;
    const points = [];
    points.push(getLocationFromAngle(start, radius, center));
    points.push(getLocationFromAngle(start + degree / 3, radius, center));
    points.push(getLocationFromAngle(start + degree * 2 / 3, radius, center));
    points.push(getLocationFromAngle(end, radius, center));
    return getCirclePath(points, radius, (degree < 180) ? 0 : 1);
}

const getCirclePath = (points: Array<{ x: number, y: number }>, radius: number, clockWise: 0 | 1): string => (
    ['M', points[0].x, points[0].y,
        'A', radius, radius, 0, 0, clockWise, points[1].x, points[1].y,
        'A', radius, radius, 0, 0, clockWise, points[2].x, points[2].y,
        'A', radius, radius, 0, 0, clockWise, points[3].x, points[3].y
    ].join(' ')
);

const getLocationFromAngle = (degree: number, radius: number, center: { x: number, y: number }): Point2D => {
    const radian = (degree * Math.PI) / 180;
    return {
        x: Math.cos(radian) * radius + center.x,
        y: Math.sin(radian) * radius + center.y
    }
}
