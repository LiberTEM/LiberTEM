export type ModifyCoords = ((x: number, y: number) => ({ x: number, y: number }));

/**
 * Call this function whenever a keyboard event happens. The keyboard event is then translated
 * to a coordinate transformation, which is passed to the update callback. The update callback is
 * only called if relevant keys were pressed (example: arrow keys).
 * 
 * @param e keyboard event
 * @param update a callback to handle coordinate updates
 */
export const handleKeyEvent = (e: React.KeyboardEvent<SVGElement>, update: (fn: ModifyCoords) => void): void => {
    let delta = 1;
    if (e.shiftKey) {
        delta = 10;
    }
    switch (e.key) {
        case "ArrowUp":
            update((x: number, y: number) => ({ x, y: y - delta }));
            break;
        case "ArrowDown":
            update((x: number, y: number) => ({ x, y: y + delta }));
            break;
        case "ArrowLeft":
            update((x: number, y: number) => ({ x: x - delta, y }));
            break;
        case "ArrowRight":
            update((x: number, y: number) => ({ x: x + delta, y }));
            break;
        default:
            return;
    }
    e.preventDefault();
}
